# Startup and loading

Making startup and scene loading faster by running independent work in parallel, and overlapping CPU work with device
creation. Shared rules and measurement setup: [working_process.md](working_process.md). Scenes and their files:
[scenes.md](scenes.md). GPU texture compression, part of the upload: [texture_compression.md](texture_compression.md).

## Done so far

Shader compilation, which was most of the visible startup cost, is no longer on the critical path:

| Commit | Change |
| --- | --- |
| vk-renderer `51c5eff`, brutus-ng `99a4774` | On-disk SPIR-V cache, `--shader-cache <dir>` (default `shadercache/`, git-ignored) and `--no-shader-cache` |
| vk-renderer `3e2adbf`, brutus-ng `177df3f` | Cache files start with a 64-bit FNV-1a of the SPIR-V: damaged or old-format entries are recompiled and rewritten |
| brutus-ng `1676a2b` | All shaders on `#version 460` (the last GLSL version): Release SPIR-V byte-identical to 450 |
| vk-renderer `59d342d`, brutus-ng `1d9cb80` | `ShaderCompiler::prewarm()` on the rebuild thread, and the rebuild thread polls every 100 ms instead of 1 s |
| brutus-ng `9e65cd0` | Profiling scope on texture surface copies |

- **Cache key**: FNV-1a 64 of the glslang version (`GLSLANG_VERSION_*`), the SPIRV-Tools version string, target
  environment, optimization level, debug info flag, shader stage and the preprocessed source. Debug and Release get
  separate entries (debug info changes the SPIR-V). `shaderc_get_spv_version` is not a compiler version: it reports the
  SPIR-V spec version, which compiler updates don't change.
- **Cost of a key**: every lookup preprocesses the shader, and glslang's first preprocess or compile per process builds
  its built-in symbol tables (~90 ms in Release) under a global mutex: every other compiling thread waits for it. The
  tables are built per `#version` (and client, profile, source language), so all shaders must share one version.
  `prewarm()` builds them with a one line `#version 460` shader; it must match the shaders' version or it builds a
  table nobody uses.
- **1 s poll**: the rebuild thread started at `PipelineManager` construction, found no shaders, then slept 1 s while
  `Assets::load` waited in `wait_ready()`. Polling every 100 ms: crowd Release warm startup 3.05 s -> 2.24 s, Debug
  4.8 s -> 3.9 s (whole process, `--capture` runs).

## Current state

Crowd, Release, warm shader cache (Optick capture, `util/optick_report.py`), 1,255 ms of startup:

| Step | Time | Notes |
| --- | --- | --- |
| `Device::Device` | 571 ms | SDL init, window, instance, device: now the largest single item |
| `Assets::load` (`player.glb`) | 455 ms | One mesh's `prepare_mesh` takes 413 ms: LODs 227, shadow LODs 121, meshlets 63 |
| `Assets::load` (`sabrina.glb`) | 191 ms | Meshes 115 ms, textures decoded in 86 ms, texture upload 36 ms |

In `player.glb`, one TBB worker grinds through the big mesh while the others are idle. Bistro (Release, DDS textures):
`Assets::load` 4.4-5.6 s, of which `prepare_meshes` 3.5-4.0 s over 551 meshes, `decompress_textures` 0.7-2.4 s
(reading 2.05 GB of DDS, see Experiments) and `upload_textures` ~580 ms.

**The prewarm on the rebuild thread doesn't pay off yet.** It cuts the first shader batch from 112 ms to 8 ms, but
alternating runs gave Release +45 ms warm and +90 ms cold for the whole process (Debug -73 ms): it delays the
rebuild thread's first pass, and the shaders registered after it then wait for the next 100 ms poll (not confirmed).
It's only worth keeping once it runs before `Device` (see Plan).

## Plan

### Stages

Startup and `Assets::load` broken into stages by what they need. Nothing in stages 1 to 4 depends on the device: the
only options they read (`--no-dds`, `--texture-compression`) come from the command line.

1. **glTF parsing**: JSON or GLB, the list of meshes, materials and textures to load, the flattened node hierarchy.
   Reading a GLB reads its embedded binary chunk too. Today `parse()` also loads every external buffer, serially.
2. **File I/O**: external buffers and external images.
   - `player.glb` (10.3 MB) and `sabrina.glb` (18.7 MB) embed everything, their 20 PNGs included (`bufferView`
     images): `load()` returns a view into the GLB buffer, no I/O and no copy. The crowd has no stage 2.
   - Bistro: `bistro.bin` (95.7 MB, read during parsing today) and 343 DDS files (2.05 GB); its 343 PNGs are only read
     with `--no-dds`.
3. **Mesh processing**, per mesh:
   - Vertex prep: 16-bit index widening, MikkTSpace tangents (only meshes without `TANGENT`: 2 of `sabrina.glb`'s 5
     primitives, none in Bistro), vertex repack. Both LOD chains depend on it.
   - Main LOD chain (simplify with normals) and shadow LOD chain (positions only, then remapped to shadow indices) are
     independent: they only read the vertices and LOD 0 indices.
   - Within a chain, every LOD simplifies from LOD 0 (`generate_lods`): all LODs are independent.
   - Meshlets per chain, and per LOD. LOD 0 meshlets only need vertex prep. `build_meshlets` appends to shared vectors
     with absolute offsets: parallel LODs need local outputs, merged and rebased in today's order to keep the GPU data
     byte-identical.
   - Bounds (`meshopt_computeSphereBounds`, in `upload_mesh` today) can move here.
4. **Image decoding**: PNG to RGBA8. Embedded PNGs depend on stage 1 only, external ones on stage 2. DDS skip this
   stage.
5. **Shader preparation**: prewarm, cache lookups, compiles. No device needed, but the shader set is: meshlet pipelines
   are only registered with `mesh_shader_support` (`scene_renderer.cpp`), the compressor's pipelines by the `Assets`
   constructor. Prewarm before `Device`, then either compile once the properties are known or compile the possible set
   speculatively (~4 extra shaders). glslang frees its tables when the last shaderc compiler is destroyed: an early
   prewarm needs the `ShaderCompiler` that `PipelineManager` will use, created before `Device` and handed over.
6. **GPU work**, after `Device`: pipelines, buffer and texture creation, staging copies (mapped Vulkan memory), uploads,
   BC compression (waits for the compressor pipelines: `wait_ready()` in `upload_textures`), mip blits, bindless
   registration. Then CPU work that needs handles: `gpu_material` from glTF materials (texture handles), scene
   instances (mesh handles).

```
1 parse ──┬─► 2 I/O (external .bin, DDS, PNG) ──┬─► 3 meshes (need buffers)
          │                                      └─► 4 PNG decode (external)
          ├─► 3 meshes (GLB: buffers already in memory)
          └─► 4 PNG decode (embedded)
5 shader prep (prewarm needs nothing; the shader set needs device properties)
1-5 ──► 6 GPU: device -> pipelines -> uploads -> compression -> materials and instances
```

### Prepare and commit

Split `load_job` in two:

- `prepare()`: stages 1 to 4, CPU only, no `Device` or `BindlessManager` (already true of `parse`, `prepare_meshes` and
  `decompress_textures`; the upload jobs created in `load_job`'s constructor move to commit). Its output is plain data:
  parsed glTF, mesh data, decoded surfaces or DDS bytes.
- `commit()`: stage 6, **serially and in input order** on the main thread. This keeps the rules we rely on without
  locks: the command pool used by uploads isn't thread safe (`asset.cpp`, `run()`), `BindlessManager` has no locking,
  `Assets::add_content` asserts handles arrive in order, the compressor shares scratch textures, and a fixed handle
  order keeps the benchmark bit-identical.

Prepared data is the queue of pending uploads: no recorded upload commands are needed before the device exists (buffer
sizes and handles aren't known before commit anyway).

### Order of work

1. Prepare/commit split, behavior unchanged (pixel identical, same timings).
2. Start `prepare()` for every asset right after argument parsing, before `Device`; commit after it, waiting on each
   prepared asset in order. Crowd prepare work adds up to ~559 ms even serially, against 571 ms of device creation.
3. Prewarm before `Device` (see stage 5). Without it, commit waits ~190 ms for the compressor pipelines (prewarm,
   compile, pipeline creation) once asset work is hidden.
4. Parallel LODs, chains and meshlets (stage 3). Matters most for Bistro, whose mesh work (3.5-4 s) is far longer than
   device creation.
5. Dedicated reader threads for stage 2 (see Open questions).
6. Later: an on-disk cache for LODs and meshlets, the same pattern as the shader cache.

Estimates for the crowd from one capture, not measurements: 1,255 ms today, ~860 ms with step 2, ~700-750 ms with step
3. After that, device creation dominates.

### Open questions

**Task system.** Plain TBB looks sufficient for now:

- Nested `parallel_for` / `parallel_invoke` for stage 3: they share the existing pool and scheduler.
- `tbb::task_arena::enqueue` (with a counter or task group to wait on) is the documented way for threads outside TBB,
  like readers or the main thread, to submit work.
- `tbb::parallel_pipeline` (serial-in-order input, parallel prepare, serial-in-order commit) was the first idea, with its
  token count as the budget. It doesn't fit once commit waits for the device, since a filter would block on it.
- `tbb::flow::graph` or our own task graph only becomes worth it for priorities, cancellation or streaming during
  frames.

**Budget.** Prepared assets can't commit before the device exists, so the peak is every prepared asset at once. A token
count still caps how much prepares ahead (the rest waits for the device). Not needed for the crowd (156 MB of assets
in total) or Bistro (one asset, whose decoded images already stay in memory until `upload_textures` today). A byte
budget would weigh each file before admitting it.

**Reader threads.** Blocking reads in TBB tasks occupy workers that mesh processing needs. Options:

- A few dedicated `std::jthread`s outside TBB, each taking the next file from an atomic index, handing PNGs to TBB
  (`task_arena::enqueue`) and marking DDS ready. About 4 threads: reads scale little past that (Experiments).
- It relies on oversubscription only while readers copy from the OS cache; waiting on the disk costs no CPU.
- Above-normal priority for readers keeps requests in flight; only if a cold profile shows them waiting for a core.
- Readers need profiler registration (`tbb_observer` only covers TBB threads).
- True async I/O isn't in the standard library (C++26 `std::execution` has no I/O). On Windows: Win32 thread pool I/O
  (`CreateThreadpoolIo`, overlapped `ReadFile`, callback as continuation) or IoRing (Windows 11); Asio for portability.
  DirectStorage is D3D12-centric. Not needed until cold loads prove I/O bound: overlapped reads of cached files often
  complete synchronously anyway.

**Other questions:**

- Is SDL safe for surface decoding while `SDL_Init` runs inside `Device` on another thread? Calling `SDL_Init( 0 )`
  before starting prepare work is cheap insurance; unverified.
- Errors: a task group must be waited on even if `Device` throws; a bad asset now reports at commit, after the window
  opened.
- On low core counts, prepare work could slow device creation.

## Experiments

### Bistro DDS reads

Release, default options: Bistro takes the DDS path (no `copy_surface`, i.e. no decoded texture reached the upload).
The `decompress_tex` scope wraps all of `decompress_texture`, including DDS reads and the 348 unused images (343 PNGs,
5 DDS) that return immediately.

Reading the 343 DDS files (2.05 GB, OS cache warm, Python, unbuffered whole-file reads), Samsung 970 EVO 1TB (NVMe,
PCIe 3):

| Threads | Median | GB/s |
| --- | --- | --- |
| 1 | 1,045 ms | 1.96 |
| 2 | 760 ms | 2.70 |
| 3 | 542 ms | 3.79 |
| 4 | 497 ms | 4.13 |
| 6 | 464 ms | 4.43 |
| 8 | 461 ms | 4.46 |
| 16 | 440 ms | 4.67 |

Above ~3.4 GB/s this is memory speed through the OS cache, not the drive. Cold numbers need the cache cleared (reboot);
not measured.

## Findings

- **`decompress_tex` is misleading**: it also covers DDS reads and unused images. Return early for unused images before
  the scope and use separate scopes (e.g. `read_dds`, `decode_image`) so a fallback to PNG shows up.
- **`decompress_textures` time isn't read time**: it shares TBB workers with `prepare_meshes`, and varied 0.7-2.4 s
  over 3 cached runs. Per-file reads: median 34 ms on a run that was likely cold, 10-13 ms cached, against ~3 ms for a
  plain single-threaded read.
- **`load()` zeroes every file buffer**: `std::make_unique<std::byte[]>` value-initializes, 2 GB for Bistro;
  `make_unique_for_overwrite` skips it. Not measured.
- **Optick drops events** in these captures: 229-328 of Bistro's 338 DDS scopes were recorded depending on the run.
  Treat scope counts as approximate.
- Profiles: Optick saves `startup_debug.opt` / `startup_release.opt` at every startup (overwritten), read with
  `util/optick_report.py`. Compare startup timings with alternating runs of both builds and medians: single runs vary
  by ~0.1 s.
