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

Assets are then prepared while the device is created (steps 2 and 3 of the order of work):

| Commit | Change |
| --- | --- |
| vk-renderer `33c2bcd`, brutus-ng `6f32f67` | `PipelineManager::is_ready()`: checks without waiting, false if a pipeline failed |
| brutus-ng `dd35d32` | `texture_compressor` in its own files, created by `main` and passed to `Assets` |
| brutus-ng `5a5a796` | Terminate handler printing uncaught exceptions, without unwinding (see Findings) |
| brutus-ng `5e8aaef` | The compressor's max texture size is a constructor parameter, larger textures throw |
| brutus-ng `72295d9` | `load_job` split into `prepare()` and `commit()`, see Prepare and commit |
| brutus-ng `7360d78` | `Assets::load()` before `Device`, then `attach()` and `wait_ready()` |

`main` now calls `SDL_Init( 0 )`, starts every load, creates the device and the rest, waits for the pipelines when
compressing (`attach()` asserts the compressor is ready), then `attach()` and `wait_ready()`. The device lives in a
`std::optional` declared before `Assets`, so it's still destroyed after the assets' buffers.

Release, 5 alternating runs against the previous commit, medians in ms (`scratch/step3`):

| | Crowd before | Crowd after | Bistro before | Bistro after |
| --- | --- | --- | --- | --- |
| Startup | 1,343 | 959 | 5,333 | 4,766 |
| `create_device` | 613 | 730 | 599 | 1,515 |
| `prepare` (all loads) | 582 | 770 | 3,667 | 3,843 |
| `commit` | 90 | 85 | 810 | 675 |
| `Assets::wait_ready` | | 104 | | 2,819 |

- Images are pixel identical (crowd, Bistro, crowd without compression).
- **Device creation slows down next to the prepare work**: on Bistro it more than doubles, as mesh work keeps all 16
  hardware threads busy and the main thread competes with the TBB workers for a core. The crowd's prepare work slows
  the same way (582 -> 770 ms). See step 9.
- Bistro still waits 2.8 s in `wait_ready()` for its meshes: hiding the device was worth ~0.6 s, as estimated.
- Memory: the crowd's peak working set goes from 766 to 876 MB, private bytes +36 MB (within build noise, see Findings).
  Both crowd assets are now prepared at once, where the first one's CPU data used to be freed before the second
  started. Bistro is within noise.
- Both builds ran slower than earlier sessions (the baseline crowd took 1,169 ms in the previous step's runs): compare
  only alternating runs of one session.

Then device creation and mesh processing got cheaper:

| Commit | Change |
| --- | --- |
| vk-renderer `3ec5c6e`, brutus-ng `8e53f95` | `VK_EXT_debug_utils` always required, without a `vkb::SystemInfo` scan; `debug_labels_support` removed |
| brutus-ng `5eeb1f3` | Each LOD simplified from the previous one, errors tracked like niagara and Godot |
| vk-renderer `eaee371`, `4c005d2` | vk-bootstrap only queries the extensions of requested layers; profiling scopes in `Device` |

**Debug utils** (`scratch/devicefix`): `Device` made a `vkb::SystemInfo` only to check for `VK_EXT_debug_utils`, and
`InstanceBuilder::build()` makes its own. Each one enumerates every layer's extensions, and each loader enumeration
reparses every layer manifest: ~140-160 ms each in Release on this machine, with 18 layers registered (see
Findings). The Khronos loader implements the extension itself, so it's now required. Medians over 5 alternating
Release runs:

| | Crowd before | Crowd after | Bistro before | Bistro after |
| --- | --- | --- | --- | --- |
| `create_device` | 672 | 515 | 1,242 | 933 |
| Startup | 889 | 732 | 4,016 | 4,323 |

Bistro got slower in all 5 rounds, from scheduling rather than added work: `main` reaches `wait_ready()` ~330 ms
earlier and joins the mesh work there (`task_group::wait()` runs tasks), which changes how TBB splits the meshes. The
mesh that finished last started at 1,916 ms instead of 1,521 ms (median), and `main` more often took one of the huge
meshes itself (`scratch/devicefix/tail.py`). Chained LODs below shortened those meshes enough to make this moot.

**Chained LODs** (`scratch/lodchain2`): as meshoptimizer's README recommends, each LOD simplifies the previous one
instead of LOD 0, with `meshopt_SimplifySparse` from LOD 2 so a coarse LOD's cost scales with its own indices, and
absolute errors with the same limits as before. A step's error is measured against its source LOD, so each LOD records
`max(1.5 x previous, step error)` and every step gets the LOD's full limit. Medians over 5 alternating Release runs, GPU
from Nsight traces (`util/gputrace.py`):

| | Crowd before | Crowd after | Bistro before | Bistro after |
| --- | --- | --- | --- | --- |
| Startup | 729 ms | 680 ms | 3,632 ms | 2,530 ms |
| `prepare` | 667 ms | 395 ms | 3,020 ms | 1,910 ms |
| `prepare_mesh` CPU time, all threads | 1.31 s | 0.69 s | 29.8 s | 14.4 s |
| Longest `prepare_mesh` | 457 ms | 216 ms | 1,654 ms | 845 ms |
| `lod_generation` + `shadow_lod_generation` CPU time | 1.08 s | 0.42 s | 22.5 s | 7.6 s |
| `meshlet_generation` CPU time | 0.20 s | 0.25 s | 6.9 s | 6.6 s |
| GPU frame | 4.33 ms | 4.28 ms | 2.40 ms | 2.41 ms |

- The crowd's startup is still bound by device creation: its prepare work is 270 ms shorter, startup within noise.
- Meshlet generation is now the largest share of Bistro's mesh work.
- LODs differ: 9% of the crowd's benchmark pixels change (1.2% on Bistro), without visible artifacts when moving
  around. Distant characters show small silhouette and shading differences.
- Memory within build variance: peak working set crowd +36 MB, Bistro -5 MB.
- **Summing step errors selected much denser LODs**: the first version summed them (a strict bound) and only allowed
  each step what was left of the LOD's limit. Crowd GPU frame time went from 4.45 to 5.32 ms, LODs 2-5 costing 35-55%
  more. niagara, Godot's importer and meshoptimizer's `clusterlod.h` all use a max instead (Findings).

**Layer extensions** (`scratch/vkbfix`): the `SystemInfo` made by `InstanceBuilder::build()` queried the extensions of
all 18 installed layers, enabled or not, and merged them into one list. Besides the cost, that let an extension from a
layer that isn't enabled pass vk-bootstrap's check and fail later in `vkCreateInstance` (`debug_utils_available` had
the same flaw). `build()` now passes the layers it will enable (user layers, plus validation if enabled or requested)
and `SystemInfo` only queries those (none in Release, validation in Debug). `get_system_info()` takes them as an
optional parameter, so existing calls compile. The same change was sent upstream as a PR against `main` (branch
`systeminfo_explicit_layers` of our fork, layers as `std::string`). Medians over 5 alternating runs, crowd:

| | Debug before | Debug after | Release before | Release after |
| --- | --- | --- | --- | --- |
| `create_device` | 515 ms | 251 ms | 466 ms | 327 ms |
| Startup | 1,401 ms | 1,099 ms | 681 ms | 558 ms |

- Bistro's `create_device` goes from 748 to 493 ms in Release, its startup is within noise (bound by meshes).
- Debug saves more because it uses vcpkg's debug `vulkan-1.dll` (unoptimized): ~7 ms per loader call against ~4 ms
  in Release. Debug's `vk_instance` goes from 328 to 81 ms, validation still loads (`VK_LOADER_DEBUG=layer`).
- The crowd's `create_device` is now `SDL_CreateWindow` 178 ms, `vk_device` 63, `vk_instance` 52, `SDL_init` 27,
  `vk_physical_device` 7, and it's shorter than the crowd's `prepare` (386 ms): device creation is off the critical
  path as long as there are meshes to prepare or textures to decompress. Memory within noise.
- Instance creation is closed. `build()` still enumerates the layer list when no layer is requested, and in Debug
  queries the validation layer's extensions although everything we request is in the base list: 2 loader calls each.
  Removing those needs `build()` to do its own queries instead of going through `SystemInfo` (whose public fields
  callers rely on), or our own instance creation. Neither is worth a few ms.
- `SDL_CreateWindow` is about half Vulkan loading and half color profile lookups (Findings).

## Profile before the loading work

Crowd, Release, warm shader cache (Optick capture, `util/optick_report.py`), 1,255 ms of startup:

| Step | Time | Notes |
| --- | --- | --- |
| `Device::Device` | 571 ms | SDL init, window, instance, device: now the largest single item |
| `Assets::load` (`player.glb`) | 455 ms | One mesh's `prepare_mesh` takes 413 ms: LODs 227, shadow LODs 121, meshlets 63 |
| `Assets::load` (`sabrina.glb`) | 191 ms | Meshes 115 ms, textures decoded in 86 ms, texture upload 36 ms |

In `player.glb`, one TBB worker grinds through the big mesh while the others are idle.

Bistro, Release, DDS textures, 5,605 ms of startup (one capture, analyzed in Experiments):

| Step | Start | Time | Notes |
| --- | --- | --- | --- |
| `create_device` | 0 ms | 611 ms | |
| `parse` | 648 ms | 110 ms | Includes reading `bistro.bin` (95.7 MB) |
| `decompress_textures` | 758 ms | 717 ms | Reading 2.05 GB of DDS, alongside the meshes; varied 0.7-2.4 s over runs |
| `prepare_meshes` | 758 ms | 4,004 ms | 551 meshes, 38.9 s of CPU time; the second half is a tail of a few big meshes |
| `upload_meshes` | 4,762 ms | 152 ms | |
| `upload_textures` | 4,915 ms | 522 ms | Only starts after every mesh is prepared and uploaded |

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
   - Within a chain, every LOD simplifies the previous one (`generate_lods`, since `5eeb1f3`): a chain is serial. Until
     then every LOD simplified LOD 0 and they were independent, at twice the total CPU time.
   - Meshlets per chain, and per LOD: each LOD's meshlets only need that LOD's indices, LOD 0's only vertex prep.
     `build_meshlets` appends to shared vectors with absolute offsets: parallel LODs need local outputs, merged and
     rebased in today's order to keep the GPU data byte-identical.
   - Bounds (`meshopt_computeSphereBounds`, in `upload_mesh` today) can move here.
4. **Image decoding**: PNG to RGBA8. Embedded PNGs depend on stage 1 only, external ones on stage 2. DDS skip this
   stage.
5. **Shader preparation**: prewarm, cache lookups, compiles. No device needed, but the shader set is: meshlet pipelines
   are only registered with `mesh_shader_support` (`scene_renderer.cpp`), the compressor's pipelines by its
   constructor. Prewarm before `Device`, then either compile once the properties are known or compile the possible set
   speculatively (~4 extra shaders). glslang frees its tables when the last shaderc compiler is destroyed: an early
   prewarm needs the `ShaderCompiler` that `PipelineManager` will use, created before `Device` and handed over.
6. **GPU work**, after `Device`: pipelines, buffer and texture creation, staging copies (mapped Vulkan memory), uploads,
   BC compression (needs the compressor pipelines: `main` waits for them before `Assets::attach()`), mip blits, bindless
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

Done (`72295d9`, `7360d78`). `load_job` is split in two:

- `prepare()`: stages 1 to 4 (`parse`, then `prepare_meshes` and `decompress_textures` in parallel), CPU only, no
  `Device` or `BindlessManager`. Its output is plain data: parsed glTF, mesh data, decoded surfaces or DDS bytes.
- `commit()`: stage 6, **serially and in input order** on the main thread. It builds a `gpu_state` with the device,
  bindless manager, index buffer, compressor and both upload jobs (so staging buffers only exist during commit), then
  uploads meshes, uploads textures and registers everything. This keeps the rules we rely on without locks: the command
  pool used by uploads isn't thread safe, `BindlessManager` has no locking, `Assets::add_content` asserts handles
  arrive in order, the compressor shares scratch textures, and a fixed handle order keeps the benchmark bit-identical.

Prepared data is the queue of pending uploads: no recorded upload commands are needed before the device exists (buffer
sizes and handles aren't known before commit anyway).

`Assets` on top of it (`asset.h`), modeled on Granite's asset manager (device-side instantiator attached later) and
Godot's threaded loads (request, then a blocking get):

- `Assets( max_indices, options )` needs no device.
- `load( path )` returns a `scene_handle` (an index) and queues `prepare()` with
  `tbb::this_task_arena::enqueue( f, _prepare_tasks )`, the documented way for a thread outside TBB to submit work.
- `attach( device, bindless_manager, compressor )` creates the GPU side, including the index buffer. A compressor must
  be ready: it asserts `PipelineManager::is_ready()`.
- `wait_ready()` waits on the task group (the main thread helps with any TBB task meanwhile), then commits every
  prepared load in order and frees each job's CPU data after its commit. It rethrows the first load error.
- `get( handle )` asserts the load was committed. Scenes live in a `std::deque`, so references stay valid as more
  loads commit.
- Not copyable or movable: running prepare tasks are registered with the `tbb::task_group` member. The destructor
  cancels and waits for loads that were never waited for (only after an error).

One gate for every load is enough at startup, where everything is waited for anyway. Per-load waits (a task group per
slot, `get()` waiting instead of asserting) can come later without changing call sites.

### Order of work

1. Largest meshes first: **tried, kept on branch `mesh-largest-first`, not merged.** It removes Bistro's tail but the
   mesh phase only drops 6%, and startup is within noise: the big meshes get much slower when they all run at once
   (Experiments).
2. Prepare/commit split, behavior unchanged: **done** (`72295d9`), pixel identical, same load times.
3. Start `prepare()` for every asset before `Device`, commit after it in order: **done** (`7360d78`). Startup crowd
   1,343 -> 959 ms, Bistro 5,333 -> 4,766 ms (Done so far).
4. Prewarm before `Device` (see stage 5). Without it, commit waits ~190 ms for the compressor pipelines (prewarm,
   compile, pipeline creation) once asset work is hidden.
5. Commit textures as soon as they're prepared, while meshes are still processed, then meshes. Texture and mesh
   handles are separate kinds, so their order stays deterministic; materials need both and come last. Bistro's
   textures are read 1.5 s into startup but uploaded at the very end (522 ms). Also worth trying: two staging buffers,
   so the copy into one overlaps the GPU transfer from the other (2 GB through 128 MB is ~16 rounds of copy, submit,
   wait).
6. Smaller tasks within a mesh (stage 3): after vertex prep, the main and shadow LOD chains as two tasks, and each
   LOD's meshlets as soon as that LOD exists. A single continuation runs inline in the task that finishes its input
   (scheduler bypass: earliest start, warm cache, no queueing); only fan-out spawns tasks. A per-mesh atomic counter
   lets the last task to finish run the merge (concatenating LOD outputs in a fixed order) inline. This shortens the
   tail that decides Bistro's end and stops starving the DDS reads, without reducing total work. Chained LODs (since
   `5eeb1f3`) already halved the longest meshes.
7. Dedicated reader threads for stage 2 (see Open questions).
8. Later: an on-disk cache for LODs and meshlets, the same pattern as the shader cache. With chained LODs, meshlet
   generation is the largest share of the remaining mesh CPU work, which bounds Bistro once the rest is done.
9. Device creation next to the prepare work: since step 3, `create_device` slows from ~600 ms to 730 ms (crowd) and
   1,515 ms (Bistro). TBB already leaves `main` a slot (the default arena creates hardware threads - 1 workers), so a
   smaller arena isn't the fix. Likely causes: `main` sharing a core with a worker (~1.3-1.4x, see Experiments),
   driver and SDL threads on top of 16 busy threads, and on Bistro the 2 GB of DDS reads and page faults while the
   loader and driver load their DLLs. Device creation isn't on Bistro's critical path, so this only costs the crowd.
   Removing the duplicate layer scan (`3ec5c6e`) took ~160 ms off, and querying only requested layers (`eaee371`)
   another ~140 ms. `SDL_CreateWindow` is now the largest part (178 ms on the crowd).

Estimates were from one capture each, before step 3:

- Crowd: 1,255 ms in that capture, ~860 ms with step 3 (measured: 959 ms, with device creation slowed down), ~700-750
  ms with step 4. After that, device creation dominates.
- Bistro: 5,605 ms in that capture, 5.0-5.1 s median over 5 runs. Step 3 ~0.6 s (measured: 0.57 s), step 5 up to
  ~0.5 s. Step 1 was estimated at 1.1-1.6 s from a replay of measured mesh durations, but measured 0.2 s on the mesh
  phase: the mesh work is limited by core throughput, not by scheduling (Experiments).

### Open questions

**Task system.** Plain TBB looks sufficient for now (step 3 uses `this_task_arena::enqueue` into a `task_group`):

- Nested `parallel_for` / `parallel_invoke` for stage 3: they share the existing pool and scheduler.
- `tbb::task_arena::enqueue` (with a counter or task group to wait on) is the documented way for threads outside TBB,
  like readers or the main thread, to submit work.
- `tbb::parallel_pipeline` (serial-in-order input, parallel prepare, serial-in-order commit) was the first idea, with
  its token count as the budget. It doesn't fit once commit waits for the device, since a filter would block on it.
- `tbb::flow::graph` or our own task graph only becomes worth it for priorities, cancellation or streaming during
  frames.

**Budget.** Prepared assets can't commit before the device exists, so the peak is every prepared asset at once. A token
count still caps how much prepares ahead (the rest waits for the device). Not needed for the crowd (156 MB of assets
in total, +110 MB peak working set since both are prepared at once) or Bistro (one asset, whose decoded images already
stayed in memory until `upload_textures`). A byte budget would weigh each file before admitting it.

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

- SDL decodes surfaces on workers while `SDL_Init( SDL_INIT_VIDEO )` runs inside `Device`. `main` calls
  `SDL_Init( 0 )` before the first load to set up SDL's main thread state; whether that's needed was never verified.
- Errors: a bad asset reports from `wait_ready()`, after the window opened. If `Device` throws first, nothing catches
  it: the terminate handler prints it without unwinding, so `~Assets` never runs.
- Prepare work slows device creation even on 16 hardware threads (step 9); likely worse on low core counts.

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

### Bistro mesh scheduling

From the capture in Profile before the loading work (`scratch/bistro_profile/analyze.py` replays its timeline).
Threads busy with `prepare_mesh` or `decompress_tex`, out of 16:

| Time into the mesh phase | 0-2.0 s | 2.1 s | 2.4 s | 2.7 s | 3.0 s | 3.4 s | 3.8 s | 4.0 s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Busy threads | 16 | 11.9 | 10.5 | 8.0 | 5.0 | 4.2 | 1.9 | 0 |

The longest meshes and when they started:

| Mesh time | Starts at | Ends at | LODs | Shadow LODs | Meshlets |
| --- | --- | --- | --- | --- | --- |
| 2,172 ms | 1,715 ms | 3,887 ms | 1,425 ms | 557 ms | 185 ms |
| 2,115 ms | 1,539 ms | 3,654 ms | 1,205 ms | 588 ms | 320 ms |
| 1,813 ms | 2,191 ms | 4,004 ms | 894 ms | 500 ms | 414 ms |
| 1,657 ms | 1,575 ms | 3,232 ms | 973 ms | 471 ms | 211 ms |
| 1,633 ms | 1,859 ms | 3,492 ms | 649 ms | 423 ms | 554 ms |
| 1,379 ms | 487 ms | 1,866 ms | 628 ms | 534 ms | 215 ms |
| 1,261 ms | 1,678 ms | 2,939 ms | 741 ms | 349 ms | 168 ms |

The other 544 meshes are small (median 9 ms). Replaying the measured durations on 16 threads, greedily:

| Schedule | Mesh phase |
| --- | --- |
| Measured | 4,004 ms |
| Same start order as measured | 3,477 ms (the model doesn't capture TBB's range splitting) |
| Largest first | 2,434 ms |
| Floor: 38.9 s of CPU / 16 threads | 2,434 ms |

The replay assumed each mesh takes the same time whatever runs alongside it. Measuring largest-first ordering showed
that's wrong (next section): the floor above is not reachable on this CPU.

### Largest-first ordering

Branch `mesh-largest-first`: `prepare_meshes` sorts meshes by index count, largest first, and one task per TBB thread
pulls the next mesh from an atomic counter (a `parallel_for` over the sorted list wouldn't keep the order: thieves
steal the far half of the range first). Release, `--capture`, alternating runs against a baseline build, medians:

| | Bistro baseline | Bistro ordered | Crowd baseline | Crowd ordered |
| --- | --- | --- | --- | --- |
| Runs | 5 | 5 | 8 | 8 |
| Startup | 5,075 ms | 5,036 ms | 1,253 ms | 1,222 ms |
| `prepare_meshes` | 3,520 ms | 3,307 ms | 517 ms | 507 ms |
| `prepare_mesh` CPU time, all threads | 34.7 s | 46.2 s | 1.08 s | 1.05 s |
| `decompress_textures` (DDS reads done) | 856 ms | 3,306 ms | 162 ms | 162 ms |
| Peak working set | 1,732 MB | 1,938 MB | 768 MB | 804 MB |

Images are pixel identical. Single runs vary by ~0.1-0.3 s, so the startup differences are within noise. From one
capture of each (`scratch/bistro_profile/analyze.py`):

- **The tail is gone**: all 16 threads stay busy until the end of the mesh phase.
- **The work got slower**: the 7 biggest meshes took 17.7 s of CPU together instead of 10.9 s (+62%), the 544 others
  28.4 s instead of 23.2 s (+23%). The baseline ran its big meshes partly in its tail, with 8 threads or fewer busy on
  8 cores; ordered runs them all at once on 16 hardware threads, two per core. Mostly hyper-threading (next section).
- **DDS reads were starved**: the pulling tasks never hand their thread back to TBB, so the read tasks of the other
  `parallel_invoke` branch waited for meshes to run out. Harmless today (the texture upload waits for meshes anyway)
  but it would block committing textures early (step 5). Fewer pullers than threads, or one TBB task per mesh, would
  fix it.
- **More memory**: +206 MB peak on Bistro, with the big meshes in memory together.

So once every hardware thread is busy, Bistro's mesh phase is bound by core throughput, ~3.3 s here, and scheduling
can't take it much lower. Reducing the work (step 8) is the lever for Bistro. Splitting meshes (step 6) should hit
the same limit there, but still applies to the crowd.

### Thread affinity

Core bound or memory bound? Bistro, Release master build, process affinity set with `start /affinity`
(`scratch/affinity/run.ps1`). Logical processors 2n and 2n+1 share a core on the i7-10700F, so a mask of every other
bit gives one thread per core. 3 runs per configuration in rotating order, medians:

| Configuration | Mask | `prepare_meshes` | `prepare_mesh` CPU time | 7 biggest meshes | Median small mesh |
| --- | --- | --- | --- | --- | --- |
| 4 threads, 4 cores | `0x0055` | 4,643 ms | 15.2 s | 5.0 s | 3.65 ms |
| 4 threads, 2 cores | `0x000F` | 6,567 ms | 21.9 s | 6.9 s | 5.62 ms |
| 8 threads, 8 cores | `0x5555` | 3,026 ms | 17.3 s | 6.0 s | 3.93 ms |
| 8 threads, 4 cores | `0x00FF` | 3,729 ms | 22.8 s | 7.2 s | 5.51 ms |
| 16 threads, 8 cores (default) | `0xFFFF` | 2,861 ms | 27.3 s | 8.3 s | 5.75 ms |

TBB ran meshes on exactly the threads each mask allowed; images are pixel identical. Same work in every run, so CPU
time measures how fast each thread goes:

- **Sharing a core costs the most**: at the same thread count, two threads per core make each thread 1.32x (8
  threads) to 1.44x (4 threads) slower. Hyper-threading still gets 1.4-1.5x more work out of each core.
- **Memory bandwidth isn't the limit**: from 4 to 8 cores, one thread each, each thread is only 14% slower. A
  bandwidth-bound workload would degrade far more when doubling the cores doing it.
- **Some contention across the chip remains**: doubling the active cores costs 14-20% per thread with or without
  hyper-threading. Memory traffic, the shared L3, or lower all-core clocks (65 W part); not separated. xperf's PMC
  counters would split it (core vs reference cycles for clocks, LLC misses for traffic, against ~34 GB/s for the
  installed DDR4-2133).
- **16 threads is still the fastest**: fewer threads would be a loss.

The default runs used 27 s of mesh CPU time, against 35 s for the same code in the largest-first runs a day earlier:
only compare runs made in the same session, alternated.

## Findings

- **Bistro's mesh phase is half tail**: all 16 threads are busy for 2.0 s, then a few big meshes started late keep 1 to
  8 threads busy for another 2.0 s. Removing it barely helps: the tail runs on uncontended cores (Experiments).
- **Thread time isn't a fixed cost**: on 8 cores with 16 hardware threads, a mesh takes far longer when every thread
  is busy. Estimates that replay measured durations on a different schedule overestimate the gain.
- **Mesh work is core bound, not memory bound**: sharing a core slows each thread 1.3-1.45x, while doubling the
  cores costs only 14-20% per thread (Experiments).
- **`lod_generation` dominated the big meshes** before chained LODs: 40-66% of each of Bistro's 7 longest meshes (over
  half for 4 of them), against 26-39% for shadow LODs and 9-34% for meshlets. Chaining cut both LOD scopes by about
  3x; meshlets are now the largest share.
- **Combine chained LOD errors with a max, not a sum**: meshoptimizer's README only says errors of previous levels
  "should be accumulated". The author's projects and Godot do it with a max and give each step the same limit:
  niagara (`src/scene.cpp`) and Godot's importer (`importer_mesh.cpp`) use `max(previous * 1.5, step)`, and
  `clusterlod.h` defaults to `max(previous, step)`. Godot's comment: the step error is usually right as is, and the
  1.5x floor only keeps errors strictly increasing and limits popping distances. A sum overstates the error and
  selects denser LODs (+20% GPU time on the crowd). `meshopt_SimplifyErrorClamped` (meshoptimizer 1.3, we use 1.2)
  clamps attribute error to the position error's scale; niagara found it reduces triangles on complex scenes.
- **Device creation scanned every layer manifest about 40 times** (before `eaee371`): `vkb::SystemInfo` enumerated
  layer properties, then the extensions of each layer, each through `get_vector` (a count call, then a data call), and
  each loader call reparses every manifest. A full `SystemInfo` cost ~140-160 ms in Release, ~4 ms per loader call.
  This machine registers 9 implicit layers (OBS, RenderDoc, Epic, 4 Nsight, Steam overlay and Fossilize) and 9
  explicit ones (Vulkan SDK). Only OBS's is active without an environment variable, but all manifests are parsed.
  `VK_LOADER_LAYERS_DISABLE=~implicit~` turns implicit layers off for one run. Instance-level optional extensions don't
  exist in vk-bootstrap (only `PhysicalDevice::enable_extension_if_present`).
- **`vkEnumerateInstanceExtensionProperties( nullptr )` is all an app needs** to check instance extensions it enables
  without naming a layer: per the spec it returns the implementation's (driver and loader, which implements
  `VK_EXT_debug_utils` and the surfaces) plus those of active implicit layers, matching what `vkCreateInstance` will
  load. A layer's own list only matters for extensions missing from it, and only for layers being enabled. No API
  says which implicit layers are active (`vkEnumerateInstanceLayerProperties` also lists inactive ones), and none is
  needed. Layers without extensions still do their work (they intercept calls).
- **`VK_LAYER_KHRONOS_validation`** provides the instance extensions `VK_EXT_debug_report`, `VK_EXT_debug_utils`,
  `VK_EXT_layer_settings` and `VK_EXT_validation_features`, and the device extensions `VK_EXT_debug_marker`,
  `VK_EXT_validation_cache` and `VK_EXT_tooling_info` (manifests of SDK 1.4.335 and vcpkg 1.4.357). vk-bootstrap
  chains `VkValidationFeaturesEXT` / `VkValidationFlagsEXT` without enabling their extensions (v1.4.305 and v1.4.367),
  which the spec forbids and the layer tolerates. Its test mock's `VulkanMock::add_layer` stores device extensions in
  the instance list, unnoticed because no test uses it.
- **`SDL_CreateWindow` loads Vulkan and the driver** (VTune capture, SDL 3.4 and loader 1.4.357 sources): with
  `SDL_WINDOW_VULKAN`, `WIN_Vulkan_LoadLibrary` loads `vulkan-1.dll` and calls `vkEnumerateInstanceExtensionProperties`
  to check for the surface extensions, which preloads the driver DLL (`loader_preload_icds`). That cost moves rather
  than adds: the preloaded driver stays loaded until `vkDestroyInstance` and `vkCreateInstance` reuses it. The other
  half is `WIN_UpdateWindowICCProfile` (`CreateDCW` + `GetICMProfileW`), called unconditionally at creation and again
  when the window gets focus, with no hint to disable it. Possible later work: create the instance on a worker while
  `main` creates the window (DLL loads still serialize on the Windows loader lock), or ask SDL to query the color
  profile lazily. An Optick scope around an explicit `SDL_Vulkan_LoadLibrary` before `SDL_CreateWindow` would split
  the two.
- **vcpkg doesn't install `vulkan-1.pdb`**: the `vulkan-loader` port never calls `vcpkg_copy_pdbs()` (still the case
  upstream, no issue filed), and the PDB left in the buildtree doesn't match a DLL restored from the binary cache. To
  profile the loader, rebuild it from the vcpkg clone (at the manifest's baseline) and use the buildtree's DLL and PDB
  together: `vcpkg install vulkan-loader:x64-windows --overlay-triplets=../brutus-ng/triplets
  --x-install-root=../brutus-ng/scratch/vulkan-loader --no-binarycaching`.
- **Bistro's end time depends on when the big meshes start**: anything that moves when `main` joins `wait_ready()`
  shifts TBB's splitting of `prepare_meshes` and can move Bistro's startup by ~0.3 s either way (debug utils above).
- **`decompress_tex` is misleading**: it also covers DDS reads and unused images. Return early for unused images before
  the scope and use separate scopes (e.g. `read_dds`, `decode_image`) so a fallback to PNG shows up.
- **`decompress_textures` time isn't read time**: it shares TBB workers with `prepare_meshes`, and varied 0.7-2.4 s
  over 3 cached runs. Per-file reads: median 34 ms on a run that was likely cold, 10-13 ms cached, against ~3 ms for a
  plain single-threaded read.
- **`load()` zeroes every file buffer**: `std::make_unique<std::byte[]>` value-initializes, 2 GB for Bistro;
  `make_unique_for_overwrite` skips it. Not measured.
- **Optick drops events** in these captures: 229-328 of Bistro's 338 DDS scopes were recorded depending on the run.
  Treat scope counts as approximate.
- **Peak memory varies by build, not just by run**: the same code built twice peaked at 1,653 and 1,718 MB of private
  bytes on the crowd, consistently over 5 runs each, even under swapped exe names. Crowd memory deltas below ~65 MB
  between two builds aren't attributable to the change. Bistro's peak working set varies 1,425-2,041 MB between runs
  of one exe. Measure with `PeakPagedMemorySize64` (private bytes) as well as the working set.
- **Run exes under neutral names** when comparing: the first run of a new exe path is slow (a cold driver cache, ~2.1
  s for the crowd instead of ~1.2 s), so copy both builds to new names and drop or discount the first round.
- **Uncaught exceptions print**: `main` installs a terminate handler that prints `what()` and aborts. Nothing unwinds
  (checked with MSVC), so breaking on throw and crash dumps still show the throw site. A crash that prints nothing is
  not an exception: 0xC0000409 is also what a /GS failure or any other fail-fast exits with.
- **`--no-dds` needs `--texture-compression none` on Bistro**: 53 of its PNGs exceed the compressor's 2048 limit, and
  one is 2560x2560. It now throws `std::invalid_argument` instead of losing the device (see
  [texture_compression.md](texture_compression.md)).
- Profiles: Optick saves `startup_debug.opt` / `startup_release.opt` at every startup (overwritten), read with
  `util/optick_report.py`. Compare startup timings with alternating runs of both builds and medians: single runs vary
  by ~0.1 s.
