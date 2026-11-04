# Draw pass performance

Notes for the draw pass (opaque, culling, LODs, draw commands) performance work: what's done, the current state, the
plan and the experiments. Shared rules and measurement setup: [working_process.md](working_process.md). Shadows have
their own notes: [shadows.md](shadows.md).

## Done so far

| Commit | Change |
| --- | --- |
| vk-renderer `a09ec3c`, `aacef3a` | `CommandBuffer::debug_label()` RAII scopes, labels around every pass (GPU Trace/RenderDoc read them) |
| `af705b6` | Draw statistics in device-local memory with per-workgroup aggregation (culling 0.45 -> 0.06 ms) |
| vk-renderer `9557792`, `cb274f8` | Meshlet occlusion culling in the task shader: two-phase, one visibility bit per main LOD meshlet per draw (`MeshDraw::meshlet_visibility_offset`, `mesh::meshlets_count`). `ResourceState::SHADER_STORAGE_READ_WRITE` for task shader writes. Also fixed transparents (drawn from late pass commands, used to miss draws visible the previous frame) |
| `dd26a19` | Shifts/masks for visibility bits (no measurable change) |
| `1ddcaf5` | One indirect task dispatch per bucket: culling writes one `MeshTaskCommand` per task workgroup (draw id, LOD, drawn_early, first meshlet) instead of one draw record per draw. `TASK_GROUP_SIZE` (32) and `TASK_COMMANDS` (0/1) shader defines, `mesh::max_task_groups` |
| vk-renderer `b0e4fcd`, `8a17099`; `60a8593` | `Device` tracks live allocations (`MemoryTracker`): by resource (texture, storage, index, other buffer) and by memory (VRAM, BAR, RAM, disjoint like `Device::Properties`). Shown in the UI |
| `dea4d05` | Index buffer capped at 32M indices (512 -> 128 MB, Bistro needs ~27M) |
| `e794033` | One draw commands buffer per pass type (main, shadows) instead of 4 each: union of the vertex/task layouts (`headers[4]` + `commands[]`), per-bucket ranges in `DrawCommandsSegment` tables updated incrementally by `add_mesh`; `material::type`; `draw_bucket()` |
| vk-renderer `ca9b0cc`; `b028c4e` | (user) raii `Buffer`/`Texture`/`TextureView` move assignment frees what the target held |
| `2602c6a` | Small triangle culling in every pass: bounding box scaled by `SceneConstants::sample_count` (standard sample locations for 1/2/4/8 samples project to (k + 0.5) / N per axis). Main pass -3.5% triangles, no measurable time change, kept for a single path |
| `0c601d0` | 8 LOD levels (was 6), error driven: no index target, relative error limits doubling from 0.0025 to 0.16. Shadow chain much coarser (frame 7.42 -> 6.79 ms), main chain barely changes. `upload_lods` throws when the index buffer is full |
| `e0e258f` | Scene constants split into `FrameConstants` (written once per frame) and `PassConstants` (per pass) |
| `b99cfec` | (user) GPU timings reworked and shown in the UI (`gpu_timing.h`) |
| `164bab8` | Draws sorted into one bin per bucket and LOD, LOD 0 first, a debug label per bin (early pass median 4.41 -> 3.82 ms) |
| `b11ad66` | Two-pass command generation: culling counts bins and records each draw's slot, `mesh_commands.comp` packs the bins in their bucket and writes the commands (back to 4 MiB per commands buffer). Vertex path sorted too |
| `2d9bbb8` | Naming: bucket (material type, pipeline state) > bin (draw order by a sort key, today the LOD) > command |
| `316845e` | 8-byte task commands (`draw_id`, `first_meshlet` with the drawn-early bit, LOD from the bin): -10 MiB VRAM |
| `1808416` | Task shader emits from every invocation in uniform control flow (lane 0 only was undefined behavior), real barriers |
| `64ca18b` | Task shader ranks its meshlets by view depth in the payload (front to back): LOD 0 bin 0.78 -> 0.70 ms, LOD 0 PS invocations -14%. `INFINITY` in `math.glsl` |
| `ac6310d` | `--msaa 1/2/4/8` (default 4); one sample renders straight into the resolved textures |
| `ba2ce05` | Mesh shader outputs packed into 4 locations (normal + U, tangent + V, world position, flat material with the bitangent sign): LOD 4-6 bins 1.06 -> 0.96 ms, frame mean 5.18 -> ~5.03 ms |

## Current state

- **Culling**: per draw in `mesh_cull.comp` (frustum, LOD, mesh-level HZB in the late pass), then per meshlet in
  `mesh.task` (frustum, cone, two-phase HZB with visibility bits). Both passes must select the same LOD. Bits aren't
  reset on scene changes (the late pass corrects them); the 100x100 crowd uses ~12M bits (1.5 MB, capacity 32M).
- **Commands** (since `b11ad66`): buckets (material type: opaque/alpha x single/double sided) are ranges of one
  `commands` array sized for the scene's worst case (`DrawCommandsSegment`, CPU known). Within a bucket, culling sorts
  draws into `DRAW_BINS_PER_BUCKET` bins by a sort key (the LOD; keep it a power of two). `mesh_cull.comp` counts each
  bin (`atomicAdd` of the draw's task workgroups) and records each draw's bin and offset in it (`DrawSlot`, by draw
  id); `mesh_commands.comp` prefix sums the bin counts from the bucket's start and writes the commands (a counting
  sort). Layouts in `shaders/draw_commands.glsl`. Commands buffers: 2 MiB of task commands (256k x 8 B) + 512 KiB of
  slots each, 5 buffers (main + one per cascade).
- **Dispatch**: mesh path, one `vkCmdDrawMeshTasksIndirectCountEXT` per bin, LOD 0 first (the task shader reads its
  bin's first command from the bin header, `draw_bin` push constant); vertex path, one multi draw per bucket with its
  bins in order. No Y split: this GPU allows 4M groups in X, the spec only guarantees 65,535.
- **Task shader**: every invocation reaches `EmitMeshTasksEXT` (uniform control flow, required by the spec); visible
  meshlets are ranked by view depth into the payload, so mesh workgroups draw front to back within a task workgroup.
- **MSAA**: `--msaa` (default 4). With one sample the passes render into the resolved textures, no resolves.
- **Timings** (Release, 2 x 10 traced frames, `64ca18b`): frame ~4.7-5.4 ms, early pass median ~3.4-3.7 ms, shadow
  map ~0.64 ms, late pass ~0.1 ms. Early pass per bin: LOD 0 0.70 (0.67-0.84), LOD 1 0.33, LOD 2 0.56, LOD 3 0.67,
  LOD 4 0.68, LOD 5 0.25, LOD 6 0.08, LOD 7 0.07 ms (0.4-0.8 ms in about a third of frames: end of pass work, DRAM
  76%). Before the bins (`0c601d0`): frame ~6.8 ms, early pass ~4.5 ms.
- **Limiters** (Nsight unit metrics, per range): LOD 0-2 bins shading bound (SM 56-74%, pixel warps 53-78%, ~1.9 ms);
  LOD 3-6 bins primitive engine bound (PES 51-64%, SM 12-26%, ~1.7 ms); shadow cascades ZROP/raster bound (ZROP 49-60%,
  SM 4-11%, ~0.56 ms); grid and tonemap bandwidth bound.
- **Counters** (RenderDoc, `0c601d0`): early pass 8.52M rasterized triangles (12.68M without meshlet culling), 10.76M
  fragment invocations for ~1.4M covered pixels (~7x), 3.49M mesh invocations, 328k task invocations. Task lane use is
  54% in the main pass (17 of 32 meshlets per group) and 23% in cascade 3 (7 of 32), but task invocations are a small
  share of the work: lane packing across draws is deprioritized.
- **LODs** (`0c601d0`): 8 levels, each the coarsest mesh within a relative error of 0.0025 x 2^lod (normals weighted 0.5
  for the main chain, positions only for shadows). Selection: coarsest LOD whose projected error is under
  `lod_pixel_tolerance` (1 px main, 2 shadow texels). The main chain's error is dominated by normal error (2-4x the
  position-only error at the same triangle count), which is why it stays fine. Tolerance experiments (old 6-level
  chain): 2 px = -1.1 ms frame, mid-distance faces slightly softer; 4 px = -1.7 ms, near faces' mouths distort.
  Distant faces look smeared at 1 px already (collapses slide UVs along the surface without geometric error).
- **Memory** (UI, benchmark, `316845e`): VRAM ~660 MB, storage buffers 31 MB, index buffer 128 MB (32M indices),
  textures 510 MB (414 bindless + MSAA color/depth). Bistro (`--scene ../niagara_bistro/bistro.gltf`): VRAM ~3.2 GB, textures 2.9 GB, storage
  181 MB, **30.7M of 32M indices** (overflow throws "Index buffer is full"), ~4.9k visible draws. Bistro still needs
  camera defaults for the benchmark (user to provide a "Copy camera" snippet).
- **Startup** (2026-10-28, 3 runs each): Release ~1.76 s; Debug ~11.1 s, of which ~6.5 s waiting for pipelines:
  shader compilation is ~60x slower in Debug (likely the Debug shaderc build) and paces startup.

## Next steps (updated 2026-10-29)

The early opaque pass is the focus (shadows are fast and acceptable for now).

1. **Far LODs (LOD 3-6 bins, primitive engine bound, ~1.7 ms)**: fewer triangles through LOD selection with a smaller
   normal weight in the selection error (0.1-0.25, computed at the same index count; position-only was too loose, see
   Experiments). Smaller mesh shader outputs: done (4 locations instead of 5, see Experiments); no 16-bit
   interpolants on this GPU. Check current mesh/task/fragment register counts alongside.
2. **Near LODs (LOD 0-2 bins, shading bound, ~1.9 ms)**: order task workgroups within a draw. Needs spatially coherent
   groups of up to 32 meshlets per LOD (`meshopt_partitionClusters` if our meshoptimizer has it, or a spatial sort) with
   a bounding sphere each, then `mesh_commands.comp` writes a draw's task commands sorted by group depth (~10-20 per
   draw). Measure with PS invocations and the bin timings, not RenderDoc's overdraw overlay. Ceiling: the LOD 0 prepass
   brought the bin to 0.37 ms.
3. **Visibility buffer**: the direction for this kind of scene (dense geometry, shading getting heavier). Roughly break
   even today (an ID pass costs ~2.5 ms like the full prepass), pays off with heavier shading. Open points: MSAA edges
   (per distinct triangle shading) vs temporal AA, analytic derivatives once mips are sampled, material binning. Worth
   measuring quad overshading (helper lanes) first.
4. **Startup**: SPIR-V caching or a Release shader compiler in Debug (user has ideas).

Other candidates:

- **LOD selection alternatives**: raise the default tolerance to 2 px (-1.1 ms, slight softening at mid distance);
  protect faces by locking face vertices (meshopt vertex lock) or a much larger UV weight (50-100; weights 1-4 changed
  nothing because a face covers a small UV region). Rejected: UV weights 1-4 (no effect), raising the deep LODs' error
  caps (coarser LOD 5 but selected farther away, net +2% triangles), normal weight 0.1 for generation (breaks faces).
- **Finer bins**: 64-256 depth bins per bucket instead of the LOD (one counting sort pass, needs a real workgroup scan
  in `mesh_commands.comp`), if the LOD order turns out too coarse.
- **Vertex path with meshlet culling**: compute meshlet culling writing compacted indices for plain indexed draws (the
  vertex path rasterizes faster but loses meshlet and small triangle culling, see Leads from the Unity comparison).
- **Index buffer**: Bistro uses 30.7M of 32M indices; raise to 48M (+64 MB) or size it from the loaded assets.
- **Scaling**: split `MAX_DRAWS` (draws upload, commands, visibility) into separate limits; device-local draws buffer
  filled from staging or updated incrementally (the per-frame host-visible copy only fits BAR up to ~100 MB, past that
  GPU reads go over PCIe: 10M draws cost +2 ms); Y split of task dispatches for GPUs limited to 65,535 groups.
- Deprioritized: task lane packing across draws (task work is small next to mesh/pixel work), MSAA-aware culling
  refinements (done in `2602c6a`, little to gain at 4x), a full GPU sort of draws (coarse bins get most of the gain
  between draws).

## Experiments

Temporary experiments and measured changes, oldest first.

- **Position-only LOD selection error** (2026-10-27, temporary): 9.12M -> 4.55M triangles, early draw 4.03 -> 2.60 ms
  (in-app Release timings, approximate), but the nearest face's mouth distorts and hair highlights facet. Combined error
  is a median 2.2x the position-only one at LOD 1, 3.8x at LOD 7.
- **Full depth prepass** (2026-10-27, temporary, 10 traced frames each): early pass 3.9-5.5 -> 5.2-5.9 ms, a net loss of
  ~1 ms. The depth-only pass alone costs 2.5-2.7 ms (ZROP 62%, raster 36%, SM 9%: fixed function bound, so a
  depth-only mesh shader wouldn't help much), the shaded pass after it 2.6-3.2 ms (PES 50%, PS warps 44% -> 20%), about
  what albedo-only shading cost: hidden overdraw shading is gone but geometry is paid twice. Geometry is the floor:
  ~2.5 ms of the early pass with no fragment work. A visibility buffer would pay it once plus visible-only shading.
- **Per-LOD draw lists** (`164bab8`): pixel identical on both paths, early pass median 4.41 -> 3.82 ms (10 traced
  frames each, separate sessions). Sized per list for the worst case it cost +30 MiB VRAM (`MAX_TASK_COMMANDS` 256k ->
  640k, the benchmark needs ~615k against ~215k), which the two-pass generation removed. The last bin's range spikes
  to ~0.58 ms in some frames (DRAM 76%, CROP 28%), presumably end of pass color/resolve work landing in the last label,
  which also explains the early pass's frame to frame spread.
- **Overdraw heatmap** (`docs/overdraw_early_1x.png`, 2026-10-27, 1 sample): overlay values highest by far in the
  distant crowd band (64+), then near characters' heads, hair and skirts (24-63); limbs and large surfaces 4-11. Same
  frame: 5.89M PS invocations for 1.50M covered pixels (3.9 per pixel at 1 sample). 1 sample vs 4x MSAA: early draw
  2.91 vs 4.03 ms (in-app Release timings, approximate).
- **Two-pass command generation** (`b11ad66`): back to 4 MiB per commands buffer plus 512 KiB of slots (storage 69 ->
  41 MB, +2.5 MiB against before the lists). Culling +0.019 ms for the shadows, ~+0.006 early, +0.004 late; draws
  within noise. The task commands (256k x 16 B) sized the union, not the vertex commands (64k x 28 B = 1.75 MiB).
- **8-byte task commands** (`316845e`): task commands 4 -> 2 MiB per buffer, still above the vertex commands: -10 MiB
  VRAM over the 5 buffers (storage 41 -> 31 MB). Pixel identical, timings unchanged within noise.
- **Overdraw heatmap with LOD 0 first** (`docs/overdraw_early_1x_2d9bbb8.png`): mean 36.6 -> 18.4, median 23 -> 13,
  the far band mostly gone. Up to the end of the LOD 0 bin alone, the near characters already show the same hot spots
  (mean 14.8 on 47% of the screen): what remains is overdraw within the nearest draws (hair, skirt and overlapping
  characters), which draw order between bins can't fix.
- **LOD 0 only prepass** (2026-10-28, temporary, 2 x 10 traced frames each way, alternating): the LOD 0 bin's shaded
  draws drop from 0.74-0.94 to 0.36-0.38 ms, but the depth-only prepass of that bin costs 0.52-0.61 ms: early pass
  median 3.79-3.80 ms without, 3.81-3.84 ms with. Break-even: the near characters' geometry (~1M triangles at 4x MSAA)
  is as expensive to rasterize again as the overdraw it removes.
- **Front to back meshlets within a task workgroup** (`64ca18b`): LOD 0 PS invocations 3.72M -> 3.20M (RenderDoc).
  Early bins (2 x 10 traced frames each): LOD 0 median 0.78 -> 0.70 ms, LOD 1 0.36-0.38 -> 0.32-0.34, LOD 2 0.60-0.64
  -> 0.55-0.59, LOD 3-6 and the cascades unchanged: ~-0.17 ms, a third of the LOD 0 prepass ceiling. Pixel identical.
- **RenderDoc's overlay and the sort** (2026-10-28): after the sort the overlay statistics match the `2d9bbb8` heatmap
  exactly (mean 18.37, LOD 0 bin 14.77) while PS invocations dropped 14%; with a temporary full depth prepass the
  overlay drops to a mean of 1.72 (max 5). RenderDoc's source explains it, see the tools section of
  [working_process.md](working_process.md).
- **Mesh shader outputs** (2026-11-04, Release, 10 traced frames per run, `scratch/packing/`): 13 interpolated scalars
  in 5 locations (UV, world position, normal, tangent, flat material). Cost of 8 more scalars in 2 more locations
  (temporary): early pass +0.24 ms on its fastest frame, LOD 2-6 bins +0.02-0.04 ms each. Two packings, early pass
  mean / frame mean (base 3.84 / 5.18 ms):
  - A, 3 locations (normal + U, tangent + V, flat material carrying the bitangent sign; world position rebuilt from
    `gl_FragCoord` with an inverse view-projection): LOD 4-6 bins 1.06 -> 0.92 ms but LOD 0 0.77 -> 0.85 ms (16 FMAs
    and a divide per fragment in the shading bound bin), 3.66-3.73 / 5.15 ms. Rejected.
  - B, 4 locations (A with the world position interpolated): LOD 0 back to 0.78-0.80, LOD 2-3 slightly faster, LOD
    4-6 0.96 ms, 3.63-3.67 / 5.01-5.04 ms (fastest frame 4.89 -> 4.68-4.72 ms). Kept: 84 pixels differ by at most
    2/255 on both paths (the bitangent sign is now the provoking vertex's on triangles mixing signs).
  The device has no 16-bit interpolants (`storageInputOutput16` false), so half precision UVs/normals aren't an option.
- **Mesh shader triangle culling off** (2026-11-04, Release, alternating 10 frame traces, `scratch/tri_cull/`): the
  back face, zero area and small triangle tests culled 3.4% of the early pass triangles (0.3% in LOD 0, 3-9% in LOD
  3-7) and 11.4% of the shadows' (cascade 3: 16%), RenderDoc rasterized primitives with and without. Without them
  (`TRIANGLE_CULLING` 0: no shared positions, barrier or per triangle tests) the early pass drops 3.39-3.57 -> 3.02-3.16
  ms at 4x MSAA (LOD 3-5 bins 1.54-1.59 -> 1.27-1.28 ms), 2.68 -> 2.32 ms at 1x, frame 4.73-4.91 -> 4.36-4.48 ms;
  shadows unchanged; pixel identical at 1x and 4x. Only disabling the flag (keeping the shared writes and the barrier)
  recovered 0.2 ms, so the workgroup sync costs as much as the tests. Compaction (only emitting survivors) would add a
  prefix sum on top: dropped. Niagara also keeps its mesh shader culling behind a define (`MESH_CULL`).

## Leads from the Unity comparison

Measured against the Unity 6.3 URP reproduction (`../Renderer Compare`, results in its `docs/benchmark_results.md`) at
`47c6203`, Nsight at locked clocks, RenderDoc counters, driver register counts from `KHR_pipeline_executable_properties`
(RenderDoc disassembly target). Scene setup in [benchmark.md](benchmark.md).

1. **Mesh shader register footprint**: task 40, mesh 86, fragment 64 registers. Fragment alone would fit 32 warps per
   SM, but 34% of PS warp launches in the early pass stall on register allocation (Unity: 11%, with a 38-register
   vertex shader and a 72-register pixel shader): mesh warps (~2.8k registers each) crowd the register file. The vertex
   path (43-register vertex shader) brings it to 27%. Trimming the mesh shader's live state is worth a look.
2. **Vertex path with meshlet culling**: the vertex path rasterizes 2.47M triangles/ms in the early pass against
   1.77M/ms for mesh shaders (11.49M triangles in 4.65 ms against 7.68M in 4.33 ms, same pixel work), but loses
   meshlet and small triangle culling (+50% triangles, PES 46%, SMs 13% idle). Compute meshlet culling writing
   compacted indices for plain indexed draws could combine both.
3. **Pixel shading is ~38% of the early pass**: 4.33 ms with full shading, 2.70 ms with `mesh.frag` reduced to the
   albedo fetch (temporary experiment). The rest is geometry, raster and depth/color writes. Not memory bound (DRAM
   reads 13%, L2 30%, texture reads hit L2 80% of the time).

## Backlog

- In this paused benchmark frame the late pass draws almost nothing: presumably meshlets at the HZB threshold, frame
  unchanged, not confirmed.
- Small triangle culling (`TRIANGLE_CULLING` in `mesh.mesh`, off) assumes the standard sample locations; vk-renderer
  doesn't expose `VkPhysicalDeviceLimits::standardSampleLocations` to check it.
- To compare traces with and without triangle culling again, pass `TRIANGLE_CULLING` as a pipeline define (shaders
  recompile on the fly) instead of editing `mesh.mesh`.
- Environment: `hemisphere_environment()` becomes an irradiance (diffuse, sampled with N) and prefiltered (specular,
  sampled with R) cubemap lookup once there is an environment map.
- The early culling variant could declare `VisibilityBuffer` readonly if `LATE` became a 0/1 define (minor).
- Non-zero meshlet `cone_weight` once assets are single sided (cone culling barely ever triggers now).
- Render graph / resource state tracker for barriers (user draft in `vk-renderer/src/renderer/render_graph.*`,
  untracked).
- Transparents aren't sorted back to front.

## Key findings

- **Per-draw indirect records were front-end bound**: ~7k records per pass, each launching 1-3 task workgroups, left
  SMs ~60% idle in the late pass (0.57 ms for little work). One dispatch per bucket fixed it.
- Mesh-level HZB alone barely helps a crowd (nearly every mesh is partly visible); meshlet culling cuts 33% of main pass
  triangles but only 11% of fragment invocations: the remaining overdraw is inside visible meshlets.
- Draw order matters: LOD 0 first halved the overdraw overlay, front to back meshlets cut LOD 0 PS invocations 14%.
  Depth prepasses don't pay here: geometry is the expensive part.
- `meshopt_simplifyWithAttributes` result error includes attribute (normal) error; position-only simplification wrecks
  faces in the main pass but is fine for shadows. Simplify on the original indices then remap with
  `meshopt_generateShadowIndexBuffer`.
- LOD error is the distance off the original surface: it doesn't see vertices sliding along the surface, which moves
  painted features (faces). meshopt's combined error (position + weighted attributes) is what selection compares to
  pixels, so attribute weights also change which LOD is picked, not only how it's simplified.
- `gpu_mesh` is 368 bytes with 8 LODs; bindless buffers are capped at 16 MiB (`MAX_MESHES` 4096).
- Most test assets are double sided; back face culling in the mesh shader was validated against hardware culling.
