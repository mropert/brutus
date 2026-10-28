# Shadows and draw pass: progress and plan

Handoff notes for continuing the shadow quality and draw pass performance work. Measurement scripts are documented in
[util/README.md](../util/README.md); this file covers the state of the work, what's left, and the working process.

All numbers are from the `benchmark` scene (10k characters, ~40k draws, fixed camera and time) on an RTX 2080 SUPER
(8 GB, 214 MB BAR, no resizable BAR), Debug build, GPU Trace at locked base clocks, mesh shader path unless stated.

## Done so far

### Shadows

| Commit | Change |
| --- | --- |
| vk-renderer `a09ec3c`, `aacef3a` | `CommandBuffer::debug_label()` RAII scopes, labels around every pass (GPU Trace/RenderDoc read them) |
| `af705b6` | Draw statistics in device-local memory with per-workgroup aggregation (culling 0.45 -> 0.06 ms) |
| `e124a43` | Orthographic cone culling uses the view direction (`projection_matrix[3][3]` convention) |
| `4ace47c` | Vertex shader path for the shadow pass |
| `f2f1cd8` | LOD errors stored in absolute units (`meshopt_simplifyScale`) |
| `b2c476f` | Separate position-only shadow LOD chain (`gpu_mesh::shadow_lods`, `SceneConstants::shadow_pass`), shadow tris 9.9M -> 6.3M |
| `5f4fcc8` | Mesh shader triangle culling: back faces (single sided), zero area, small triangles (shadow pass only) |
| `41ef760` | clang-format applied to shaders (byte-identical SPIR-V) |
| `019ad2f` | `shadow.glsl` (`compute_shadow`), ground (grid) receives shadows, fade over last 10% of shadow distance |
| vk-renderer `6ad6899`, `581e0be`, `5bd7c89` | Bindless cap 16 MiB; texture arrays + per-layer views, D16, dynamic depth bias, depth clamp; `Pipeline::DynamicState` + `set_dynamic_state` |
| `f43164f` | Cascade plumbing (4-layer array, per-cascade fit/cull/draw loop, array sampling) |
| `be08a86` | `util/` scripts |
| `faad762` | D16 shadow maps (32 MB for 4 cascades) |
| `1278968` | 4 cascades, UI: cascade count 0-4 (0 = off), cascade debug tint, shadow distance min 5 m; depth clamp on shadow pipelines |
| `3278d40` | Camera position/pitch/yaw in the UI with a "Copy camera" button |
| `9812590` | Normal offset (1.5 texels, slope scaled) + rasterizer slope bias replace the constant NDC bias |
| `a29f96d` | Cascade blending over the end of each cascade's depth slice (`sample_shadow_cascade`) |
| `5bc7ab1` | 5x5 tent PCF (9 bilinear compare taps, weights from the tent area per texel), blend band widened to 25% |
| `cbcc6d8` | Bump terminator: direct light multiplied by Chiang, Li, Burley "Taming the Shadow Terminator" (SIGGRAPH 2019), 0 where the geometric normal faces away from the light. Fixes light leaking on normal mapped surfaces in shadow (and without shadows). Both shading paths share `hemisphere_environment()` (sky/ground ambient), the basic path no longer has a constant ambient |
| `a1cfe26` | `SceneRenderer::FOV` in radians (0.884962, 50.70°, the angle 70 "radians" gave), image unchanged (0.03% of pixels within 2/255) |
| `a1875b4` | Shadow caster culling against each cascade's receivers (`shadow_caster_visible` in `constants.glsl`), per draw and per meshlet: shadow map 1.71 -> 0.67 ms, pixel identical. See Current state: shadows |

### Draw pass and memory

| Commit | Change |
| --- | --- |
| vk-renderer `9557792`, `cb274f8` | Meshlet occlusion culling in the task shader: two-phase, one visibility bit per main LOD meshlet per draw (`MeshDraw::meshlet_visibility_offset`, `mesh::meshlets_count`). `ResourceState::SHADER_STORAGE_READ_WRITE` for task shader writes. Also fixed transparents (drawn from late pass commands, used to miss draws visible the previous frame) |
| `dd26a19` | Shifts/masks for visibility bits (no measurable change) |
| `1ddcaf5` | One indirect task dispatch per bucket: culling writes one `MeshTaskCommand` per task workgroup (draw id, LOD, drawn_early, first meshlet) instead of one draw record per draw. `TASK_GROUP_SIZE` (32) and `TASK_COMMANDS` (0/1) shader defines, `mesh::max_task_groups` |
| vk-renderer `b0e4fcd`, `8a17099`; `60a8593` | `Device` tracks live allocations (`MemoryTracker`): by resource (texture, storage, index, other buffer) and by memory (VRAM, BAR, RAM, disjoint like `Device::Properties`). Shown in the UI |
| `dea4d05` | Index buffer capped at 32M indices (512 -> 128 MB, Bistro needs ~27M) |
| `e794033` | One draw commands buffer per pass type (main, shadows) instead of 4 each: union of the vertex/task layouts (`headers[4]` + `commands[]`), per-bucket ranges in `DrawCommandsSegment` tables updated incrementally by `add_mesh`; `material::type`; `draw_bucket()` |
| vk-renderer `ca9b0cc`; `b028c4e` | (user) raii `Buffer`/`Texture`/`TextureView` move assignment frees what the target held |
| `2602c6a` | Small triangle culling in every pass: bounding box scaled by `SceneConstants::sample_count` (standard sample locations for 1/2/4/8 samples project to (k + 0.5) / N per axis). Main pass -3.5% triangles, no measurable time change, kept for a single path |
| `0c601d0` | 8 LOD levels (was 6), error driven: no index target, relative error limits doubling from 0.0025 to 0.16. Shadow chain much coarser (frame 7.42 -> 6.79 ms), main chain barely changes. `upload_lods` throws when the index buffer is full |

## Current state: shadows

- **Cascades**: 4 x 2048² D16 array (`SHADOW_CASCADE_MAX`), practical split scheme (lambda 0.75) from 0.1 m to the
  shadow distance: about 1.9 / 4.3 / 9.4 / 25 m at the default 25 m. Each cascade is a bounding sphere fit of its view
  depth slice with texel snapping (`build_scene_shadow_constants`). Splits depend on the shadow distance, so a larger
  distance also enlarges the near cascades.
- **Selection** in `shadow.glsl`: by view depth against `shadow_cascade_splits`. Each cascade lerps into the next over
  the last `SHADOW_CASCADE_BLEND_RANGE` (25%) of its slice, keeping the current one where the next doesn't cover the
  point; the last cascade uses the distance fade.
- **Constants**: `FrameConstants` (main camera, lighting, shadow sampling, debug; written once at the start of
  `render`, after `setup_shadow_cascades`) and `PassConstants` (view, render target, shadow pass only fields; one per
  pass: main view and each cascade), separate push constant pointers. Fragment shaders only read the frame constants,
  culling/task/mesh/vertex shaders only the pass constants. C++ mirrors use explicit padding members for std430.
- **Bias**: no receiver depth bias. Normal offset of `SHADOW_NORMAL_OFFSET_TEXELS` (1.5) texels of the selected cascade,
  scaled by `sin(angle to light)`, texel sizes in `SceneConstants::shadow_cascade_texel_sizes`. Caster side: rasterizer
  bias `SHADOW_DEPTH_BIAS_CONSTANT` 1 / `SHADOW_DEPTH_BIAS_SLOPE` 2 (D16 units). No acne on the back view
  (`{ -4.7130, 1.6786, 3.6568 }`, pitch -0.0140, yaw 1.8799), no peter-panning at ground level (user checked).
  `compute_shadow` gets the interpolated vertex normal, not the true geometric one.
- **Filtering**: 5x5 texel tent (`SHADOW_PCF_TENT_RADIUS` 2.5) from 3x3 hardware bilinear compare taps. Soft edges, but
  thin contact shadows (collar, bangs) wash out. Rejected: a world-sized 16-tap grid (1.2 cm radius) only hides cascade
  switches by blurring the near cascade to the far one's resolution, +0.7 ms.
- **Caster culling**: each cascade's receivers are the visible points that sample it: camera frustum sides, view
  depth from the start of the previous cascade's blend band (0 for cascade 0) to the cascade's split. A caster's
  bounding sphere (per draw in `mesh_cull.comp`, per meshlet in `mesh.task`) is swept along the light up to the
  farthest receiver along it and the capsule is culled when it's outside a receiver plane. Planes are pushed out by
  `SHADOW_CASTER_MARGIN_TEXELS` (6: 1.5 normal offset + 3.5 tent/bilinear reach + 1 slack) so the result is exact:
  pixel identical on both paths at the benchmark view, the back view, a view facing the light (`{ 0, 6, -60 }`, pitch
  -0.30, yaw 3.1416) and a high view (`{ 0, 15, 10 }`, pitch -0.9, yaw 0). Changing the cascade selection, blend
  range, normal offset or filter in `shadow.glsl` must update `set_shadow_receivers` / the margin.
- **Cost** (caster culling, 5 traced frames, same session): shadow map 0.67 ms against 1.71 without it. Cascades
  0.09 / 0.12 / 0.18 / 0.28 ms (were 0.32 / 0.31 / 0.43 / 0.64), rasterized primitives 0.08 / 0.12 / 0.28 / 0.83M
  (1.31M, was 3.69M), depth samples 0.97 / 1.38 / 1.63 / 1.59M (5.57M, was 18.0M; Unity 12.4M). Draws still ZROP
  44-56%, SMs idle 8-60%: cascades 0-1 are now small enough that the fixed cost per cascade (culling dispatch
  ~0.027 ms, clears, barriers) is a third of their time. Frame ~5.2-6.0 ms (early pass noise dominates).
- **Batched culling** (7592996): one commands buffer per cascade (+12 MiB VRAM: 3 more 4 MiB buffers), the four culling dispatches
  run between one set of barriers, then the four draws. Shadow map 0.67 -> 0.61-0.62 ms (5 traced frames each, same
  session): culling 0.106 -> 0.055 ms (each dispatch is a single latency-bound wave, SM 3.5%, so they overlap), draws
  unchanged (0.56 ms, no measurable overlap between cascades). Next candidate: one dispatch culling each draw against
  all cascades (reads each draw once).
- **Cost before caster culling**: shadow pass ~1.8 ms since the error-driven LODs (`0c601d0`, was ~2.2-2.7). Before that: cascades 0.31 /
  0.33 / 0.48 / 1.1 ms, triangles 0.22M / 0.43M / 1.04M / 4.62M; cascades 0-2 depth fill bound (ZROP 56-61%), cascade 3
  bound by tiny triangles (raster 41%, ZROP 42%, primitive engine 31%). Per cascade numbers not re-measured since.
- **Terminator** (`cbcc6d8`): `compute_shadow` skips the lookup when the geometric normal faces away from the light;
  callers must give those surfaces no direct light (`bump_terminator_shadowing` in `mesh.frag`). Repro of the old leak:
  `{ -7.2700, 2.0876, 3.5322 }`, pitch -0.2620, yaw 2.5379.
- **Visible issue: popping at cascade switches**, mostly 0 -> 1 at ~1.9 m: casters thinner than a texel (bangs strands
  seen from the light) are rasterized in cascade 0 (~2 mm texels) but drop out of cascade 1 (~4.3 mm), so the shadow
  shrinks. Confirmed with 4096² maps (shadow stays); normal offset, filter and caster LOD each ruled out. Repro:
  `{ -7.1963, 2.4145, 5.0727 }` vs `{ -7.4208, 2.4759, 5.2944 }`, pitch -0.1920, yaw 0.7919. The benchmark is a
  dialogue-distance shot (main character at 1.5-2.5 m, ~51° vertical FOV, see FOV below) and cascade 0 covers
  0.1-1.9 m, often nothing visible. Also terminator facets.
- **Not a shadow issue**: the seam and light/dark split on Miku's upper arm at `{ -5.4757, 2.1348, 3.6858 }`, pitch
  -0.4440, yaw -1.8301 come from the mesh normals (visible without shadows; normal offset, filter and 4096² ruled out).
  Fix in the asset, or weld vertices and merge normals at load.

## Current state: draw pass

- **Culling**: per draw in `mesh_cull.comp` (frustum, LOD, mesh-level HZB in the late pass), then per meshlet in
  `mesh.task` (frustum, cone, two-phase HZB with visibility bits). Both passes must select the same LOD. Bits aren't
  reset on scene changes (the late pass corrects them); the 100x100 crowd uses ~12M bits (1.5 MB, capacity 32M).
- **Dispatch**: one `vkCmdDrawMeshTasksIndirectCountEXT` per bucket (draw count 1 stored in the header), task
  workgroups read `commands[ gl_WorkGroupID.x ]`. ~2 draw calls per pass instead of ~7k. No Y split: this GPU allows 4M
  groups in X, the spec only guarantees 65,535.
- **Command buffers**: `gpu_draw_commands` union, buckets are ranges of one `commands` array. `_draw_mesh_segments` /
  `_task_segments` hold each bucket's commands offset and worst case capacity, header offsets come from
  `calc_header_offset<Layout>( bucket )`. Task path clears only the headers; vertex path clears only the counts when
  draw indirect count is supported.
- **Timings** (10 traced frames, `0c601d0`): frame ~6.8 ms (9.43 before task records, 7.42 before the 8 LODs). Shadow
  pass ~1.8, early pass ~4.5, late pass ~0.08 ms. Vertex path last measured at ~7.8 ms before the 8 LODs.
- **Counters** (RenderDoc, default view): early pass 8.52M rasterized triangles (12.68M without meshlet culling),
  10.76M fragment invocations for ~1.4M covered pixels (~7x), 3.49M mesh invocations, 328k task invocations. Task lane
  use is 54% in the main pass (17 of 32 meshlets per group) and 23% in cascade 3 (7 of 32), but task invocations are a
  small share of the work: lane packing across draws is deprioritized.
- **Bottleneck**: early pass draw (~65% of the frame): SM 45%, pixel warps 45%, primitive engine 33%, ZROP 35%, idle
  <2%: ~8.2M tiny triangles plus ~6.6x shading overdraw, each pixel paying 9 shadow taps. Small triangle culling runs in
  every pass but only removes 3.5% in the 4x MSAA main pass (18% in cascade 3); culled triangles are only flagged
  (`gl_CullPrimitiveEXT`), which recovers ~40% of their cost.
- **LODs** (`0c601d0`): 8 levels, each the coarsest mesh within a relative error of 0.0025 x 2^lod (normals weighted 0.5
  for the main chain, positions only for shadows). Selection: coarsest LOD whose projected error is under
  `lod_pixel_tolerance` (1 px main, 2 shadow texels). The main chain's error is dominated by normal error (2-3x the
  position-only error at the same triangle count), which is why it stays fine. Tolerance experiments (old 6-level
  chain): 2 px = -1.1 ms frame, mid-distance faces slightly softer; 4 px = -1.7 ms, near faces' mouths distort.
  Distant faces look smeared at 1 px already (collapses slide UVs along the surface without geometric error).
- **Memory** (UI, benchmark): VRAM 655 MB, BAR 9 MB, RAM 0. Textures 510 MB (414 bindless + MSAA color/depth), storage
  buffers 27 MB, index buffer 128 MB (32M indices). Bistro (`#define BISTRO`): VRAM ~3.2 GB, textures 2.9 GB, storage
  181 MB, **30.7M of 32M indices** (overflow now throws "Index buffer is full"), ~4.9k visible draws. Bistro still
  needs camera defaults for the benchmark (user to provide a "Copy camera" snippet).
- **FOV**: `SceneRenderer::FOV` is 0.884962 radians, 50.70° vertical (80.2° horizontal at 16:9): it used to be 70
  passed to `tan()` as radians, `a1cfe26` kept the angle.

## Remaining plan

### Shadow quality

1. **Resolution where characters are**: depth-bounded cascades (SDSM: fit splits to the visible depth range from the
   depth pyramid, one frame of readback latency or GPU-side cascade setup) or fixed small near cascades. 4096² works
   but is 128 MB and only moves the problem. Skip cascades with nothing visible.
2. **Screen-space contact shadows** for strand-scale features (bangs, collar) that no reasonable shadow map catches.
3. Optional: per-character shadow maps for close-ups, 4096² as an "ultra" setting.

### Next steps (plan agreed 2026-10-27)

1. ~~FOV fix~~, ~~caster culling against cascade receivers~~ (done).
2. **SDSM**: first `--shadow-near` / `--shadow-split-lambda` options to measure quality and cost, then a min/max
   reduction of the previous frame's depth (the depth pyramid holds only early pass characters, and a single
   reduction) read back with 1-2 frames of latency. Splits from the measured range, but cascade 0 keeps starting at
   the camera so nothing new is unshadowed. In this scene only the near bound helps (the ground fills the view up to
   the shadow distance). Tighter (non sphere) fits are the bigger resolution lever, at the cost of rotation stability.
   Receiver planes could also use the measured range (fewer casters).
3. **Shader occupancy**: depth-only `mesh.mesh` variant for shadows (it fetches the full vertex and writes every
   attribute today); main mesh shader outputs (material per primitive, packed normal/tangent) and quantized vertices;
   ~~task shader barriers and early return~~ (fixed 2026-10-28: it also emitted from lane 0 only, undefined behavior
   per SPV_EXT_mesh_shader, every invocation must execute `EmitMeshTasksEXT` once in uniform control flow); next, sort
   the payload's meshlets front to back; fragment shader after Nsight stall reasons.
4. Later: depth prepass experiment (overdraw, and it would give SDSM this frame's depth), contact shadows.

### Draw pass and shadow cost

1. **Main pass LODs** (biggest measured lever, the early pass is triangle bound): the main chain stays fine because
   normal error dominates the recorded error. Candidates:
   - Select with position-only error (record it per LOD, e.g. from a position-only simplify at the same index count)
     while keeping normals to guide collapses. Tried 2026-10-27 (temporary): 9.12M -> 4.55M triangles, early draw
     4.03 -> 2.60 ms (UI), but the nearest face's mouth distorts and hair highlights facet. Combined error is a median
     2.2x the position-only one at LOD 1, 3.8x at LOD 7. Next idea: select with a smaller normal weight (0.1-0.25)
     computed at the same index count, or lock face vertices.
   - Raise the default tolerance to 2 px (-1.1 ms, slight softening at mid distance).
   - Protect faces: lock face vertices (meshopt vertex lock) or a much larger UV weight (50-100; weights 1-4 changed
     nothing because a face covers a small UV region).
   - Rejected: UV weights 1-4 (no effect), raising the deep LODs' error caps (coarser LOD 5 but selected farther away,
     net +2% triangles), normal weight 0.1 (breaks faces).
2. **Triangle compaction in the mesh shader**: emit only surviving triangles instead of flagging them.
3. **Exact sample test for tiny shadow triangles**: when the box spans exactly one texel center per axis, edge test it.
4. **Depth prepass**: tried 2026-10-27 (temporary, 10 traced frames each): early pass 3.9-5.5 -> 5.2-5.9 ms, a net
   loss of ~1 ms. The depth-only pass alone costs 2.5-2.7 ms (ZROP 62%, raster 36%, SM 9%: fixed function bound, so a
   depth-only mesh shader wouldn't help much), the shaded pass after it 2.6-3.2 ms (PES 50%, PS warps 44% -> 20%),
   about what albedo-only shading cost: hidden overdraw shading is gone but geometry is paid twice. Geometry is the
   floor: ~2.5 ms of the early pass with no fragment work. A visibility buffer would pay it once plus visible-only
   shading; fewer triangles helps every option.
   **Two-pass command generation** (uncommitted when written): culling only counts each list and records each draw's
   list and slot, `mesh_commands.comp` packs each bucket's LOD lists from the bucket's worst case start and writes the
   commands. Back to 4 MiB per commands buffer plus 512 KiB of slots (storage 69 -> 41 MB, +2.5 MiB against before the
   lists). The vertex path now sorts too (one multi draw per bucket in LOD order). Pixel identical on both paths.
   Culling +0.019 ms for the shadows, ~+0.006 early, +0.004 late (10 traced frames each, same session); draws within
   noise. The task commands (256k x 16 B) size the union, not the vertex commands (64k x 28 B = 1.75 MiB).
   **8-byte task commands** (uncommitted when written): `{ draw_id, first_meshlet | drawn early bit }`, the LOD comes
   from the bin. Task commands 4 -> 2 MiB per buffer, still above the vertex commands: -10 MiB VRAM over the 5 buffers
   (storage 41 -> 31 MB). Pixel identical, timings unchanged within noise.
   **Per-LOD draw lists** (`164bab8`, 2026-10-27): culling sorts task commands into one list per bucket
   and LOD, passes draw LOD 0 first. Pixel identical on both paths. Early pass median 4.41 -> 3.82 ms (10 traced frames
   each, separate sessions). +30 MiB VRAM: `MAX_TASK_COMMANDS` 256k -> 640k (the benchmark needs ~615k against ~215k
   before), 4 -> 10 MiB for each of the 5 commands buffers. Early pass per LOD (ms): LOD 0 0.74-0.94, 1 0.37, 2 0.61,
   3 0.68-0.78, 4 0.66-0.70, 5 0.25, 6 0.08, 7 0.07. LODs 0-2 are shading bound (PS warps 52-78%), LODs 3-6 geometry
   bound (PES 46-62%). The last range (LOD 7) spikes to ~0.58 ms in 4 of 10 frames (DRAM 76%, CROP 28%), presumably end
   of pass color/resolve work landing in the last label, which also explains the early pass's frame to frame spread.
   **Overdraw heatmap** (`docs/overdraw_early_1x.png`, 2026-10-27): RenderDoc 1.46's Quad Overdraw overlay is empty on
   4x MSAA targets (both render paths), so it was taken from a temporary 1-sample build (early draw 2.91 ms against
   4.03 at 4x, 7.77M triangles). Overlay values (unit unclear, not multiples of 4) are highest by far in the distant
   crowd band (64+), then near characters' heads, hair and skirts (24-63); limbs and large surfaces 4-11. Same frame:
   5.89M PS invocations for 1.50M covered pixels (3.9 per pixel at 1 sample).
   Again at `2d9bbb8` with LOD 0 first (`docs/overdraw_early_1x_2d9bbb8.png`): mean 36.6 -> 18.4, median 23 -> 13, the
   far band mostly gone. Up to the end of the LOD 0 bin alone, the near characters already show the same hot spots
   (mean 14.8 on 47% of the screen): what remains is overdraw within the nearest draws (hair, skirt and overlapping
   characters), which draw order between bins can't fix.
   **LOD 0 only prepass** (temporary, 2026-10-28, 2 x 10 traced frames each way, alternating): the LOD 0 bin's shaded
   draws drop from 0.74-0.94 to 0.36-0.38 ms, but the depth-only prepass of that bin costs 0.52-0.61 ms: early pass
   median 3.79-3.80 ms without, 3.81-3.84 ms with. Break-even. The near characters' geometry (~1M triangles at 4x MSAA)
   is as expensive to rasterize again as the overdraw it removes.
   **Front to back meshlets within a task workgroup** (uncommitted when written): the task shader ranks its drawn
   meshlets by view depth (count of nearer keys in shared memory) instead of appending them with an atomic. LOD 0 PS
   invocations 3.72M -> 3.20M (RenderDoc). Early bins (2 x 10 traced frames each): LOD 0 median 0.78 -> 0.70 ms,
   LOD 1 0.36-0.38 -> 0.32-0.34, LOD 2 0.60-0.64 -> 0.55-0.59, LOD 3-6 and the cascades unchanged: ~-0.17 ms, a
   third of the LOD 0 prepass ceiling. Pixel identical (RenderDoc resolved color). Next: order the task workgroups of
   a draw too (spatially coherent meshlet groups with bounds).
   RenderDoc's quad overdraw overlay doesn't see this: taken again after the sort (`--msaa 1`), its statistics match
   the `2d9bbb8` heatmap exactly (mean 18.37, LOD 0 bin 14.77), while PS invocations dropped 14%. With a temporary
   full depth prepass before the early pass, the same overlay drops to a mean of 1.72 (max 5): it does depth test,
   against the depth the pass starts with and what earlier draws wrote. RenderDoc's source (`vk_overlay.cpp`,
   `VulkanQuadOverdrawCallback`) confirms why: each draw of the pass is replayed first with the overdraw pipeline, depth
   test kept but depth writes off, then for real. So it sees order between draws (passes, LOD bins) but never within a
   draw. Use PS invocations or the bin timings for that. The overlay also only runs on single-sample targets (it's
   skipped unless the overlay's sample count is 1), hence `--msaa 1`.
5. **Far cascade**: coarser LOD multiplier (currently `lod_target = 2 * texel_world_size` everywhere), or update it
   every 2-4 frames.
6. **Index buffer**: Bistro uses 30.7M of 32M indices; raise to 48M (+64 MB) or size it from the loaded assets.
7. **Scaling**: split `MAX_DRAWS` (draws upload, commands, visibility) into separate limits; device-local draws buffer
   filled from staging or updated incrementally (the per-frame host-visible copy only fits BAR up to ~100 MB, past that
   GPU reads go over PCIe: 10M draws cost +2 ms); Y split of task dispatches for GPUs limited to 65,535 groups.
8. Deprioritized: task lane packing across draws (task work is small next to mesh/pixel work), MSAA-aware culling
   refinements (done in `2602c6a`, little to gain at 4x).

### Leads from the Unity comparison

Measured against the Unity 6.3 URP reproduction (`../Renderer Compare`, results in its `docs/benchmark_results.md`) at
`47c6203`, Nsight at locked clocks, RenderDoc counters, driver register counts from `KHR_pipeline_executable_properties`
(RenderDoc disassembly target):

1. **Done (caster culling, see Current state: shadows)**. **Cull shadow casters already covered by a closer cascade**: cascades 0-2 are depth fill bound (ZROP 58-60%, PES
   7-12%, SMs idle 24-38%). Unity writes 12.4M shadow depth samples against our 18.0M (cascade 3: 1.37M against 3.90M)
   because its caster culling (`ReceiverSphereCuller.ComputeSplitVisibilityMask` in URP's core package) treats each
   caster as a capsule along the light clipped to the camera frustum: it skips casters whose shadow can't reach a
   cascade's receiver sphere, and once a caster's whole shadow lies in a closer cascade's sphere it isn't drawn into the
   farther ones. We draw every caster inside each cascade's box, so the characters near the camera land in all four.
2. **Mesh shader register footprint**: task 40, mesh 86, fragment 64 registers. Fragment alone would fit 32 warps per
   SM, but 34% of PS warp launches in the early pass stall on register allocation (Unity: 11%, with a 38-register
   vertex shader and a 72-register pixel shader): mesh warps (~2.8k registers each) crowd the register file. The vertex
   path (43-register vertex shader) brings it to 27%. Trimming the mesh shader's live state is worth a look.
3. **Vertex path with meshlet culling**: the vertex path rasterizes 2.47M triangles/ms in the early pass against
   1.77M/ms for mesh shaders (11.49M triangles in 4.65 ms against 7.68M in 4.33 ms, same pixel work), but loses
   meshlet and small triangle culling (+50% triangles, PES 46%, SMs 13% idle). Compute meshlet culling writing
   compacted indices for plain indexed draws could combine both.
4. **Pixel shading is ~38% of the early pass**: 4.33 ms with full shading, 2.70 ms with `mesh.frag` reduced to the
   albedo fetch (temporary experiment). The rest is geometry, raster and depth/color writes. Not memory bound (DRAM
   reads 13%, L2 30%, texture reads hit L2 80% of the time).

### Backlog

- Alpha-tested materials don't cast shadows (needs a shadow pipeline with a discarding fragment shader, matters for
  Bistro). The second model's hair textures have 0 alpha (export bug): ~2000 invisible alpha draws are expected.
- In this paused benchmark frame the late pass draws 0 triangles (12k before task records): presumably meshlets at the
  HZB threshold, frame unchanged, not confirmed.
- Small triangle culling assumes the standard sample locations; vk-renderer doesn't expose
  `VkPhysicalDeviceLimits::standardSampleLocations` to check it.
- Environment: `hemisphere_environment()` becomes an irradiance (diffuse, sampled with N) and prefiltered (specular,
  sampled with R) cubemap lookup once there is an environment map.
- The early culling variant could declare `VisibilityBuffer` readonly if `LATE` became a 0/1 define (minor).
- Non-zero meshlet `cone_weight` once assets are single sided (cone culling barely ever triggers now).
- Fixed near cascade sizes independent of shadow distance (if wanted).
- Render graph / resource state tracker for barriers (user draft in `vk-renderer/src/renderer/render_graph.*`,
  untracked); SPIR-V caching if startup shader compilation becomes a problem.
- Possible ray traced shadows later as an optional path (no acceleration structure support in vk-renderer yet).

## Key findings to keep in mind

- The shadow pass is limited by **per-triangle cost** (primitive setup/raster of tiny triangles), not by vertex work or
  bandwidth: cutting triangles is the lever. Remapping indices or D16 didn't change time; position-only LODs and small
  triangle culling did. Since the error-driven LODs (`0c601d0`) cascades 0-2 are depth fill bound (ZROP ~60%, PES
  7-12%) and cascade 3 is mixed (ZROP 51%, raster 43%, PES 26%), see Leads from the Unity comparison.
- The geometry path doesn't matter for shadows: the vertex path takes 1.67 ms against 1.71 ms for mesh shaders, with
  more triangles (4.21M against 3.69M) and the same ZROP-bound profile.
- **Per-draw indirect records were front-end bound**: ~7k records per pass, each launching 1-3 task workgroups, left
  SMs ~60% idle in the late pass (0.57 ms for little work). One dispatch per bucket fixed it (late 0.09 ms, cascade 3
  1.73 -> 1.08 ms).
- Mesh-level HZB alone barely helps a crowd (nearly every mesh is partly visible); meshlet culling cuts 33% of main pass
  triangles but only 11% of fragment invocations: the remaining overdraw is inside visible meshlets.
- `#if` on a define set to `true`/`false` always evaluates to 0: preprocessor-tested defines must be 0/1
  (`TASK_COMMANDS`); `LATE` works because it's only used in code.
- Host-visible buffers land in BAR (214 MB here) only while they fit; VMA falls back to system memory silently.
- Most test assets are double sided (shadow pass culls nothing by facing); back face culling in the mesh shader was
  validated against hardware culling.
- `meshopt_simplifyWithAttributes` result error includes attribute (normal) error; position-only simplification wrecks
  faces in the main pass but is fine for shadows. Simplify on the original indices then remap with
  `meshopt_generateShadowIndexBuffer`.
- `gpu_mesh` is 368 bytes with 8 LODs; bindless buffers are capped at 16 MiB (`MAX_MESHES` 4096).
- LOD error is the distance off the original surface: it doesn't see vertices sliding along the surface, which moves
  painted features (faces). meshopt's combined error (position + weighted attributes) is what selection compares to
  pixels, so attribute weights also change which LOD is picked, not only how it's simplified.
- Normal mapped lighting needs a geometric term: a surface whose geometry faces away from the light is in its own
  shadow whatever the normal map says (the leak existed before shadow maps, they made it visible).

## Reproducing the benchmark in another engine

Goal: the same scene in Unity or Godot to compare frame time, triangle counts and visuals. Everything below is what
`main.cpp` / `scene_renderer.cpp` set up for `brutus-vulkan.exe --benchmark` (non-Bistro build), with what the Unity port
taught us.

A Unity 6.3 URP reproduction exists in `../Renderer Compare` (local git repository): editor scripts that import the
models and build the scene from the crowd dump, a `-benchmark` runner, and its findings in `docs/benchmark_results.md`
and `docs/visual_comparison.md`. Headline at `47c6203`: Unity takes 18.4 - 19.1 ms against our 6.4 - 7.1 ms at its
default Mesh LOD threshold, 9.6 - 10.8 ms at threshold 2 (near characters unchanged), mostly from triangle counts.

### Scene

- **Assets**: `assets/player.glb` (Miku) and `assets/sabrina.glb` (Sabrina). As downloaded, Sabrina's
  metallic-roughness textures hold roughness in blue (glTF metalness) with green constant at 1:
  `python util/fix_metallic_roughness.py assets/sabrina.glb` fixes the file (original kept as `sabrina.orig.glb`),
  and the other engine must use the fixed file. Sabrina's body, head and face meshes carry 7 UV sets that are all
  copies of `TEXCOORD_0` (the only one we read), Miku is skinned (bind pose, we ignore the skin): import Miku as static
  meshes, and strip the extra UV sets and bone weights if the engine keeps them, to compare the same vertex data.
- **Crowd**: 100 x 100 characters on a 1 m grid on the y = 0 plane: for `z` in 0..99 (outer loop), `x` in 0..99,
  position `( x - 50, 0, z - 50 )` in metres. Per character, in that order: asset index (uniform 0 or 1), yaw
  rotation, uniform scale in [0.5, 1.5].
- **Yaw quirk**: the yaw is a uniform integer in 0..360 passed to `glm::quat( glm::vec3( 0, yaw, 0 ) )`, which takes
  radians: it's effectively a random angle, not degrees.
- **Transforms are not portable as an algorithm**: `std::minstd_rand` (default seed) is fully specified, but
  `std::uniform_int_distribution` / `std::uniform_real_distribution` are implementation defined, so another engine
  can't regenerate the same crowd. Export it instead: `brutus-vulkan.exe --dump-scene <file>` (non-Bistro build, no
  device created) writes the benchmark crowd as CSV, one line per character: asset path, position, rotation quaternion
  (x, y, z, w), scale, in glTF conventions. It shares `generate_crowd()` with the renderer, so it stays in sync.
- **Meshes are placed directly with the character transform**: every mesh (glTF primitive) of the asset gets the
  character's translation/rotation/scale; glTF node transforms are ignored in this mode (only the Bistro path walks
  the node hierarchy). If an asset's nodes aren't identity, the other engine's import must flatten them the same way.
  For both assets the node transforms cancel out (Sketchfab's Z-up rotations): every mesh's world matrix is identity,
  so taking each mesh at identity matches.
- **Ground**: an infinite unlit grid on y = 0 (`grid.frag`) that receives shadows. A large plane is close enough; its
  cost is negligible, but it's needed to check that shadow bias doesn't detach shadows at the feet.
- **Time is paused**: no animation, the characters are static in their bind pose.

### Camera

- **Conventions**: right-handed, Y up, camera looking down -Z (glTF convention, same as Godot; Unity is left-handed
  and its glTF importers mirror an axis, so positions and rotations need the same conversion as the meshes). glTFast
  (Unity) negates X: positions `( -x, y, z )`, quaternions `( x, -y, -z, w )`, so the camera's forward becomes
  `( -0.6190, -0.2917, -0.7292 )`.
- **Pose**: position `( -7.5234, 2.5384, 4.8755 )`, pitch -0.2960, yaw 0.7039 (radians). Rotation is yaw about -Y
  then pitch about X, so forward = `( sin yaw cos pitch, sin pitch, -cos yaw cos pitch )` = `( 0.6190, -0.2917,
  -0.7292 )`: look at `( -6.9044, 2.2467, 4.1463 )`, up = +Y, no roll.
- **Projection**: vertical FOV **50.70°** (not 70, see FOV above), 1920 x 1080 (aspect 16:9, 80.2° horizontal), near
  0.01 m, infinite far (reverse Z). Other views from these notes use the same convention (UI "Copy camera"). Unity has
  no infinite far plane: 1000 m loses nothing in this view.

### Lighting and output

- **Sun**: one directional light travelling along `( 0.4045, -0.8660, -0.2939 )` (60° elevation, azimuth 144° from
  +X towards +Z). Intensity 4.5π in our units with exposure 0.25 and a Reinhard tonemap `x / (1 + x)`. The units do
  map: our direct diffuse is `albedo / π * intensity * NdotL`, URP's has no 1/π, so with the exposure folded in the
  Unity sun is 4.5π / π x 0.25 = 1.125 (no tonemapping, URP's default). Without Reinhard the other engine looks
  brighter and more saturated: that's most of the remaining visual difference.
- **Ambient**: sky/ground hemisphere, sky `( 0.5, 0.6, 0.7 )` and ground `( 0.2, 0.18, 0.15 )`, both scaled by the
  sun intensity x 0.06 (`sky_to_sun`), used as is for the diffuse ambient (sampled with N, no 1/π, no cosine
  convolution) and as the specular environment (sampled with R). It's linear in the direction's y, so an SH probe's L0
  and L1 (y) bands represent it exactly: Unity's gradient ("Trilight") ambient flattens the sky/ground slope to ~56%,
  the Unity port sets the probe from a script on load (Unity doesn't save a custom probe with the scene) and uses the
  same hemisphere as a small reflection cubemap. No AO (disable the engine's SSAO, URP's template enables it), no GI.
- **Background**: we clear to 0.1 before exposure and tonemap: the displayed value is 0.1 x 0.25 / (1 + 0.025) ≈
  0.024 (linear), which is what a non-tonemapped engine should clear to.
- **Shadows**: 4 cascades, 2048² each, 25 m shadow distance, splits ~1.9 / 4.3 / 9.4 / 25 m (practical scheme,
  lambda 0.75), 5x5 PCF, cascades blended over the last 25%. Use the engine's closest settings. URP: the shadow map
  resolution is the atlas (4096 for 4 x 2048²), splits as fractions of the distance (0.076 / 0.172 / 0.376), "Medium"
  soft shadows are the same 5x5 tent, no blending between cascades (only the last one fades out). The assets are
  double sided: URP's caster-side normal bias pushes back faces towards the light and causes striped acne, it needs
  normal bias 0 (depth bias 1 is enough).
- **Rendering**: 4x MSAA, HDR (RGBA16F) then tonemap, PBR (GGX) on, vsync off. Texture filtering is bilinear without
  mip interpolation or anisotropy (our sampler only sets min/mag filters); the URP template's PC quality level forces
  anisotropic filtering on every texture.

### Comparing

- **LODs**: brutus generates 8 LOD levels at load with meshoptimizer and picks them at 1 px error (shadows: 2 cascade
  texels on a position-only chain). Godot 4 also generates LODs on import with meshoptimizer and picks them by screen
  space error. Unity 6.2+ has Mesh LOD (`MeshLodUtility.GenerateMeshLods`, levels halving the triangle count, selected
  by screen size through a per-mesh curve scaled by `QualitySettings.meshLodThreshold`), but its default threshold
  selects far finer levels than our 1 px (445 renderers at LOD 0 in the benchmark view against our 47): threshold 2 is
  the closer visual match. Its shadow cascades reuse the camera's LODs, which is most of the remaining gap.
- **Culling**: brutus does GPU frustum + occlusion culling per mesh and per meshlet; the other engine's culling is
  part of what's being compared. Unity's GPU Resident Drawer culls per instance; its GPU occlusion culling gains
  nothing in this crowd (same as our mesh-level HZB), its shadow caster culling is better than ours (see Leads from the
  Unity comparison).
- **Measure**: same GPU, resolution and fixed camera; GPU frame time (brutus numbers here are a Debug build, compare
  against a release build of brutus too), triangles rendered and draw calls from each engine's own stats or a capture.
  Our scripts work on another engine's player: `util/gputrace.py --exe <player> --args <...>` and
  `renderdoc_capture.py` with `BRUTUS_EXE` / `BRUTUS_ARGS`; prefer them to the engine's counters (Unity's editor
  triangle counts sum all passes and weren't reliable). Run the player in exclusive fullscreen (a borderless window can
  be paced by the compositor at 60 Hz) and repeat runs or trace at locked clocks: at boost clocks the same Unity
  configuration measured anywhere from 16.2 to 18.1 ms across runs.

## Working process

Conventions (also in the assistant memory): clang-format every changed C++ and GLSL file with `--style=file`; comments
only for what names/types don't say (contracts, special cases); `size_t` for index loops; `auto` / `const auto`, never
`auto*`; commits end with `Assisted-By: Claude Opus 5.5 <noreply@anthropic.com>` (`Co-Authored-By` only for fully
generated tooling); commit vk-renderer first then bump the submodule; nothing is pushed by the assistant.
`core.autocrlf = true`: working copies are CRLF, git stores LF.

Verification loop used for every change, from the repository root:

1. `cmake --build build --config Debug --target brutus-vulkan` (check the exit code, not only the last output line)
   and `python util/compile_shaders.py` (shaders are compiled at runtime, a failed first compile throws and shows an
   `abort()` popup).
2. Smoke run for validation output (nothing printed is clean): `timeout 25 ./build/Debug/brutus-vulkan.exe --benchmark >
   out.txt 2>&1` from Git Bash (from PowerShell use `C:\Program Files\Git\usr\bin\timeout.exe`, Windows' `timeout`
   is something else). Exit code 124 means it was still running when stopped.
3. Correctness: `util/screenshot_benchmark.ps1` + `util/image_diff.ps1` against the previous build; refactors must be
   pixel identical (frames are deterministic). Check both paths: add `--use-vertex-shader` to the arguments for the
   vertex path (every script below takes extra arguments). If a diff shows only a strip at the bottom, the taskbar was
   captured: retake.
4. Counters: `util/renderdoc_capture.py` (`passes.txt` has per-pass rasterized primitives, fragment/task/mesh
   invocations; `actions.txt` shows the indirect draws with their group counts). Reading buffers from a capture (e.g.
   indirect commands) works from a qrenderdoc Python script using `BufferDescription.gpuAddress` and push constants.
5. Cost: `python util/gputrace.py --out <dir> --metrics`, compare against a previous trace dir. Use 10 frames (two
   runs) when a difference is small: the main pass varies +-0.3-0.5 ms frame to frame.
6. Prefer command line options to source edits for comparisons (`brutus-vulkan.exe --help` lists them): they need no
   rebuild and can't be left behind. When a test or benchmark keeps needing the same edit, suggest an option for it.
   Temporary experiments are marked `// TEMP EXPERIMENT, revert`, checked with `grep -rn "TEMP EXPERIMENT" src shaders`.
   Back up the file first, restore from the backup and **touch it** (`(Get-Item f).LastWriteTime = Get-Date`):
   `Copy-Item` keeps the old timestamp and MSBuild then skips recompiling. Never `git checkout` a file with
   uncommitted work. Confirm runtime settings (mesh shading...) in the UI before trusting measurements.
7. Editing: prefer the Edit tool; Git Bash `sed -i` converts CRLF files to LF (restore with PowerShell `ReadAllText`,
   `-replace "\r?\n", "\r\n"`, `WriteAllText`). PowerShell `Set-Content -Encoding utf8` writes a BOM that the runtime
   shader compiler rejects.
8. Reproducing a manual view: the user pastes the UI's "Copy camera" snippet, swapped into the `is_benchmark` setup as
   a temporary experiment.
9. Debugging crashes/`abort()` popups: `cdbX64.exe` (WinDbg store app) from PowerShell with a command file, e.g.
   `sxe -c "kn 30;gn" eh` and `bu ucrtbased!abort "kn 40;q"`, `-y` pointing at `build/Debug`. A throw from
   `waitForFences` means device lost: look for out-of-bounds buffer reads.

Other gotchas:
- RenderDoc swapchain images are saved with an "undefined contents" pattern: use screenshots for final frames.
- RenderDoc per-draw counter values shuffle between replays (culling atomics change draw order), only per-pass totals
  are stable. RenderDoc replays counters for a few minutes on large captures, it isn't stuck.
- Every app run overwrites `startup_debug.opt`; runs under RenderDoc/Nsight distort startup times.
- Bistro takes much longer to load (texture decompression): give screenshots 45-60 s.
