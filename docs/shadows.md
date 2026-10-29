# Shadows

Notes for the shadow work (cascaded shadow maps, caster culling, filtering): what's done, the current state, the plan
and the findings. Shared rules and measurement setup: [working_process.md](working_process.md). Draw commands,
culling and LOD machinery shared with the main pass: [draw_performance.md](draw_performance.md).

Status (2026-10-29): shadows are fast (~0.64 ms) and visually acceptable; SDSM and contact shadows are tabled while the
early opaque pass is the focus.

## Done so far

| Commit | Change |
| --- | --- |
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
| `a1875b4` | Shadow caster culling against each cascade's receivers (`shadow_caster_visible` in `constants.glsl`), per draw and per meshlet: shadow map 1.71 -> 0.67 ms, pixel identical |
| `7592996` | Shadow culling batched: one commands buffer per cascade, all culling dispatches then all draws (shadow map 0.67 -> 0.62 ms) |

## Current state

- **Cascades**: 4 x 2048² D16 array (`SHADOW_CASCADE_MAX`), practical split scheme (lambda 0.75) from 0.1 m to the
  shadow distance: about 1.9 / 4.3 / 9.4 / 25 m at the default 25 m. Each cascade is a bounding sphere fit of its view
  depth slice with texel snapping (`setup_shadow_cascades`). Splits depend on the shadow distance, so a larger distance
  also enlarges the near cascades.
- **Selection** in `shadow.glsl`: by view depth against `shadow_cascade_splits`. Each cascade lerps into the next over
  the last `SHADOW_CASCADE_BLEND_RANGE` (25%) of its slice, keeping the current one where the next doesn't cover the
  point; the last cascade uses the distance fade.
- **Constants**: `FrameConstants` (main camera, lighting, shadow sampling, debug; written once at the start of
  `render`, after `setup_shadow_cascades`) and `PassConstants` (view, render target, shadow pass only fields; one per
  pass: main view and each cascade), separate push constant pointers. Fragment shaders only read the frame constants,
  culling/task/mesh/vertex shaders only the pass constants.
- **Bias**: no receiver depth bias. Normal offset of `SHADOW_NORMAL_OFFSET_TEXELS` (1.5) texels of the selected cascade,
  scaled by `sin(angle to light)`. Caster side: rasterizer bias `SHADOW_DEPTH_BIAS_CONSTANT` 1 /
  `SHADOW_DEPTH_BIAS_SLOPE` 2 (D16 units). No acne on the back view (`{ -4.7130, 1.6786, 3.6568 }`, pitch -0.0140, yaw
  1.8799), no peter-panning at ground level (user checked). `compute_shadow` gets the interpolated vertex normal, not
  the true geometric one.
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
- **Cost** (Release, 2 x 10 traced frames, `64ca18b`): shadow map ~0.64 ms: culling 0.074 ms (all cascades, batched),
  cascades 0.075 / 0.090 / 0.143 / 0.25 ms. Draws bound by depth writes and raster (ZROP 49-60%, raster 33-42%, SM
  4-11%, SMs idle 8-40% in the near cascades): a depth-only shadow mesh shader wouldn't help, fewer depth samples and
  triangles would. History: 1.71 ms before caster culling (`a1875b4`, depth samples 18.0M -> 5.57M, Unity 12.4M,
  rasterized primitives 3.69M -> 1.31M), 0.67 -> 0.62 ms with batched culling (`7592996`: each culling dispatch is a
  single latency-bound wave, so batching overlaps them and avoids compute/graphics switches).
- **Terminator** (`cbcc6d8`): `compute_shadow` skips the lookup when the geometric normal faces away from the light;
  callers must give those surfaces no direct light (`bump_terminator_shadowing` in `mesh.frag`). Repro of the old leak:
  `{ -7.2700, 2.0876, 3.5322 }`, pitch -0.2620, yaw 2.5379.
- **Visible issue: popping at cascade switches**, mostly 0 -> 1 at ~1.9 m: casters thinner than a texel (bangs strands
  seen from the light) are rasterized in cascade 0 (~2 mm texels) but drop out of cascade 1 (~4.3 mm), so the shadow
  shrinks. Confirmed with 4096² maps (shadow stays); normal offset, filter and caster LOD each ruled out. Repro:
  `{ -7.1963, 2.4145, 5.0727 }` vs `{ -7.4208, 2.4759, 5.2944 }`, pitch -0.1920, yaw 0.7919. The benchmark is a
  dialogue-distance shot (main character at 1.5-2.5 m, ~51° vertical FOV) and cascade 0 covers 0.1-1.9 m, often
  nothing visible. Also terminator facets.
- **Not a shadow issue**: the seam and light/dark split on Miku's upper arm at `{ -5.4757, 2.1348, 3.6858 }`, pitch
  -0.4440, yaw -1.8301 come from the mesh normals (visible without shadows; normal offset, filter and 4096² ruled out).
  Fix in the asset, or weld vertices and merge normals at load.

## Plan

### Quality

1. **Resolution where characters are**: depth-bounded cascades (SDSM: fit splits to the visible depth range) or fixed
   small near cascades. 4096² works but is 128 MB and only moves the problem. Skip cascades with nothing visible.
2. **Screen-space contact shadows** for strand-scale features (bangs, collar) that no reasonable shadow map catches.
3. Optional: per-character shadow maps for close-ups, 4096² as an "ultra" setting.

SDSM plan (agreed 2026-10-27, tabled): first `--shadow-near` / `--shadow-split-lambda` options to measure quality and
cost, then a min/max reduction of the previous frame's depth (the depth pyramid holds only early pass characters, and a
single reduction) read back with 1-2 frames of latency. Splits from the measured range, but cascade 0 keeps starting at
the camera so nothing new is unshadowed. In this scene only the near bound helps (the ground fills the view up to the
shadow distance). Tighter (non sphere) fits are the bigger resolution lever, at the cost of rotation stability.
Receiver planes could also use the measured range (fewer casters).

### Cost

1. **Far cascade**: coarser LOD multiplier (currently `lod_target = 2 * texel_world_size` everywhere), or update it
   every 2-4 frames.
2. **Exact sample test for tiny shadow triangles**: when the box spans exactly one texel center per axis, edge test it.
3. One culling dispatch testing each draw against all cascades (reads each draw once).
4. A depth-only `mesh.mesh` variant for shadows (it fetches the full vertex and writes every attribute today) is
   unlikely to help: the cascades aren't shader bound, and the driver may already strip unused outputs (unchecked).

### Backlog

- Alpha-tested materials don't cast shadows (needs a shadow pipeline with a discarding fragment shader, matters for
  Bistro). The second model's hair textures have 0 alpha (export bug): ~2000 invisible alpha draws are expected.
- Fixed near cascade sizes independent of shadow distance (if wanted).
- Possible ray traced shadows later as an optional path (no acceleration structure support in vk-renderer yet).

## Leads from the Unity comparison

Measured against the Unity 6.3 URP reproduction at `47c6203` (setup in [benchmark.md](benchmark.md)).

- **Done (`a1875b4`)**: cull shadow casters already covered by a closer cascade. Cascades 0-2 were depth fill bound
  (ZROP 58-60%, PES 7-12%, SMs idle 24-38%). Unity wrote 12.4M shadow depth samples against our 18.0M (cascade 3: 1.37M
  against 3.90M) because its caster culling (`ReceiverSphereCuller.ComputeSplitVisibilityMask` in URP's core package)
  treats each caster as a capsule along the light clipped to the camera frustum. Ours now does the same against each
  cascade's receivers (5.57M samples).

## Key findings

- The shadow pass is limited by **per-triangle cost and depth fill**, not by vertex work or bandwidth: cutting
  triangles and depth samples is the lever. Remapping indices or D16 didn't change time; position-only LODs, small
  triangle culling and caster culling did.
- The geometry path doesn't matter for shadows: the vertex path took 1.67 ms against 1.71 ms for mesh shaders (before
  caster culling), with more triangles (4.21M against 3.69M) and the same ZROP-bound profile.
- Batching the four culling dispatches helped (they overlap, and fewer compute/graphics switches); the four cascade
  draws don't overlap measurably, but they're bound by shared fixed-function units anyway.
- Most test assets are double sided: the shadow pass culls nothing by facing.
- Normal mapped lighting needs a geometric term: a surface whose geometry faces away from the light is in its own
  shadow whatever the normal map says (the leak existed before shadow maps, they made it visible).
