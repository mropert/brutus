# Scenes

Notes for loading and describing what's rendered: the benchmark crowd and glTF scenes such as Amazon Lumberyard
Bistro. Shared rules and measurement setup: [working_process.md](working_process.md), the crowd benchmark:
[benchmark.md](benchmark.md).

## Done so far

| Commit | Change |
| --- | --- |
| `8508c2b` | (user) `Assets` owns meshes, materials and the shared index buffer, looked up by bindless handle |
| `b3880f4` | Loader arrays indexed like the glTF ones (with gaps for unused entries), only loaded entries reach `Assets`: an unused mesh or material used to shift lookups |
| `682c29a` | `Assets::load` flattens the default scene's node tree into `scene_instance`s (mesh handle + world TRS); the crowd composes them with each character's transform. Crowd benchmark: 673 isolated pixels changed (Sabrina's 1.6e-7 node transforms), Bistro identical |
| `1be485f` | `--scene <path.gltf>` replaces `#define BISTRO`, the crowd stays the default; `--dump-scene` refused with `--scene` |
| `613ac04` | glTF camera and sun: the scene starts at its first perspective camera (no roll, its vertical FOV, now `Camera::fov`) and its first directional light sets the sun orbit's elevation and starting azimuth. Crowd unchanged |

## Current state

- **Loading**: `Assets::load( path )` returns a `scene`: a flat list of instances, an optional `Camera` and an
  optional sun direction. Only the default scene (scene 0 if none) is walked, and only its meshes are loaded. Node
  transforms are composed as matrices and decomposed once per mesh (`glm::decompose`, skew is lost); skinned meshes
  sit at the scene root, as glTF specifies.
- **Instances refer to meshes by handle**: `Assets` owns all GPU data and nothing is unloaded, so handles stay valid
  and `scene`s can be copied freely. Unloading would need bindless entry removal, an index buffer allocator and
  deferred destruction.
- **The crowd** loads each model as a `scene` and builds its own by composing the character's transform with each
  instance's: translation `t + r * ( s * t_mesh )`, rotation `r * r_mesh`, scale `s * s_mesh` (exact, the
  character's scale is uniform).
- **Camera**: position, yaw and pitch from the camera node's world -Z; roll is dropped (`Camera` has none). glTF
  `yfov` replaces the crowd's 0.884962; near plane (0.01) and infinite far are ours, the window gives the aspect.
- **Sun**: the orbit keeps a constant elevation and turns the azimuth over `day_period` (it never sets). A glTF
  directional light gives `elevation = asin( to_sun.y )`, `start azimuth = atan2( to_sun.z, to_sun.x )`, the exact
  inverse of the arc, so time 0 is the file's direction. glTF scenes start paused. Intensity stays on the UI slider
  (the file's lux aren't mapped to our units), color is ignored.
- **Ground grid**: a UI toggle, on for the crowd and off for glTF scenes (they have their own ground). The grid pass
  also resolves MSAA, so it runs without the grid: its "Grid draw" timing is then the resolve alone.
- **One primitive per mesh**: the loader throws otherwise, and a mesh carries its material. No file we load shares
  vertex data between meshes or primitives.

## Bistro

- `../niagara_bistro/bistro.gltf` (zeux's lightly edited copy of NVIDIA's RTXDI Bistro): 5,910 nodes, 2,909 mesh
  nodes over 551 meshes (one primitive each), 254 materials. VRAM ~3.2 GB, 30.7M of the 32M index buffer.
- **Camera**: pitch -1.83°, roll -2.05° (dropped), yfov 0.628 (36°). The roll looks accidental (hand placed in a DCC
  tool), game cameras stay level.
- **Sun**: `KHR_lights_punctual` directional, 6830 lux, elevation 68.5°, azimuth ~130°.
- **Materials** (next): 234 use `KHR_materials_pbrSpecularGlossiness` (`*_diff.png` diffuse, `*_spec.png`
  specular-glossiness, factors), 18 `KHR_materials_transmission`, 2 none. fastgltf 0.9.0 compiled
  specular-glossiness out (`FASTGLTF_ENABLE_DEPRECATED_EXT`, off in vcpkg); 0.9.1 (2026-09-27) always has it, after
  the user's report. `vcpkg.json` pins fastgltf to 0.9.1 with an `overrides` entry, ahead of our baseline: drop it
  when the baseline moves past it.

## Plan

- **Bistro materials**: parse `KHR_materials_pbrSpecularGlossiness` (fastgltf 0.9.1), then a specular-glossiness
  shading path. Check what Lumberyard's `_spec` textures hold first.
- **Match a reference image** once materials work: ORCA's daytime
  [Bistro_Exterior_1](https://d29g4g2dyqv443.cloudfront.net/sites/default/files/akamai/Bistro_Exterior_1.png)
  (corner of "Le Petit Coin" with the Vespa, camera not published: fly there and "Copy camera"). Its verticals look
  level. Other candidates don't fit: ORCA's Exterior_2 (under the awning), WurblPT's exterior (night).

## Backlog

- **Shared geometry**: primitives sharing vertex accessors, or meshes differing only by material, would duplicate
  vertex data. `MeshDraw` already carries mesh and material separately: move the material to `scene_instance`, one
  instance per primitive, cache geometry by accessors; primitives sharing vertices but not indices could share one
  vertex buffer with their own meshlets.
- **Day arc**: rotating about a tilted axis would make the sun rise and set, but needs below-horizon handling and a
  matching ambient.
- **Physical sun intensity**: map lux with an exposure model.

## Key findings

- pbrt-v4's `bistro_vespa` camera (eye `(-9.7, 2.5, 4.3)`, 45° vertical) doesn't match this glTF's coordinates, nor
  mirrored in X: the pbrt and RTXDI conversions don't share a frame.
- vcpkg can pin one port without forking: an `overrides` entry in `vcpkg.json` takes any version from the
  registry's version history (newer than the baseline too), the rest stays on the baseline. For changes to a port's
  build (e.g. CMake options), an overlay port (`"overlay-ports"` in `vcpkg-configuration.json`, the port's folder
  copied into the repository) replaces the recipe.
- The ORCA README says its re-authored "Specular" textures pack occlusion, roughness and metalness; this glTF's
  Lumberyard `_spec` textures may differ.
