# Benchmark scene

The `--benchmark` scene, shared by every feature stream, and how to reproduce it in another engine for comparisons.

## Reproducing the benchmark in another engine

Goal: the same scene in Unity or Godot to compare frame time, triangle counts and visuals. Everything below is what
`main.cpp` / `scene_renderer.cpp` set up for `brutus-vulkan.exe --benchmark` (the crowd, without `--scene`), with what the Unity port
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
  can't regenerate the same crowd. Export it instead: `brutus-vulkan.exe --dump-scene <file>` (no device
  created, refused with `--scene`) writes the benchmark crowd as CSV, one line per character: asset path, position, rotation quaternion
  (x, y, z, w), scale, in glTF conventions. It shares `generate_crowd()` with the renderer, so it stays in sync.
- **Meshes keep their glTF node transforms**: each asset's node hierarchy is flattened at load, then every mesh's
  world transform is combined with the character's translation/rotation/scale (skinned meshes sit at the asset's
  root, as glTF specifies). For both assets the node transforms cancel out (Sketchfab's Z-up rotations): Miku's
  meshes are skinned, Sabrina's world transforms are identity within 1.6e-7, so an engine that imports the hierarchy
  or takes each mesh at identity matches. Until `b3880f4` the transforms were ignored; taking them into account
  changed 673 isolated pixels of the benchmark frame.
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
- **Projection**: vertical FOV **50.70°** (`Camera::DEFAULT_FOV`: it used to be 70 passed to `tan()` as radians, `a1cfe26` kept the angle), 1920 x 1080 (aspect 16:9, 80.2° horizontal), near
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
  nothing in this crowd (same as our mesh-level HZB), its shadow caster culling was better than ours until `a1875b4` (see
  [shadows.md](shadows.md)).
- **Measure**: same GPU, resolution and fixed camera; GPU frame time (compare against a Release build of brutus, at
  locked clocks, see [working_process.md](working_process.md)), triangles rendered and draw calls from each engine's own stats or a capture.
  Our scripts work on another engine's player: `util/gputrace.py --exe <player> --args <...>` and
  `renderdoc_capture.py` with `BRUTUS_EXE` / `BRUTUS_ARGS`; prefer them to the engine's counters (Unity's editor
  triangle counts sum all passes and weren't reliable). Run the player in exclusive fullscreen (a borderless window can
  be paced by the compositor at 60 Hz) and repeat runs or trace at locked clocks: at boost clocks the same Unity
  configuration measured anywhere from 16.2 to 18.1 ms across runs.
