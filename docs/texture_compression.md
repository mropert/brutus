# Texture compression

Compressing decoded (PNG/JPG) textures to BC formats on the GPU at load time, with compression code extracted from
[Betsy](https://github.com/darksylinc/betsy). Shared rules and measurement setup: [working_process.md](working_process.md).
Scenes and their textures: [scenes.md](scenes.md).

## Done so far

Nothing implemented yet. This file holds the plan.

## Current state

- Decoded textures (`image_usage::DECODED`) are uploaded as RGBA8/BGRA8 (or 565/5551) with a full mip chain blitted
  on the GPU (`gpu_upload_job::queue( const SDL_Surface&, ... )` in `src/asset.cpp`).
- `MSFT_texture_dds` images are BC7 with offline mips, copied as is (`gpu_upload_job::queue( const DDSFile&, ... )`).
- **Target: the crowd.** `player.glb` and `sabrina.glb` embed 20 PNGs, all 2048x2048 (13) or 2048x1024 (7):
  ~66M texels at mip 0, ~350 MB as RGBA8 with mips. Bistro has a BC7 DDS for every texture (normals included) and needs
  no recompression; its PNGs (7.8 GB) only serve as a stress test and a comparison against offline BC7.
- `Texture::Format` (vk-renderer `src/renderer/texture.h`) has BC7 as its only block format.

## Plan

### What Betsy implements

Taken from `master` at `cc723dc` (2022-10-25). The last tag, `v1.1-beta` (2020-10-25), has the same encoding code: the
9 commits since change one shader line (`layout( location = 0 )` on BC1's refinement uniform, which we drop), the
license's date range ("2020-present" -> "2020-2022"), and host code, build fixes and docs we don't use.

The README lists BC1, BC3 ("just BC1 for RGB + BC4 for alpha"), BC4, BC5 (UNORM and SNORM), BC6H (unsigned), ETC1,
ETC2, EAC. **No BC2 and no BC7.**

- BC1: `bin/Data/bc1.glsl`, a port of stb_dxt (PCA endpoints, 1D index matching, least squares refinement).
  `bc1_dither.glsl` is just `#define BC1_DITHER` + `bc1.glsl`.
- BC4: `bin/Data/bc4.glsl`, stb_dxt's alpha block (min/max endpoints, Giesen's optimal index rounding), 4 threads per
  block with shared memory and atomics.
- BC3 and BC5 are not shaders of their own: the host (`src/betsy/EncoderBC1.cpp`, `EncoderBC4.cpp`) dispatches BC1 +
  BC4 (alpha) or BC4 twice (R, G) into `RG32_UINT` images, a third pass (`etc2_rgba_stitch.glsl`) packs them into
  `RGBA32_UINT`, then `glCopyImageSubData` copies that into the BC image.

Formats we use: BC1 (opaque color), BC3 (color with alpha we read), BC5 (normal maps), BC4 if a single channel use
comes up. **BC2 is skipped**: same size as BC3 with 16 fixed alpha levels (4 bits per texel) instead of two 8 bit
endpoints and 8 interpolated levels; BC3 is as good or better on smooth alpha and on alpha tested cutouts, which is
all we have.

### License and provenance

- Betsy: MIT, "Copyright 2020-2022 Matias N. Goldberg".
- BC1 and BC4 code: from stb_dxt (fabian "ryg" giesen, ported to C by Sean Barrett), MIT or public domain at our
  choice; Betsy's README credits it. `src/betsy/BC1_tables.inl` (the `stb__OMatch5` / `stb__OMatch6` single color
  tables) was generated from stb_dxt too.
- Each extracted file starts with: where it comes from (repo URL, commit `cc723dc`, original path), the Betsy MIT
  notice in full, the stb_dxt credit, and a line listing our changes. Betsy's `LICENSE.md` is also kept as is in
  `shaders/betsy/` (imported with the sources, see Order of work).

### Includes

Plain functions, no bindings, no `main`, no `#version`:

- `shaders/betsy/bc1.glsl`
  - From `bc1.glsl`: `rgb565to888`, `rgb888to565`, `lerp13`, `EvalColors`, `OptimizeColorsBlock`, `MatchColorsBlock`,
    `RefineBlock`, and Betsy's `main` body turned into `uvec2 bc1_compress_block( uint pixels[ 16 ], uint refinements )`
    (pixels packed with `packUnorm4x8`, as Betsy does; returns the 8 byte block).
  - The `c_oMatch5` / `c_oMatch6` SSBO becomes two `const uint` arrays of 256 entries in the file (each entry is two
    values under 64, packed in one uint) generated from `BC1_tables.inl`, so the include needs no binding.
  - The HLSL style macros (`float3`, `uint2`, `max3`, ...) from `CrossPlatformSettings_piece_all.glsl` are replaced by
    GLSL types during extraction instead of being carried over.
  - Dithering left out at first: `MatchColorsBlock`'s dithered path loads pixel `y * 4 + 2` twice instead of
    `y * 4 + 3` (a Betsy bug), and dithering is off by default there. Fixed if we ever try it.
- `shaders/betsy/bc4.glsl`
  - `uvec2 bc4_compress_block( float values[ 16 ] )` (UNORM only; SNORM if a use appears): same min/max, bias and index
    selection as Betsy, written for one thread per block instead of 4 threads + shared memory + atomics. Same output,
    and it lets BC3 and BC5 be one thread per block without a stitch pass.
  - Values rounded to integers after `* 255` (Betsy doesn't round, the bias thresholds compare exact integers).
- No `common.glsl`: `min3` / `max3` are local helpers.

BC3 = `uvec4( bc4_compress_block( alpha ), bc1_compress_block( rgb ) )` (alpha block first), BC5 =
`uvec4( bc4( r ), bc4( g ) )`. BC1's block is written with `max >= min` endpoints, so it decodes the same in BC3
(which always uses 4 color mode).

### Our dispatch shader

`shaders/bc_compress.comp`, compiled 3 times by the `PipelineManager` with `BC1`, `BC3` or `BC5` defined (`SRGB` too,
see below):

- `local_size 8 x 8`, one thread per 4x4 block, one dispatch per mip.
- Push constants: source `textures[]` index, mip, mip extent, blocks per row, output address (`buffer_reference`,
  like `shadow_cascades.comp`), BC1 refinement count (Betsy uses 2).
- Loads the 16 texels with `texelFetch( sampler2D( textures[ src ], samplers[ 0 ] ), clamp( pos, 0, extent - 1 ), mip )`
  (edge texels replicated for the last mips, smaller than a block: 2x2, 2x1, 1x1).
- sRGB: the temporary source keeps today's `*_SRGB` format so the blits filter in linear space (mips identical to
  today's), the fetch returns linear values and the shader re-encodes to sRGB before quantizing to 8 bits; BC1/BC3
  then encode the gamma values, written to a `BC1_RGB_SRGB` / `BC3_SRGB` texture. To check: the 8 bit round trip is
  exact (all 256 values through a test dispatch); if not, vk-renderer needs `MUTABLE_FORMAT` views and we read a UNORM
  view instead.
- Writes `uvec2` (BC1) or `uvec4` (BC3, BC5) at `block_y * blocks_per_row + block_x`, mips one after the other: the
  layout `copy_buffer_to_texture` expects, as for DDS files.

Why a buffer and not Betsy's `RG32_UINT` storage image + image copy: vk-renderer has `copy_buffer_to_texture` per mip
already (with the block alignment asserts), no image to image copy, and our bindless storage images are declared
`r32f`. A buffer needs neither a new image view type nor `BLOCK_TEXEL_VIEW_COMPATIBLE`.

Mips are generated first, on the uncompressed temporary texture, then each level is compressed. Blitting can't
produce BC data (BC formats never have `BLIT_DST`), and downscaling a compressed mip 0 would compound its error.

### Hooking into `src/asset.cpp`

The loop in `upload_textures` stays as it is: `make_image_from_surface` returns the final BC texture,
`_texture_upload.queue( *surface, tex )` does the compression, `add_texture` takes the result.

1. **A semantic usage per texture.** `texture_usage` (today `UNUSED` / `LINEAR` / `SRGB`) becomes what the shaders
   read the texture as, which decides sRGB, the compressed format, the uncompressed format (every decoded surface is
   RGBA32 now, so the surface no longer says) and the sRGB check of DDS files:

   | Usage | Shader reads | Compressed | Uncompressed |
   | --- | --- | --- | --- |
   | `ALBEDO` (opaque material) | RGB, sRGB | `BC1_RGB_SRGB` | `R8G8B8A8_SRGB` |
   | `ALBEDO_ALPHA` (`MASK` / `BLEND` material) | RGB sRGB, alpha | `BC3_SRGB` | `R8G8B8A8_SRGB` |
   | `NORMAL` | RG (z rebuilt) | `BC5_UNORM` | `R8G8B8A8_UNORM` |
   | `METALLIC_ROUGHNESS` | G, B | `BC1_RGB_UNORM` | `R8G8B8A8_UNORM` |
   | `OCCLUSION` | R | `BC1_RGB_UNORM` (BC4 later) | `R8G8B8A8_UNORM` |
   | `OCCLUSION_METALLIC_ROUGHNESS` (one texture) | R, G, B | `BC1_RGB_UNORM` | `R8G8B8A8_UNORM` |
   | `EMISSIVE` | RGB, sRGB | `BC1_RGB_SRGB` | `R8G8B8A8_SRGB` |
   | `SPECULAR_GLOSSINESS` | RGB sRGB, alpha (glossiness) | `BC3_SRGB` | `R8G8B8A8_SRGB` |
   | `SPECULAR` (`KHR_materials_specular`) | A | `BC3_UNORM` (BC4 later) | `R8G8B8A8_UNORM` |

   - sRGB formats only decode RGB as sRGB, alpha stays linear: what alpha and glossiness need.
   - One texture referenced with two usages: `ALBEDO` + `ALBEDO_ALPHA` -> `ALBEDO_ALPHA`, `METALLIC_ROUGHNESS` +
     `OCCLUSION` -> `OCCLUSION_METALLIC_ROUGHNESS` (the shader already reuses that sample); any other mix throws, like
     the mixed sRGB case today.
   - BC4 for `OCCLUSION` / `SPECULAR` puts the value in `.r`, `mesh.frag` reads specular from `.a`: needs a shader
     change or swizzled views (not in vk-renderer), later.
   - All-RGBA8 drops today's 565 / 5551 uploads in uncompressed mode; nothing in the crowd uses them.
   - DDS images are uploaded as is; the usage only gives the sRGB expectation the parser checks.
   - **Power of two only**: the compression path asserts both sides are powers of two (not necessarily square; the crowd
     has 2048x1024). Every blit is then an exact 2:1 and no edge texel is read from outside the real extent. The
     uncompressed path keeps loading any size.
2. **Upload**, per texture, in the job's command buffer (no new architecture):
   - copy into the top left of the matching scratch image (see Resource management), mip blits on sub-rectangles;
   - `barrier( TRANSFER, COMPUTE )`, bind the `bc_compress` pipeline, one dispatch per mip into a device local scratch
     buffer (`STORAGE_BUFFER | TRANSFER_SRC | SHADER_DEVICE_ADDRESS`, e.g. 64 MB; BC output is 1/8 to 1/4 of the RGBA8
     input, so it fills more slowly than the 128 MB staging buffer);
   - `barrier( COMPUTE, TRANSFER )`, `transition_texture` + `copy_buffer_to_texture` per mip into the final texture.
   - `flush_full` also triggers when the scratch buffer is full.
3. `Assets` needs the `PipelineManager` (constructed just before it in `main.cpp`).
4. **Command line**: `--texture-compression none|gpu` (default `none` until measured, then flip it) and `--no-dds`
   (ignore `MSFT_texture_dds`) so Bistro can compare PNG, PNG + GPU BC and DDS BC7 in one build.

**Kept simple on purpose.** Every crowd texture's mip 0 is at least 131k blocks (~2k groups of 64 threads), several
times what a 2080 SUPER keeps resident (~50k threads), and mips 0 and 1 are ~94% of a texture's blocks. Only the
levels at 256x256 and below leave the GPU partly idle, a few microseconds each, ~0.1 ms per texture. The job already
records a whole load in one command buffer and only submits early when the staging buffer is full. Estimates to
confirm with timestamps: BC1 ~0.2-1 ms per 2k texture with its mips (BC3/BC5 ~1.5-2x), the crowd's 20 textures
~10-20 ms of compression, small next to its PNG decoding (tens of ms per texture on the CPU) and the PCIe copies.
Only if measurements show otherwise:
- batch by phase (all copies, blits level by level across textures, all dispatches without barriers between them,
  all final copies) so small mips overlap;
- a background compressor that the render loop waits on before its first frame.

### Resource management

**Scratch images, created once per load.** The same idea as virtual texturing systems compressing fixed size pages
into a fixed scratch area (id Tech 5, Far Cry 4's adaptive virtual textures, Unreal's runtime virtual textures):

- Two images, `R8G8B8A8_UNORM` and `R8G8B8A8_SRGB`, `TRANSFER_SRC | TRANSFER_DST | SAMPLED`, full mip chain, sized to
  the largest decoded texture of the load (sizes are known after decoding, before the upload): 2048x2048 for the
  crowd, ~22 MB each with mips. Two because blits filter in the image's own format, not a view's: one image with
  `MUTABLE_FORMAT` views would filter sRGB mips in gamma space. They can share one allocation since only one is used
  at a time (~22 MB instead of ~44 MB for the crowd); first version: two plain allocations.
- Owned by `gpu_upload_job`, with their bindless slots added once, destroyed with the job.
- SDL converts every surface to RGBA32 (instead of BGRA32 / 565 / 5551 today), so one format per scratch image fits
  all.
- The real extent goes in the compress shader's push constants, its fetches clamp to it.
- vk-renderer: `copy_buffer_to_texture` with an explicit extent (copies into the top left of mip 0), `blit_texture`
  with explicit source and destination extents (mip n's real size inside the scratch image's mip n).
- **Why power of two only**: with a whole image, linear blits clamp at the image edge; with a sub-rectangle, non power
  of two sizes (1365 -> 682) sample between texels at the edge and can read the texel just outside, left over from
  the previous texture. Exact 2:1 levels only read the 2x2 inside. If a non power of two asset needs compressing: copy
  one more row and column replicating the edge, or accept a 1 texel seam.
- **Serial textures**: the next texture's copy waits for the current one's dispatches to finish reading
  (`barrier( COMPUTE, TRANSFER )` before the copy, then `transition_texture( UNDEFINED, GENERAL )`). The PCIe copy of
  one texture no longer overlaps the compression of the previous one: ~20 x (1-1.5 ms copy + up to ~1 ms compression)
  ~ 50 ms for the crowd. If it shows, two pairs of scratch images used in alternation.
- A texture bigger than the scratch image can't happen (sized to the largest); a cap (e.g. 4096) with a dedicated
  temporary above it only if a huge texture makes the scratch size unreasonable.

Rejected for now: a temporary texture per texture, released after each fence wait. It needs a VMA allocation per
texture and `remove_texture` with a free list of bindless slots. Its later refinement, temporaries aliased on one
worst case allocation (`vmaCreateAliasingImage`), would follow Vulkan's rules: creating and binding an image are host
operations, allowed while the memory is still in use by pending work; first use needs an execution dependency from
the previous image's last access and a transition from `UNDEFINED`; destroying needs every submitted command using it
to have completed, so handles and slots live until the fence.

**Bindless descriptors: a vk-renderer change is required.** The bindless set is created without
`UPDATE_AFTER_BIND` and `PARTIALLY_BOUND` on its bindings, layout and pool.

- `UPDATE_AFTER_BIND`: by default, the driver may snapshot descriptors at bind time, so updating a set that is bound
  in a recording (or executable) command buffer invalidates that command buffer. With the flag, writes between bind
  and submit are legal and the submit uses the descriptors as they are then. Today the upload command buffer never
  binds the set, so `add_texture` during the load is fine; once it binds the compress pipeline for texture 1,
  `add_texture` for texture 1's final BC texture invalidates the recording.
- `UPDATE_UNUSED_WHILE_PENDING`: also lets us write descriptors while submitted work using the set runs, as long as
  that work doesn't use them (fill slot 500 while frames read 0-499, never overwrite a slot pending work samples).
  Not needed for the load, needed for streaming later, free to set now.
- `PARTIALLY_BOUND`: by default every element of an array a shader references must be valid (written, resource
  alive) whenever a draw or dispatch uses the set. Only the descriptors actually accessed need to be with the flag.
  Our `textures[]` has 8192 slots and only the loaded textures written, which strictly breaks the default rule today;
  the scratch images' slots go stale once the job is destroyed, which would make every later draw invalid without it.
- Setup: `eUpdateAfterBind | eUpdateUnusedWhilePending | ePartiallyBound` on the texture, image and sampler bindings,
  `eUpdateAfterBindPool` on the layout, `eUpdateAfterBind` on the pool, and the features
  `descriptorBindingSampledImageUpdateAfterBind`, `descriptorBindingStorageImageUpdateAfterBind`,
  `descriptorBindingUpdateUnusedWhilePending`, `descriptorBindingPartiallyBound`. vk-renderer already requires
  `descriptorIndexing`, which guarantees all four. Update after bind pools have their own limits
  (`maxDescriptorSetUpdateAfterBind*`), far above 8192 on desktop; those bindings can't be dynamic buffers (we have
  none in that set).
- All descriptors are written by the CPU, so no GPU barrier is involved. A descriptor-hazard barrier (Aaltonen's
  `HAZARD_DESCRIPTORS`, `VK_ACCESS_2_DESCRIPTOR_BUFFER_READ_BIT_EXT`) only matters when the GPU writes descriptors.

**Not now: descriptor buffers or heaps.** `VK_EXT_descriptor_buffer` (and its likely successor
`VK_EXT_descriptor_heap`, status and support to check) turn descriptors into plain memory the GPU can also write.
With the three flags above, our bindless set already behaves like an index table written by the CPU, and everything
else is already device addresses. Revisit for GPU-driven texture streaming (the GPU writing its own descriptors),
per-frame descriptor churn showing up in CPU time, or once descriptor heaps are supported on every GPU we target
(going straight to them rather than migrating twice).

### Normal maps in BC5

BC5 has no blue channel: `mesh.frag` reads `.xyz`. glTF normal textures are tangent space by spec: unit length,
pointing out of the surface (z >= 0), so blue is redundant and comes back as
`n.xy = t.xy * 2 - 1; n.z = sqrt( max( 1 - dot( n.xy, n.xy ), 0 ) )` (one `dot` and one `sqrt` per pixel). The
sources keep all 3 channels, we drop blue on purpose. This is the usual setup: BC5 (ATI's 3Dc originally) was made for
it, Unreal compresses normal maps to BC5, Unity too on desktop (its older DXT5nm put x in alpha and y in green for the
same reason). BC1 does badly on normals (3 channels on one 565 color line); BC7 keeps 3 channels at the same size but
BC5 spends its 16 bytes on two independent channels.

Doing it for every normal map (BC7 DDS and uncompressed too) avoids a per-material flag. It changes pixels wherever a
source normal isn't unit length, and in filtered mips (averaged normals are shorter, the rebuilt z treats them as unit
length; we already `normalize` after the TBN transform, so the change should be small). Measured on its own commit
before the compression lands.

### vk-renderer formats

Add to `Texture::Format`, `get_texel_block_size` (8 bytes for BC1/BC4, 16 for BC3/BC5) and `get_texel_block_extent`
(4), with `static_assert`s on mip sizes like BC7's:

- `BC1_RGB_UNORM`, `BC1_RGB_SRGB` (`eBc1RgbUnormBlock`, `eBc1RgbSrgbBlock`). The `RGB` is Vulkan's naming: BC1 has
  RGB and RGBA variants that decode a 3 color mode block's index 3 (black) with alpha 1 or 0, BC3 and BC5 have one
  variant each. RGB keeps alpha at 1 (no punch-through, a stray 3 color block can't cut a hole); a plain `BC1_UNORM`
  would read like DXGI's, which is the RGBA flavor. Betsy's output decodes the same either way.
- `BC3_UNORM`, `BC3_SRGB`
- `BC4_UNORM`, `BC5_UNORM` (SNORM variants only when needed)

All are covered by `textureCompressionBC`, already mandatory. The DDS parser can map these later if an asset needs it.

### Quality and speed

- BC1/BC3 color is 565 endpoints and 4 colors per block: clearly below BC7 on gradients and multi-colored blocks (BC7's
  modes and partitions). BC5 normals are as good or better than BC7 RGB normals (two independent 8 level channels).
- Knobs: BC1 refinement count (0 to 2+, Betsy defaults to 2, stops when the mask stops changing); dithering (off).
- Memory: the crowd goes from ~350 MB of RGBA8 with mips to ~45-90 MB depending on the BC1 vs BC3/BC5 split.
- **BC7 later**: same size as BC3/BC5, twice BC1, much better color. The format is open (D3D11 spec, Khronos Data
  Format Specification) and has MIT encoders; Betsy just doesn't have one. Candidate: DirectXTex's BC7 compute shader
  (HLSL, MIT), ported to a GLSL include the same way. The dispatch, buffer and upload plumbing here is reused.

### Validation

- **Encoder check** on its own: a debug mode that, after compressing, samples the BC texture and the temporary source
  per mip in a compute pass and accumulates squared error, logging RMSE/PSNR per texture (and the worst ones). Catches
  layout bugs (swapped endpoints, wrong block order, wrong mip offsets) and sRGB mistakes immediately.
- **Image diffs** (Release, the usual tools): the crowd benchmark none vs gpu; Bistro `--no-dds` none vs gpu, gpu vs
  DDS BC7. Pixel counts and max/mean deltas, as in scenes.md's PNG vs DDS entry.
- **Memory delta**: textures (UI), VRAM used / spilled, peak RAM while loading, and peak VRAM during the load (the
  temporaries).
- **GPU time**: timestamps around the compression dispatches (total and per format), an Nsight trace of a load;
  startup time; and the frame GPU time / Opaque Early draw.

### Order of work

1. vk-renderer: the formats; `UPDATE_AFTER_BIND` / `UPDATE_UNUSED_WHILE_PENDING` / `PARTIALLY_BOUND` on the bindless
   set; `copy_buffer_to_texture` and `blit_texture` with explicit extents.
2. `mesh.frag` z reconstruction, measured alone (diffs on Bistro DDS and the crowd).
3. Betsy import, verbatim: `bin/Data/bc1.glsl`, `bin/Data/bc4.glsl`, `bin/Data/CrossPlatformSettings_piece_all.glsl`,
   `src/betsy/BC1_tables.inl` and `LICENSE.md` from `cc723dc`, copied byte for byte into `shaders/betsy/` under their
   original names (no clang-format, an exception to the usual rule). Nothing includes them yet, so the build doesn't
   compile them. Its own commit, with the source commit in the message: it needs no review beyond checking it matches
   upstream (`git diff --no-index` against a clone).

   3a. clang-format only, its own commit: mechanical (re-running clang-format on the import reproduces it), so the next
   step's diff only shows real edits instead of Betsy's style turning into ours.

   3b. Our edits on top: provenance headers, HLSL style macros replaced by GLSL types, `main` bodies turned into
   `bc1_compress_block` / `bc4_compress_block`, BC4 to one thread per block, the rounding after `* 255`, the
   `BC1_tables.inl` data as packed `const uint` arrays, the dither path removed; then deleting the files we no longer
   need (`CrossPlatformSettings_piece_all.glsl`, `BC1_tables.inl`) so `git log -p shaders/betsy` shows where everything
   went. Only this commit needs a code review.
   `bc_compress.comp` and the PSNR check come after, as our own code.
4. `asset.cpp`: the usage enum and its merges, RGBA32 conversion, power of two assert, scratch images and buffer,
   upload path, `--texture-compression`, `--no-dds`.
5. Measurements, then this file's findings and the default.

## Experiments

None yet.

## Findings

- Betsy has no BC2 or BC7; BC3 and BC5 are BC1 and BC4 passes stitched on the host. `master` and `v1.1-beta` have the
  same encoding code.
- Betsy's dithered BC1 path reads pixel 2 of each row twice (`MatchColorsBlock`, `BC1_DITHER`).
- vk-renderer's bindless set has no `UPDATE_AFTER_BIND`: any pass that binds it during the load blocks `add_texture`
  until that command buffer is done.
