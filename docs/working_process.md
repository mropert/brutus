# Working process

Rules and workflow shared by every feature stream. Each feature has its own notes (see [README.md](README.md)); this
file covers how we work, measure and verify, whatever the feature. Measurement scripts are documented in
[util/README.md](../util/README.md), the benchmark scene in [benchmark.md](benchmark.md).

## Measurement conditions

- **Hardware**: RTX 2080 SUPER (8 GB, 214 MB BAR, no resizable BAR).
- **Scene**: `--benchmark` (10k characters, ~40k draws, fixed camera and time, frames bit-identical between runs of the
  same build), mesh shader path unless stated.
- **GPU timings**: Nsight GPU Trace at locked base clocks (`util/gputrace.py`). Before 2026-10-27 traces ran the Debug
  build, since then mostly Release: GPU times match at locked clocks.
- **In-app GPU timings are only clock-stable in Debug**: the stable power library refuses Release builds, so Release UI
  timings follow boost clocks (~0.5 ms off for the same work). Compare with GPU Trace, Debug, or Release built with
  `GPU_STABLE_POWER_ENABLED=1`. Never compare UI timings across builds.

## Conventions

Also in the assistant memory:

- clang-format every changed C++ and GLSL file with `--style=file`.
- Comments only for what names and types don't say (contracts, special values, unusual behavior).
- `size_t` for index loops, cast at the use sites; `auto` / `const auto` for pointers, never `auto*`; `begin( a )` /
  `end( a )` rather than `a.begin()`.
- Standard headers first included after vk-renderer's `import vulkan` break on MSVC: prefer headers `common.h` already
  includes.
- Commits end with `Assisted-By: Claude Opus 5.5 <noreply@anthropic.com>` (`Co-Authored-By` only for fully generated
  tooling); commit vk-renderer first then bump the submodule; the user reviews every diff before a commit; nothing is
  pushed by the assistant.
- `core.autocrlf = true`: working copies are CRLF, git stores LF.
- Every change reports, next to its image check and GPU timings, its memory delta (VRAM and significant CPU memory).

## Verification loop

From the repository root, for every change:

1. Build: `cmake --build build --config Debug --target brutus-vulkan` (check the exit code, not only the last output
   line) and `python util/compile_shaders.py` (shaders are compiled at runtime, a failed first compile throws and shows
   an `abort()` popup). Build Release too when tracing it.
2. Smoke run for validation output (nothing printed is clean). The messages go to stdout (vk-bootstrap's default
   messenger uses `printf`), which is buffered when redirected: a killed process loses them, so `timeout` runs prove
   nothing. Close the window instead so the app exits and flushes: `Start-Process` with `-RedirectStandardOutput`,
   wait, `CloseMainWindow()`, `WaitForExit()` (give Bistro in Debug ~150 s to load).
3. Correctness: `util/screenshot_benchmark.ps1` + `util/image_diff.ps1` against the previous build; refactors must be
   pixel identical (frames are deterministic). Check both paths: add `--use-vertex-shader` to the arguments for the
   vertex path (every script takes extra arguments). If a diff shows only a strip at the bottom, the taskbar was
   captured: retake. When the desktop is in use, screenshots can capture other windows: compare RenderDoc color dumps
   instead (`renderdoc_capture.py` saves the resolved color target).
4. Counters: `util/renderdoc_capture.py` (`passes.txt` has per-pass rasterized primitives, fragment/task/mesh
   invocations; `actions.txt` shows the indirect draws with their group counts). Reading buffers from a capture (e.g.
   indirect commands) works from a qrenderdoc Python script using `BufferDescription.gpuAddress` and push constants.
5. Cost: `python util/gputrace.py --exe <exe> --out <dir> --metrics`, compare against a previous trace dir. The script
   traces `build/Debug` unless `--exe` is given: pass the build under test. Use 10 frames (two runs) when a
   difference is small: the main pass varies +-0.3-0.5 ms frame to frame. Compare per-range timings (per pass, per
   bin) rather than whole passes, using the `clean med` column that leaves out frames hit by other processes' GPU
   work.
6. Prefer command line options to source edits for comparisons (`brutus-vulkan.exe --help` lists them, e.g.
   `--msaa`): they need no rebuild and can't be left behind. When a test or benchmark keeps needing the same edit,
   suggest an option for it. Temporary experiments are marked `// TEMP EXPERIMENT, revert`, checked with
   `grep -rn "TEMP EXPERIMENT" src shaders`. Back up the file first, restore from the backup and **touch it**
   (`(Get-Item f).LastWriteTime = Get-Date`): `Copy-Item` keeps the old timestamp and MSBuild then skips recompiling.
   Never `git checkout` a file with uncommitted work. Confirm runtime settings (mesh shading...) in the UI before
   trusting measurements.
7. Editing: prefer the Edit tool; Git Bash `sed -i` converts CRLF files to LF (restore with PowerShell `ReadAllText`,
   `-replace "\r?\n", "\r\n"`, `WriteAllText`). PowerShell `Set-Content -Encoding utf8` writes a BOM that the runtime
   shader compiler rejects.
8. Reproducing a manual view: the user pastes the UI's "Copy camera" snippet, swapped into the `is_benchmark` setup as
   a temporary experiment.
9. Debugging crashes/`abort()` popups: `cdbX64.exe` (WinDbg store app) from PowerShell with a command file, e.g.
   `sxe -c "kn 30;gn" eh` and `bu ucrtbased!abort "kn 40;q"`, `-y` pointing at `build/Debug`. A throw from
   `waitForFences` means device lost: look for out-of-bounds buffer reads.

Temporary outputs (traces, captures, screenshots, backups, commit message drafts) go to the git-ignored `scratch/`
folder at the repository root. Screenshots only go to `docs/` when a doc references them and they are hard to
reproduce.

## Tools and their pitfalls

- **Nsight GPU Trace** debug label ranges are the unit of measurement: put a `cmd.debug_label()` around anything worth
  timing. Labels around individual cascades didn't serialize them measurably.
- **Startup profile**: every run overwrites `startup_debug.opt` / `startup_release.opt` (Optick), copy it aside for a
  baseline, compare with `util/optick_report.py`. Runs under RenderDoc or Nsight have inflated startup times.
- **RenderDoc's Quad Overdraw overlay** only runs on single-sample targets (`--msaa 1`), and replays each draw first
  with depth writes off then for real (`VulkanQuadOverdrawCallback` in `vk_overlay.cpp`): it shows overdraw and order
  between draws, never order within a draw. Use fragment shader invocations or timings for that. Its values aren't a
  plain quad count (not multiples of 4).
- RenderDoc per-draw counter values shuffle between replays (culling atomics change draw order), only per-pass totals
  are stable. RenderDoc replays counters for a few minutes on large captures, it isn't stuck.
- Bistro (`--scene ../niagara_bistro/bistro.gltf`) takes much longer to load (texture decompression): give
  screenshots 45-60 s, and prefer Release.
- `image_diff.ps1` rounds its percentages: a few hundred changed pixels print as 0.00%. Check the mean difference or
  count pixels exactly before calling a change pixel identical.
- PowerShell variable names ignore case: a script with a `$A` parameter and an `$a` local silently breaks.
- Back to back app launches occasionally fail with `0xC0000142`: wait a few seconds between runs and retry.

## GPU and Vulkan findings

- NVIDIA inserts a wait-for-idle when a queue switches between compute and graphics work ("subchannel switch" in the
  Nsight docs, documented for Ampere/Ada, presumably Turing): batch dispatches together and draws together.
- `vkCmdWriteTimestamp` defines an execution dependency on everything recorded before it: keep timing scopes around
  whole sections.
- `EmitMeshTasksEXT` must be executed by every task shader invocation, exactly once, in uniform control flow
  (SPV_EXT_mesh_shader); arguments come from the first invocation. `spirv-val` and the validation layers don't catch a
  lane 0 only emit or an early return.
- `groupMemoryBarrier()` only orders memory, it doesn't wait for the other invocations: use `barrier()`.
- GLSL has no infinity literal: `math.glsl` defines `INFINITY` as `1.0 / 0.0` (guaranteed since GLSL 4.40, folded by
  glslang). Integer division and modulo by non powers of two are emulated on GPUs: keep such divisors powers of two.
- `#if` on a define set to `true`/`false` always evaluates to 0: preprocessor-tested defines must be 0/1
  (`TASK_COMMANDS`).
- Host-visible buffers land in BAR (214 MB here) only while they fit; VMA falls back to system memory silently.
- std430 vs C++: `glm::aligned_vec3` is 16 bytes, so never put a scalar right after one in a C++ mirror (GLSL packs it
  into the vec3's last 4 bytes).
