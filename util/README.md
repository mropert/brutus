# Profiling and validation utilities

Scripts used to measure and check rendering changes. Run them from the repository root: the app expects it as
working directory, and so do the scripts.

The `--benchmark` startup argument freezes time and places the camera at a fixed spot, which makes frames
reproducible: two runs of the same build render **bit-identical** frames, so any pixel difference between builds is a
real change. Add `--use-vertex-shader` to render with the vertex path even when mesh shaders are available. The scripts
take the app's arguments (`gputrace.py --args`, `BRUTUS_ARGS` for `renderdoc_capture.py`, `screenshot_benchmark.ps1
-Arguments`), e.g. `--args "--benchmark --use-vertex-shader"`.

## Tools

| Tool | Location |
| --- | --- |
| RenderDoc | `C:\Program Files\RenderDoc\qrenderdoc.exe` (UI with embedded Python, used for scripting) |
| Nsight Graphics | `C:\Program Files\NVIDIA Corporation\Nsight Graphics 2026.3.1\host\windows-desktop-nomad-x64\ngfx.exe` |
| glslangValidator | `C:\Program Files\VulkanSDK\1.4.335.0\Bin\glslangValidator.exe` |
| clang-format | `C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\Llvm\x64\bin\clang-format.exe` |
| cdb (debugger) | `cdbX64.exe` from the WinDbg store app, only runs from PowerShell, use `-cf script.txt` for commands |

Nsight GPU Trace needs GPU performance counters to be allowed for all users (NVIDIA Control Panel, Developer,
Manage GPU Performance Counters).

## Scripts

### `gputrace.py`: GPU timings per pass

Runs an Nsight GPU Trace of the benchmark headlessly and prints the time of every debug label
(`CommandBuffer::debug_label`) for each traced frame, plus the whole frame.

```
python util/gputrace.py --out <dir> [--frames 3] [--metrics]
python util/gputrace.py --parse <dir> --metrics
```

`--metrics` adds unit throughputs per label (SM, primitive engine, raster, ZROP, L2, DRAM, warp occupancy), which
helps tell what limits a pass. Keep in mind:
- GPU clocks are locked to base during traces: compare traces with each other, not with in-app timings.
- Trace a few frames and look at the spread, single frames sometimes spike.
- Nsight prints `[ERROR: Validation]` lines about `vkCmdPipelineBarrier` with `HOST_WRITE`: these come from its own
  instrumentation, not from the app.

### `renderdoc_capture.py`: frame capture and per-pass counters

Captures a benchmark frame (or re-analyzes an existing `.rdc`) and writes the action tree with per-draw counters
(GPU duration, rasterized primitives, samples passed), counters summed per debug label path in `passes.txt`, the
texture list, and PNGs of screen sized color targets and of every depth target layer.

```
set BRUTUS_RD_OUT=<dir>
"C:\Program Files\RenderDoc\qrenderdoc.exe" --python util/renderdoc_capture.py
```

See the script header for the other environment variables (`BRUTUS_RD_CAPTURE` to re-analyze a capture). RenderDoc
times each draw in isolation, so use its durations to compare draws with each other, and `gputrace.py` for pass costs.
Primitive counts are exact and are the best way to check culling or LOD changes.

### `screenshot_benchmark.ps1`: final frame image

```
util\screenshot_benchmark.ps1 -Out <file.png>
```

Runs the benchmark, waits for it to settle and screenshots the primary display. Use this for before/after images of
the final frame: RenderDoc saves swapchain images with its "undefined contents" pattern. To get a before image,
stash the changes, build and screenshot, then restore and do the same.

### `image_diff.ps1`: exact image comparison

```
util\image_diff.ps1 -A before.png -B after.png -OutDir <dir> [-IgnoreLeftPixels 490] [-Crops "name,x,y,w,h,scale;..."]
```

Prints the share of changed pixels and writes an amplified difference image and optional side-by-side crops.
`-IgnoreLeftPixels 490` skips the ImGui panel.

### `compile_shaders.py`: shader validation

```
python util/compile_shaders.py [--out <dir>]
python util/compile_shaders.py --compare <dir_a> <dir_b>
```

Compiles all shaders for every define combination the app uses, without starting the app. Comparing the SPIR-V of
two builds proves a change (e.g. formatting) has no effect: without debug info, identical code gives identical bytes.

### `optick_report.py`: startup profile

```
python util/optick_report.py startup_debug.opt [--tree-depth 3] [--min-ms 1]
python util/optick_report.py before.opt --compare after.opt
```

Reads the Optick capture the app saves at every startup and prints the per-thread call tree and totals per event name,
or the per-name differences between two captures. Every run overwrites `startup_debug.opt` (including runs from the
scripts above), so copy it aside to keep a baseline. Runs under RenderDoc or Nsight have inflated startup times,
compare plain runs only, and preferably a few of them.
