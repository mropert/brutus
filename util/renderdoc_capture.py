"""Captures a benchmark frame with RenderDoc (or re-analyzes an existing capture) and dumps what's useful to inspect it.

Must run inside RenderDoc's UI Python environment, from the repository root:
    set BRUTUS_RD_OUT=<output dir>
    "C:\\Program Files\\RenderDoc\\qrenderdoc.exe" --python util/renderdoc_capture.py

Environment variables:
    BRUTUS_RD_OUT      required, output directory
    BRUTUS_RD_CAPTURE  optional, existing .rdc to analyze instead of capturing a new frame
    BRUTUS_RD_WARMUP   optional, seconds to wait before capturing (default 20, startup loads assets and shaders)
    BRUTUS_EXE         optional, default build/Debug/brutus-vulkan.exe
    BRUTUS_ARGS        optional, default "--benchmark"

Outputs in BRUTUS_RD_OUT:
    log.txt       progress and errors (qrenderdoc has no console)
    *.rdc         the capture
    actions.txt   action tree with debug labels and per-draw counters (GPU duration, rasterized primitives, samples...)
    passes.txt    counters summed per debug label path, e.g. "Shadow Map/Cascade 0/Draw"
    textures.txt  all textures with their formats
    color_*.png   non-MSAA screen sized color targets (HDR values are clamped), at the end of the frame
    depth_*.png   non-MSAA depth targets, one image per layer, at the end of the frame
Swapchain images are not saved: RenderDoc fills them with its "undefined contents" pattern, use
util/screenshot_benchmark.ps1 for final frame images instead.
"""
import json
import os
import shlex
import time
import traceback
from collections import defaultdict

import renderdoc as rd

ROOT = os.getcwd()
OUT = os.environ["BRUTUS_RD_OUT"]
EXE = os.environ.get("BRUTUS_EXE", os.path.join(ROOT, "build", "Debug", "brutus-vulkan.exe"))
ARGS = os.environ.get("BRUTUS_ARGS", "--benchmark")
WARMUP = float(os.environ.get("BRUTUS_RD_WARMUP", "20"))
COUNTERS = [
    rd.GPUCounter.EventGPUDuration,
    rd.GPUCounter.RasterizedPrimitives,
    rd.GPUCounter.SamplesPassed,
    rd.GPUCounter.PSInvocations,
    # Also needed for RasterizedPrimitives to be counted on mesh shader draws
    rd.GPUCounter.TSInvocations,
    rd.GPUCounter.MSInvocations,
]

os.makedirs(OUT, exist_ok=True)
log = open(os.path.join(OUT, "log.txt"), "w")


def write_log(*values):
    log.write(" ".join(str(v) for v in values) + "\n")
    log.flush()


def capture():
    options = rd.CaptureOptions()
    options.apiValidation = False
    result = rd.ExecuteAndInject(EXE, ROOT, ARGS, [], os.path.join(OUT, "capture"), options, False)
    write_log("inject", result.result)
    target = rd.CreateTargetControl("", result.ident, "brutus", True)
    start = time.time()
    while time.time() - start < WARMUP:
        target.ReceiveMessage(None)
        time.sleep(0.1)
    target.TriggerCapture(1)
    path = None
    start = time.time()
    while path is None and time.time() - start < 120:
        message = target.ReceiveMessage(None)
        if message.type == rd.TargetControlMessageType.NewCapture:
            path = message.newCapture.path
        time.sleep(0.05)
    os.system("taskkill /F /IM %s >NUL 2>&1" % os.path.basename(EXE))
    target.Shutdown()
    write_log("capture", path)
    return path


def walk(actions, structured_file, path, out):
    for action in actions:
        name = action.GetName(structured_file)
        out.append((action, path, name))
        is_marker = bool(action.flags & rd.ActionFlags.PushMarker) and not action.flags & rd.ActionFlags.MultiAction
        walk(action.children, structured_file, path + [name] if is_marker else path, out)


def analyze(capture_path):
    capture_file = rd.OpenCaptureFile()
    capture_file.OpenFile(capture_path, "", None)
    status, controller = capture_file.OpenCapture(rd.ReplayOptions(), None)
    write_log("open", status)
    structured_file = controller.GetStructuredFile()

    actions = []
    walk(controller.GetRootActions(), structured_file, [], actions)

    available = controller.EnumerateCounters()
    counters = {}
    for result in controller.FetchCounters([c for c in COUNTERS if c in available]):
        description = controller.DescribeCounter(result.counter)
        value = result.value.d if description.resultType == rd.CompType.Float else result.value.u64
        counters.setdefault(result.eventId, {})[description.name] = value

    passes = defaultdict(lambda: defaultdict(float))
    with open(os.path.join(OUT, "actions.txt"), "w") as f:
        for action, path, name in actions:
            values = counters.get(action.eventId, {})
            f.write("%6d %s%s %s\n" % (action.eventId, "  " * len(path), name, json.dumps(values) if values else ""))
            # Multi draws (indirect count) are parents of their sub-commands, only count the leaves
            is_work = action.flags & (rd.ActionFlags.Drawcall | rd.ActionFlags.Dispatch | rd.ActionFlags.MeshDispatch)
            if is_work and not action.children:
                for depth in range(1, len(path) + 1):
                    totals = passes["/".join(path[:depth])]
                    totals["actions"] += 1
                    for counter, value in values.items():
                        totals[counter] += value
    with open(os.path.join(OUT, "passes.txt"), "w") as f:
        for name, totals in passes.items():
            values = ("%s=%d" % (k, v) if float(v).is_integer() else "%s=%.3f" % (k, v) for k, v in totals.items())
            f.write("%-40s %s\n" % (name, "  ".join(values)))

    textures = controller.GetTextures()
    with open(os.path.join(OUT, "textures.txt"), "w") as f:
        for t in textures:
            f.write("%s %dx%d layers=%d mips=%d samples=%d %s\n" % (t.resourceId, t.width, t.height, t.arraysize, t.mips, t.msSamp, t.format.Name()))

    controller.SetFrameEvent(actions[-1][0].eventId, True)
    # Render targets are the largest single mip, single layer color textures (material textures have mips)
    screen = max((t for t in textures if t.msSamp == 1 and t.mips == 1 and t.arraysize == 1 and not t.format.Name().startswith("D")),
                 key=lambda t: t.width * t.height)
    for t in textures:
        if t.msSamp != 1:
            continue
        name = str(t.resourceId).replace("ResourceId::", "")
        is_depth = t.format.Name().startswith("D")
        if is_depth:
            for layer in range(t.arraysize):
                save = rd.TextureSave()
                save.resourceId = t.resourceId
                save.destType = rd.FileType.PNG
                save.slice.sliceIndex = layer
                save.comp.blackPoint = 0.0
                save.comp.whitePoint = 1.0
                controller.SaveTexture(save, os.path.join(OUT, "depth_%s_layer%d.png" % (name, layer)))
        elif (t.width, t.height) == (screen.width, screen.height) and "FLOAT" in t.format.Name():
            save = rd.TextureSave()
            save.resourceId = t.resourceId
            save.destType = rd.FileType.PNG
            save.alpha = rd.AlphaMapping.Discard
            controller.SaveTexture(save, os.path.join(OUT, "color_%s.png" % name))

    controller.Shutdown()
    capture_file.Shutdown()


try:
    path = os.environ.get("BRUTUS_RD_CAPTURE") or capture()
    if path:
        analyze(path)
    write_log("done")
except Exception:
    write_log(traceback.format_exc())
log.close()
os._exit(0)
