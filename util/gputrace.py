"""Runs an Nsight Graphics GPU Trace of the benchmark headlessly and prints per debug label timings.

Usage (from the repository root):
    python util/gputrace.py --out <dir> [--frames 3] [--start-after-ms 20000] [--metrics]
    python util/gputrace.py --parse <dir> [--metrics]     # re-read an existing trace output dir

Timings and metrics are per debug label range (CommandBuffer::debug_label), exported by Nsight's --auto-export.
GPU clocks are locked to base during traces, so absolute times are higher than in normal runs.
Ranges that spike are listed with the draws and dispatches they got on top of our own work, which point at another
process' GPU work. Those frames are left out of the "clean med" column.
"""
import argparse
import csv
import os
import subprocess
import sys

DEFAULT_NGFX = r"C:\Program Files\NVIDIA Corporation\Nsight Graphics 2026.3.1\host\windows-desktop-nomad-x64\ngfx.exe"
DEFAULT_EXE = os.path.join("build", "Debug", "brutus-vulkan.exe")

# Unit throughputs (% of peak) worth looking at to find what limits a pass
METRICS = [
    ("sm__throughput.avg.pct_of_peak_sustained_elapsed", "SM"),
    ("pes__throughput.avg.pct_of_peak_sustained_elapsed", "PES"),
    ("raster__throughput.avg.pct_of_peak_sustained_elapsed", "raster"),
    ("zrop__throughput.avg.pct_of_peak_sustained_elapsed", "ZROP"),
    ("crop__throughput.avg.pct_of_peak_sustained_elapsed", "CROP"),
    ("lts__throughput.avg.pct_of_peak_sustained_elapsed", "L2"),
    ("dramc__throughput.avg.pct_of_peak_sustained_elapsed", "DRAM"),
    ("warps_active_shader_vtg_realtime.avg.pct_of_peak_sustained_elapsed", "VTG warps"),
    ("warps_active_shader_ps_realtime.avg.pct_of_peak_sustained_elapsed", "PS warps"),
    ("warps_active_shader_cs_realtime.avg.pct_of_peak_sustained_elapsed", "CS warps"),
    ("warps_inactive_sm_idle_realtime.avg.pct_of_peak_sustained_elapsed", "SM idle"),
]

# A range spikes when it takes this much longer than its median over the traced frames
SPIKE_RATIO = 1.5
SPIKE_MIN_MS = 0.05


def run_trace(args):
    os.makedirs(args.out, exist_ok=True)
    command = [
        args.ngfx,
        "--activity", "GPU Trace Profiler",
        "--exe", os.path.abspath(args.exe),
        "--args", args.args,
        "--dir", os.getcwd(),
        "--output-dir", os.path.abspath(args.out),
        "--start-after-ms", str(args.start_after_ms),
        "--limit-to-frames", str(args.frames),
        "--auto-export",
    ]
    with open(os.path.join(args.out, "ngfx_output.txt"), "w") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout)


def read_tsv(path):
    with open(path, newline="") as f:
        return list(csv.reader(f, delimiter="\t"))


def median(values):
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def label_paths(rows):
    # The events export indents nested labels by 8 spaces per level, the regimes export names them by full path
    paths = []
    stack = []
    for row in rows[1:]:
        name = row[0].lstrip(" ")
        stack = stack[: (len(row[0]) - len(name)) // 8] + [name]
        paths.append("/".join(stack))
    return paths


def read_work_counts(export_dir, frames):
    """Returns {label path: [(draws, dispatches) per frame]}, empty when the export doesn't have them."""
    path = os.path.join(export_dir, "GPUTRACE_REGIMES.xls")
    if not os.path.exists(path):
        return {}
    rows = read_tsv(path)
    header = rows[0]
    draws = next((i for i, name in enumerate(header) if name.endswith("fe__draw_count.sum")), None)
    dispatches = next((i for i, name in enumerate(header) if name.endswith("gr__dispatch_count.sum")), None)
    if draws is None or dispatches is None:
        return {}
    counts = {}
    for row in rows[1:]:
        counts.setdefault(row[0], []).append((int(float(row[draws] or 0)), int(float(row[dispatches] or 0))))
    return {name: values for name, values in counts.items() if len(values) == frames}


def find_spikes(paths, durations, counts):
    """Returns {frame: [(path, median, duration, extra draws, extra dispatches)]} for the innermost spiking ranges.

    Other processes' GPU work (e.g. desktop composition at each vblank) gets time sliced into our frame and adds a
    context switch, the range it lands in then shows draws or dispatches we didn't record. Our own work is the
    minimum count of a range over the frames, foreign work only adds to it.
    """
    spiking = {}
    for path, values in zip(paths, durations):
        range_median = median(values)
        for frame, value in enumerate(values):
            if value > range_median * SPIKE_RATIO and value - range_median > SPIKE_MIN_MS:
                spiking.setdefault(frame, {})[path] = (range_median, value)
    spikes = {}
    for frame, ranges in sorted(spiking.items()):
        for path, (range_median, value) in ranges.items():
            if any(other.startswith(path + "/") for other in ranges):
                continue
            extra_draws = extra_dispatches = None
            if path in counts:
                draws, dispatches = zip(*counts[path])
                extra_draws = draws[frame] - min(draws)
                extra_dispatches = dispatches[frame] - min(dispatches)
            spikes.setdefault(frame, []).append((path, range_median, value, extra_draws, extra_dispatches))
    return spikes


def print_events(export_dir):
    rows = read_tsv(os.path.join(export_dir, "D3DPERF_EVENTS.xls"))
    frame_rows = read_tsv(os.path.join(export_dir, "FRAME.xls"))
    frames = len(rows[0]) - 1
    paths = label_paths(rows)
    durations = [[float(v) for v in row[1:]] for row in rows[1:]]
    spikes = find_spikes(paths, durations, read_work_counts(export_dir, frames))
    clean = [i for i in range(frames) if i not in spikes]

    def clean_median(values):
        return f"{median([values[i] for i in clean]):12.3f}" if spikes and clean else ""

    print(f"{'label':40}"
          + "".join(f"{'frame ' + str(i) + ('!' if i in spikes else ''):>12}" for i in range(frames))
          + (f"{'clean med':>12}" if spikes and clean else ""))
    for row, values in zip(rows[1:], durations):
        print(f"{row[0].replace(chr(9), '  '):40}" + "".join(f"{v:12.3f}" for v in values) + clean_median(values))
    for row in frame_rows:
        values = [float(v) for v in row[1:]]
        print(f"{row[0]:40}" + "".join(f"{v:12.3f}" for v in values) + clean_median(values))

    if spikes:
        print("\nSpikes (frames marked ! are left out of the clean median):")
        for frame, ranges in spikes.items():
            for path, range_median, value, extra_draws, extra_dispatches in ranges:
                if extra_draws is None:
                    cause = "no draw counts in the export"
                elif extra_draws or extra_dispatches:
                    cause = f"+{extra_draws} draws +{extra_dispatches} dispatches: GPU work from another process"
                else:
                    cause = "no extra draws or dispatches"
                print(f"  frame {frame:2} {path:50} {range_median:7.3f} -> {value:7.3f} ms, {cause}")


def print_metrics(export_dir):
    rows = read_tsv(os.path.join(export_dir, "GPUTRACE_REGIMES.xls"))
    header = rows[0]
    columns = []
    for metric, label in METRICS:
        index = next((i for i, name in enumerate(header) if name.endswith(metric)), None)
        if index is not None:
            columns.append((label, index))
    # Regimes repeat once per traced frame, only show the first frame
    seen = set()
    print(f"\n{'label (first frame, % of peak)':40}" + "".join(f"{label:>10}" for label, _ in columns))
    for row in rows[1:]:
        name = row[0].strip()
        if name in seen:
            break
        seen.add(name)
        print(f"{name:40}" + "".join(f"{float(row[i] or 0):10.1f}" for _, i in columns))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", help="output directory for a new trace")
    parser.add_argument("--parse", help="existing trace output directory to read instead of tracing")
    parser.add_argument("--exe", default=DEFAULT_EXE)
    parser.add_argument("--args", default="--benchmark")
    parser.add_argument("--ngfx", default=DEFAULT_NGFX)
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument("--start-after-ms", type=int, default=20000)
    parser.add_argument("--timeout", type=int, default=500)
    parser.add_argument("--metrics", action="store_true", help="also print unit throughputs per label")
    args = parser.parse_args()

    if not args.out and not args.parse:
        parser.error("either --out or --parse is required")
    if args.out:
        run_trace(args)
    export_dir = os.path.join(args.parse or args.out, "BASE")
    if not os.path.exists(os.path.join(export_dir, "D3DPERF_EVENTS.xls")):
        sys.exit(f"no export found in {export_dir}, see ngfx_output.txt")
    print_events(export_dir)
    if args.metrics:
        print_metrics(export_dir)


if __name__ == "__main__":
    main()
