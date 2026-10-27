"""Runs an Nsight Graphics GPU Trace of the benchmark headlessly and prints per debug label timings.

Usage (from the repository root):
    python util/gputrace.py --out <dir> [--frames 3] [--start-after-ms 20000] [--metrics]
    python util/gputrace.py --parse <dir> [--metrics]     # re-read an existing trace output dir

Timings and metrics are per debug label range (CommandBuffer::debug_label), exported by Nsight's --auto-export.
GPU clocks are locked to base during traces, so absolute times are higher than in normal runs.
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


def print_events(export_dir):
    rows = read_tsv(os.path.join(export_dir, "D3DPERF_EVENTS.xls"))
    frames = len(rows[0]) - 1
    print(f"{'label':40}" + "".join(f"{'frame ' + str(i):>12}" for i in range(frames)))
    for row in rows[1:]:
        print(f"{row[0].replace(chr(9), '  '):40}" + "".join(f"{float(v):12.3f}" for v in row[1:]))
    frame_rows = read_tsv(os.path.join(export_dir, "FRAME.xls"))
    for row in frame_rows:
        print(f"{row[0]:40}" + "".join(f"{float(v):12.3f}" for v in row[1:]))


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
