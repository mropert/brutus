"""Reads Optick .opt captures (e.g. startup_debug.opt) without the Optick GUI.

Usage:
    python util/optick_report.py startup_debug.opt [--tree-depth N] [--min-ms X] [--json out.json]
    python util/optick_report.py before.opt --compare after.opt [--min-ms X]

Format reference: optick/src/optick_server.cpp (file header), optick_core.cpp (DumpBoard, ScopeData and
EventDescription serialization), optick_serialization.cpp (strings are u32 length + bytes, vectors u32 count + items).
"""
import argparse
import json
import re
import struct
import zlib
from collections import defaultdict

OPTICK_MAGIC = 0xB50FB50F
PACKET_BOARD = 0
PACKET_EVENT_FRAME = 1


class Reader:
    def __init__(self, data):
        self.data = data
        self.offset = 0

    def unpack(self, fmt):
        value = struct.unpack_from(fmt, self.data, self.offset)[0]
        self.offset += struct.calcsize(fmt)
        return value

    def string(self):
        length = self.unpack("<I")
        value = self.data[self.offset:self.offset + length].decode("utf-8", "replace")
        self.offset += length
        return value


def read_packets(path):
    raw = open(path, "rb").read()
    magic, _version, flags = struct.unpack_from("<IHH", raw, 0)
    if magic != OPTICK_MAGIC:
        raise ValueError(f"{path} is not an Optick capture")
    data = zlib.decompress(raw[8:]) if flags & 3 else raw[8:]
    offset = 0
    while offset + 12 <= len(data):
        _version, size, packet_type, _app = struct.unpack_from("<IIHH", data, offset)
        yield packet_type, data[offset + 12:offset + 12 + size]
        offset += 12 + size


def parse_board(payload):
    r = Reader(payload)
    board = {"number": r.unpack("<I"), "frequency": r.unpack("<q")}
    r.unpack("<Q")  # origin
    r.unpack("<I")  # precision
    r.unpack("<q"), r.unpack("<q")  # time slice
    board["threads"] = []
    for _ in range(r.unpack("<I")):
        thread = {"id": r.unpack("<Q"), "pid": r.unpack("<I"), "name": r.string()}
        r.unpack("<i"), r.unpack("<i"), r.unpack("<I")  # max depth, priority, mask
        board["threads"].append(thread)
    for _ in range(r.unpack("<I")):
        r.unpack("<Q")  # fibers
    r.unpack("<I")  # forced main thread index
    board["names"] = []
    for _ in range(r.unpack("<I")):
        board["names"].append(r.string())
        r.string()  # file
        r.unpack("<I"), r.unpack("<I"), r.unpack("<I"), r.unpack("<f"), r.unpack("<B")  # line, filter, color, budget, flags
    return board


def parse_event_frame(payload):
    r = Reader(payload)
    r.unpack("<I")  # board number
    thread = r.unpack("<i")
    r.unpack("<i")  # fiber
    r.unpack("<q"), r.unpack("<q"), r.unpack("<i")  # scope time and frame type

    def events():
        return [(r.unpack("<q"), r.unpack("<q"), r.unpack("<I")) for _ in range(r.unpack("<I"))]

    events()  # categories
    return thread, events()


def short_name(name):
    """Turns MSVC __FUNCSIG__ style names into Class::function."""
    match = re.search(r"([\w:~<>]+)\s*\(", name)
    return match.group(1).split(" ")[-1] if match and "(" in name else name


def load(path):
    board = None
    per_thread = defaultdict(list)
    for packet_type, payload in read_packets(path):
        if packet_type == PACKET_BOARD:
            board = parse_board(payload)
        elif packet_type == PACKET_EVENT_FRAME:
            thread, events = parse_event_frame(payload)
            per_thread[thread].extend(events)
    if board is None:
        raise ValueError(f"{path} has no description board")

    to_ms = 1000.0 / board["frequency"]
    threads = {}
    totals = defaultdict(lambda: {"ms": 0.0, "count": 0, "max_ms": 0.0})
    for index, events in sorted(per_thread.items()):
        name = board["threads"][index]["name"] if 0 <= index < len(board["threads"]) else f"thread {index}"
        nodes = []
        stack = []
        for start, finish, desc in sorted((e for e in events if 0 <= e[0] <= e[1]), key=lambda e: (e[0], -e[1])):
            while stack and start >= stack[-1]:
                stack.pop()
            event_name = short_name(board["names"][desc]) if desc < len(board["names"]) else f"#{desc}"
            duration = (finish - start) * to_ms
            nodes.append({"depth": len(stack), "name": event_name, "ms": duration})
            total = totals[event_name]
            total["ms"] += duration
            total["count"] += 1
            total["max_ms"] = max(total["max_ms"], duration)
            stack.append(finish)
        threads[f"{name} [{index}]"] = nodes
    return threads, totals


def print_report(threads, totals, tree_depth, min_ms):
    for thread, nodes in threads.items():
        shown = [n for n in nodes if n["depth"] < tree_depth and n["ms"] >= min_ms]
        if shown:
            print(f"== {thread}")
            for node in shown:
                print(f"{'  ' * node['depth']}{node['name']}: {node['ms']:.2f} ms")
    print("== totals by name: ms, count, max ms")
    for name, total in sorted(totals.items(), key=lambda kv: -kv[1]["ms"]):
        if total["ms"] >= min_ms:
            print(f"{total['ms']:10.2f} {total['count']:6d} {total['max_ms']:9.2f}  {name}")


def print_compare(path_a, totals_a, path_b, totals_b, min_ms):
    print(f"A = {path_a}\nB = {path_b}")
    print(f"{'A ms':>10} {'B ms':>10} {'delta':>10} {'delta %':>8}  name")
    names = set(totals_a) | set(totals_b)
    rows = []
    for name in names:
        a = totals_a[name]["ms"] if name in totals_a else 0.0
        b = totals_b[name]["ms"] if name in totals_b else 0.0
        if max(a, b) >= min_ms:
            rows.append((b - a, a, b, name))
    for delta, a, b, name in sorted(rows, key=lambda row: -abs(row[0])):
        percent = f"{100.0 * delta / a:+7.1f}%" if a > 0 else "     new"
        print(f"{a:10.2f} {b:10.2f} {delta:+10.2f} {percent}  {name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("capture")
    parser.add_argument("--compare", metavar="CAPTURE_B", help="compare per-name totals against another capture")
    parser.add_argument("--tree-depth", type=int, default=3)
    parser.add_argument("--min-ms", type=float, default=1.0)
    parser.add_argument("--json", help="write the tree and totals as JSON")
    args = parser.parse_args()

    threads, totals = load(args.capture)
    if args.compare:
        _, totals_b = load(args.compare)
        print_compare(args.capture, totals, args.compare, totals_b, args.min_ms)
    else:
        print_report(threads, totals, args.tree_depth, args.min_ms)
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"threads": threads, "totals": totals}, f, indent=1)


if __name__ == "__main__":
    main()
