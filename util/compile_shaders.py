"""Compiles every shader with glslangValidator to catch errors without running the app.

Usage (from the repository root):
    python util/compile_shaders.py                      # just validate
    python util/compile_shaders.py --out <dir>          # also keep the SPIR-V
    python util/compile_shaders.py --compare <a> <b>    # compare two --out dirs, e.g. before/after a formatting change

Shaders are compiled for every combination of the defines the app uses (see SceneRenderer constructor).
Without debug info, whitespace-only changes produce byte-identical SPIR-V.
"""
import argparse
import filecmp
import glob
import itertools
import os
import subprocess
import sys
import tempfile

DEFAULT_GLSLANG = r"C:\Program Files\VulkanSDK\1.4.335.0\Bin\glslangValidator.exe"
STAGES = (".comp", ".vert", ".frag", ".mesh", ".task")
DEFINES = {
    "MESH_LOD_COUNT": ["8"],
    "TASK_GROUP_SIZE": ["32"],
    "DRAW_BUCKETS_COUNT": ["6"],
    "DRAW_BINS_PER_BUCKET": ["8"],
    "LATE": ["false", "true"],
    "TASK_COMMANDS": ["0", "1"],
    "USE_ALPHA": ["false", "true"],
}
# Shaders with their own defines instead of DEFINES
SHADER_DEFINES = {
    "bc_compress.comp": {"BC_FORMAT": ["1", "3", "5"], "SRGB": ["0", "1"]},
}


def compile_all(glslang, out_dir):
    failures = 0
    for path in sorted(p for p in glob.glob(os.path.join("shaders", "*")) if p.endswith(STAGES)):
        defines = SHADER_DEFINES.get(os.path.basename(path), DEFINES)
        variants = [dict(zip(defines, values)) for values in itertools.product(*defines.values())]
        for variant in variants:
            suffix = "_".join(f"{k}={v}" for k, v in variant.items())
            output = os.path.join(out_dir, f"{os.path.basename(path)}.{suffix}.spv")
            command = [glslang, "--target-env", "vulkan1.3", "-o", output, path] + [f"-D{k}={v}" for k, v in variant.items()]
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode != 0:
                failures += 1
                print(f"FAILED {path} ({suffix})\n{result.stdout}{result.stderr}")
                break
    return failures


def compare(dir_a, dir_b):
    names = sorted(set(os.listdir(dir_a)) | set(os.listdir(dir_b)))
    different = [n for n in names if not (os.path.exists(os.path.join(dir_a, n)) and os.path.exists(os.path.join(dir_b, n))
                                          and filecmp.cmp(os.path.join(dir_a, n), os.path.join(dir_b, n), shallow=False))]
    for name in different:
        print(f"differs: {name}")
    print(f"{len(names) - len(different)} identical, {len(different)} different")
    return len(different)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--glslang", default=DEFAULT_GLSLANG)
    parser.add_argument("--out", help="directory to keep the SPIR-V in")
    parser.add_argument("--compare", nargs=2, metavar=("DIR_A", "DIR_B"))
    args = parser.parse_args()

    if args.compare:
        sys.exit(1 if compare(*args.compare) else 0)
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        failures = compile_all(args.glslang, args.out)
    else:
        with tempfile.TemporaryDirectory() as out_dir:
            failures = compile_all(args.glslang, out_dir)
    print("all shaders compiled" if failures == 0 else f"{failures} shader(s) failed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
