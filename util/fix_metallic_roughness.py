"""Moves roughness from the blue to the green channel of a glb's metallic-roughness textures and clears metalness.

Some Sketchfab exports (assets/sabrina.glb) store roughness in blue, which glTF reads as metalness, with green (glTF
roughness) constant at 1: renderers that follow the spec draw them fully metallic and fully rough.

Usage (from the repository root):
    python util/fix_metallic_roughness.py assets/sabrina.glb

The original is copied to <name>.orig.glb first (an existing copy is kept). Textures whose green channel isn't
constant 1 are left alone, so running the script on a fixed file changes nothing. Only 8-bit PNG images are handled;
no dependencies besides the standard library.
"""
import argparse
import json
import os
import shutil
import struct
import sys
import zlib

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
GLB_MAGIC = 0x46546C67
CHUNK_JSON = 0x4E4F534A
CHUNK_BIN = 0x004E4942


def png_chunks(data):
    position = len(PNG_SIGNATURE)
    while position < len(data):
        length, kind = struct.unpack(">I4s", data[position:position + 8])
        yield kind, data[position + 8:position + 8 + length]
        position += 12 + length


def unfilter(raw, width, height, pixel_size):
    stride = width * pixel_size
    pixels = bytearray(height * stride)
    previous = bytearray(stride)
    position = 0
    for y in range(height):
        kind = raw[position]
        line = bytearray(raw[position + 1:position + 1 + stride])
        position += 1 + stride
        if kind == 1:
            for i in range(pixel_size, stride):
                line[i] = (line[i] + line[i - pixel_size]) & 0xFF
        elif kind == 2:
            for i in range(stride):
                line[i] = (line[i] + previous[i]) & 0xFF
        elif kind == 3:
            for i in range(stride):
                left = line[i - pixel_size] if i >= pixel_size else 0
                line[i] = (line[i] + (left + previous[i]) // 2) & 0xFF
        elif kind == 4:
            for i in range(stride):
                a = line[i - pixel_size] if i >= pixel_size else 0
                b = previous[i]
                c = previous[i - pixel_size] if i >= pixel_size else 0
                pa, pb, pc = abs(b - c), abs(a - c), abs(a + b - 2 * c)
                line[i] = (line[i] + (a if pa <= pb and pa <= pc else b if pb <= pc else c)) & 0xFF
        pixels[y * stride:(y + 1) * stride] = line
        previous = line
    return pixels


def decode_png_rgb(data):
    """Returns width, height and the RGB bytes (alpha dropped: metallic-roughness textures don't use it)."""
    if data[:len(PNG_SIGNATURE)] != PNG_SIGNATURE:
        raise ValueError("not a PNG")
    header, compressed, palette = None, b"", None
    for kind, chunk in png_chunks(data):
        if kind == b"IHDR":
            header = struct.unpack(">IIBBBBB", chunk)
        elif kind == b"IDAT":
            compressed += chunk
        elif kind == b"PLTE":
            palette = chunk
    width, height, depth, color_type, _, _, interlace = header
    if depth != 8 or interlace != 0 or color_type not in (2, 3, 6):
        raise ValueError(f"unsupported PNG (depth {depth}, color type {color_type}, interlace {interlace})")
    pixel_size = {2: 3, 3: 1, 6: 4}[color_type]
    pixels = unfilter(zlib.decompress(compressed), width, height, pixel_size)
    if color_type == 3:
        return width, height, b"".join(palette[i * 3:i * 3 + 3] for i in pixels)
    if color_type == 6:
        rgb = bytearray(width * height * 3)
        for channel in range(3):
            rgb[channel::3] = pixels[channel::4]
        return width, height, rgb
    return width, height, pixels


def encode_png_rgb(width, height, rgb):
    stride = width * 3
    raw = b"".join(b"\x00" + rgb[y * stride:(y + 1) * stride] for y in range(height))

    def chunk(kind, payload):
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload))

    return (PNG_SIGNATURE + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def read_glb(path):
    with open(path, "rb") as file:
        data = file.read()
    magic, version, _ = struct.unpack("<III", data[:12])
    if magic != GLB_MAGIC or version != 2:
        raise ValueError(f"{path} is not a glTF 2.0 binary")
    json_length, json_type = struct.unpack("<II", data[12:20])
    bin_start = 20 + json_length
    bin_length, bin_type = struct.unpack("<II", data[bin_start:bin_start + 8])
    if json_type != CHUNK_JSON or bin_type != CHUNK_BIN:
        raise ValueError(f"{path}: unexpected chunk layout")
    return json.loads(data[20:bin_start]), data[bin_start + 8:bin_start + 8 + bin_length]


def write_glb(path, gltf, views):
    """Writes the buffer views back to back (4-byte aligned) as the single binary buffer."""
    binary = bytearray()
    for view, data in zip(gltf["bufferViews"], views):
        binary += b"\x00" * (-len(binary) % 4)
        view["byteOffset"] = len(binary)
        view["byteLength"] = len(data)
        binary += data
    binary += b"\x00" * (-len(binary) % 4)
    gltf["buffers"][0]["byteLength"] = len(binary)

    text = json.dumps(gltf, separators=(",", ":")).encode()
    text += b" " * (-len(text) % 4)
    with open(path, "wb") as file:
        file.write(struct.pack("<III", GLB_MAGIC, 2, 12 + 8 + len(text) + 8 + len(binary)))
        file.write(struct.pack("<II", len(text), CHUNK_JSON) + text)
        file.write(struct.pack("<II", len(binary), CHUNK_BIN) + binary)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("glb")
    args = parser.parse_args()

    gltf, binary = read_glb(args.glb)
    if len(gltf["buffers"]) != 1 or "uri" in gltf["buffers"][0]:
        sys.exit(f"{args.glb}: expected a single embedded buffer")
    views = [binary[view.get("byteOffset", 0):view.get("byteOffset", 0) + view["byteLength"]]
             for view in gltf["bufferViews"]]

    images = sorted({gltf["textures"][material["pbrMetallicRoughness"]["metallicRoughnessTexture"]["index"]]["source"]
                     for material in gltf.get("materials", [])
                     if "metallicRoughnessTexture" in material.get("pbrMetallicRoughness", {})})
    fixed = 0
    for image in images:
        view = gltf["images"][image]["bufferView"]
        width, height, rgb = decode_png_rgb(views[view])
        green = rgb[1::3]
        if green.count(0xFF) != len(green):
            print(f"image {image}: green isn't constant 1, left alone")
            continue
        rgb = bytearray(rgb)
        rgb[1::3] = rgb[2::3]
        rgb[2::3] = bytes(width * height)
        views[view] = encode_png_rgb(width, height, rgb)
        gltf["images"][image]["mimeType"] = "image/png"
        fixed += 1
        print(f"image {image}: roughness moved to green, metalness cleared")

    if fixed == 0:
        print("nothing to fix")
        return
    root, extension = os.path.splitext(args.glb)
    backup = f"{root}.orig{extension}"
    if not os.path.exists(backup):
        shutil.copy2(args.glb, backup)
        print(f"original copied to {backup}")
    write_glb(args.glb, gltf, views)
    print(f"wrote {args.glb}")


if __name__ == "__main__":
    main()
