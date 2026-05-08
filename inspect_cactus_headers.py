# inspect_cactus_headers.py
import argparse
import struct
from pathlib import Path

PREC = {
    0: "INT8",
    1: "FP16",
    2: "FP32",
    3: "INT4",
}

FLAG_HAS_SCALES = 1 << 0
FLAG_PAGE_ALIGNED = 1 << 1
FLAG_TRANSPOSED = 1 << 2
FLAG_INTERLEAVED = 1 << 3

HEADER_SIZE = 84


def align_offset(offset, alignment):
    rem = offset % alignment
    return offset if rem == 0 else offset + (alignment - rem)


def read_header(path):
    path = Path(path)

    with open(path, "rb") as f:
        data = f.read(256)

    if len(data) < HEADER_SIZE:
        return {
            "file": str(path),
            "error": f"too small for header: {len(data)} bytes",
        }

    offset = 0

    magic, = struct.unpack_from("<I", data, offset)
    offset += 4

    # 'CACT' little-endian as uint32.
    if magic != 0x54434143:
        return {
            "file": str(path),
            "error": "bad magic",
            "magic_hex": hex(magic),
            "first_16_bytes": data[:16].hex(),
        }

    flags, = struct.unpack_from("<I", data, offset)
    offset += 4

    alignment, = struct.unpack_from("<I", data, offset)
    offset += 4
    if alignment == 0:
        alignment = 1

    ndim, = struct.unpack_from("<I", data, offset)
    offset += 4

    raw_shape = []
    for _ in range(4):
        dim, = struct.unpack_from("<Q", data, offset)
        offset += 8
        raw_shape.append(dim)

    shape = [int(raw_shape[i]) for i in range(min(ndim, 4)) if raw_shape[i] > 0]

    precision, = struct.unpack_from("<I", data, offset)
    offset += 4

    byte_size, = struct.unpack_from("<Q", data, offset)
    offset += 8

    scales_bytes, = struct.unpack_from("<Q", data, offset)
    offset += 8

    group_size, = struct.unpack_from("<I", data, offset)
    offset += 4

    num_groups, = struct.unpack_from("<I", data, offset)
    offset += 4

    original_N, = struct.unpack_from("<Q", data, offset)
    offset += 8

    aligned_header = align_offset(HEADER_SIZE, alignment)
    if scales_bytes > 0:
        scales_offset = aligned_header
        data_offset = align_offset(scales_offset + scales_bytes, alignment)
    else:
        scales_offset = 0
        data_offset = aligned_header

    return {
        "file": str(path),
        "flags": flags,
        "has_scales": bool(flags & FLAG_HAS_SCALES),
        "page_aligned": bool(flags & FLAG_PAGE_ALIGNED),
        "transposed": bool(flags & FLAG_TRANSPOSED),
        "interleaved": bool(flags & FLAG_INTERLEAVED),
        "alignment": alignment,
        "ndim": ndim,
        "raw_shape_slots": raw_shape,
        "shape": shape,
        "precision": PREC.get(precision, f"UNKNOWN({precision})"),
        "byte_size": byte_size,
        "scales_bytes": scales_bytes,
        "group_size": group_size,
        "num_groups": num_groups,
        "original_N": original_N,
        "data_offset": data_offset,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    args = ap.parse_args()

    root = Path(args.root)

    files = [
        root / "arg0_token_embeddings.weights",
        root / "arg2_layer_0_attn_q.weights",
        root / "arg3_layer_0_attn_kv.weights",
        root / "arg4_layer_0_attn_output.weights",
        root / "arg5_layer_0_ffn_gate_up.weights",
        root / "arg6_layer_0_ffn_down.weights",
        root / "arg235_output_norm.weights",
    ]

    for p in files:
        print(read_header(p))


if __name__ == "__main__":
    main()