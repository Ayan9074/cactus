# make_gemma_jax_layout_weights.py
import argparse
import struct
from pathlib import Path

import numpy as np


HEADER_SIZE = 84
ALIGNMENT = 32
CACT_MAGIC = 0x54434143

PREC_FP16 = 1


def align_offset(offset, alignment=ALIGNMENT):
    rem = offset % alignment
    return offset if rem == 0 else offset + (alignment - rem)


def read_cactus_fp16(path):
    path = Path(path)
    with open(path, "rb") as f:
        header = f.read(HEADER_SIZE)

        magic, flags, alignment, ndim = struct.unpack_from("<IIII", header, 0)
        assert magic == CACT_MAGIC, f"bad magic in {path}"

        offset = 16
        raw_shape = []
        for _ in range(4):
            dim, = struct.unpack_from("<Q", header, offset)
            offset += 8
            raw_shape.append(dim)

        shape = tuple(int(raw_shape[i]) for i in range(ndim))

        precision, = struct.unpack_from("<I", header, offset)
        offset += 4
        assert precision == PREC_FP16, f"{path} is not FP16"

        byte_size, = struct.unpack_from("<Q", header, offset)
        offset += 8

        scales_bytes, = struct.unpack_from("<Q", header, offset)
        offset += 8

        data_offset = align_offset(HEADER_SIZE, alignment)
        if scales_bytes:
            data_offset = align_offset(data_offset + scales_bytes, alignment)

        f.seek(data_offset)
        data = np.frombuffer(f.read(byte_size), dtype=np.float16).copy()

    return data.reshape(shape)


def write_cactus_fp16(path, arr):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    arr = np.asarray(arr, dtype=np.float16)
    shape = list(arr.shape)

    if len(shape) > 4:
        raise ValueError(f"Cactus tensor header supports up to 4D, got {shape}")

    ndim = len(shape)
    raw_shape = shape + [0] * (4 - ndim)

    flags = 0
    alignment = ALIGNMENT
    byte_size = arr.size * 2
    scales_bytes = 0
    group_size = 0
    num_groups = 0
    original_N = shape[0] if shape else 1

    header = bytearray(HEADER_SIZE)
    offset = 0

    struct.pack_into("<I", header, offset, CACT_MAGIC); offset += 4
    struct.pack_into("<I", header, offset, flags); offset += 4
    struct.pack_into("<I", header, offset, alignment); offset += 4
    struct.pack_into("<I", header, offset, ndim); offset += 4

    for dim in raw_shape:
        struct.pack_into("<Q", header, offset, int(dim)); offset += 8

    struct.pack_into("<I", header, offset, PREC_FP16); offset += 4
    struct.pack_into("<Q", header, offset, byte_size); offset += 8
    struct.pack_into("<Q", header, offset, scales_bytes); offset += 8
    struct.pack_into("<I", header, offset, group_size); offset += 4
    struct.pack_into("<I", header, offset, num_groups); offset += 4
    struct.pack_into("<Q", header, offset, int(original_N)); offset += 8

    data_offset = align_offset(HEADER_SIZE, alignment)
    padding = b"\x00" * (data_offset - HEADER_SIZE)

    with open(path, "wb") as f:
        f.write(header)
        f.write(padding)
        f.write(np.ascontiguousarray(arr).tobytes())

    print(f"Wrote {path} shape={arr.shape}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="weights/gemma-2-2b")
    ap.add_argument("--dst", default="weights/gemma-2-2b-jax-layout")
    ap.add_argument("--layers", type=int, default=26)
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # Embedding: [256000,2304] -> [256128,2304]
    # ------------------------------------------------------------

    emb = read_cactus_fp16(src / "token_embeddings.weights")
    if emb.shape[0] == 256000:
        padded = np.zeros((256128, emb.shape[1]), dtype=np.float16)
        padded[: emb.shape[0], :] = emb
        emb = padded
    elif emb.shape[0] != 256128:
        raise ValueError(f"Unexpected embedding shape: {emb.shape}")

    write_cactus_fp16(dst / "arg0_token_embeddings.weights", emb)

    # ------------------------------------------------------------
    # Layers
    # ------------------------------------------------------------

    for i in range(args.layers):
        base = 1 + i * 9

        # arg base+0: input norm [2304]
        write_cactus_fp16(
            dst / f"arg{base+0}_layer_{i}_input_norm.weights",
            read_cactus_fp16(src / f"layer_{i}_input_norm.weights"),
        )

        # arg base+1: q [8,256,2304]
        q = read_cactus_fp16(src / f"layer_{i}_attn_q.weights")  # [2048,2304]
        q = q.reshape(8, 256, 2304)
        write_cactus_fp16(dst / f"arg{base+1}_layer_{i}_attn_q.weights", q)

        # arg base+2: combined kv [2,4,2304,256]
        k = read_cactus_fp16(src / f"layer_{i}_attn_k.weights")  # [1024,2304]
        v = read_cactus_fp16(src / f"layer_{i}_attn_v.weights")  # [1024,2304]

        k = k.reshape(4, 256, 2304).transpose(0, 2, 1)  # [4,2304,256]
        v = v.reshape(4, 256, 2304).transpose(0, 2, 1)  # [4,2304,256]
        kv = np.stack([k, v], axis=0)                   # [2,4,2304,256]

        write_cactus_fp16(dst / f"arg{base+2}_layer_{i}_attn_kv.weights", kv)

        # arg base+3: attn output [8,2304,256]
        o = read_cactus_fp16(src / f"layer_{i}_attn_output.weights")  # [2304,2048]
        o = o.reshape(2304, 8, 256).transpose(1, 0, 2)                # [8,2304,256]
        write_cactus_fp16(dst / f"arg{base+3}_layer_{i}_attn_output.weights", o)

        # arg base+4: combined gate/up [2,2304,9216]
        gate = read_cactus_fp16(src / f"layer_{i}_ffn_gate.weights")  # [9216,2304]
        up = read_cactus_fp16(src / f"layer_{i}_ffn_up.weights")      # [9216,2304]

        gate = gate.T  # [2304,9216]
        up = up.T      # [2304,9216]
        gate_up = np.stack([gate, up], axis=0)  # [2,2304,9216]

        write_cactus_fp16(dst / f"arg{base+4}_layer_{i}_ffn_gate_up.weights", gate_up)

        # arg base+5: down [9216,2304]
        down = read_cactus_fp16(src / f"layer_{i}_ffn_down.weights")  # [2304,9216]
        down = down.T
        write_cactus_fp16(dst / f"arg{base+5}_layer_{i}_ffn_down.weights", down)

        # arg base+6/7/8: norms [2304]
        write_cactus_fp16(
            dst / f"arg{base+6}_layer_{i}_post_attn_norm.weights",
            read_cactus_fp16(src / f"layer_{i}_post_attn_norm.weights"),
        )

        write_cactus_fp16(
            dst / f"arg{base+7}_layer_{i}_pre_ffn_norm.weights",
            read_cactus_fp16(src / f"layer_{i}_pre_ffn_norm.weights"),
        )

        write_cactus_fp16(
            dst / f"arg{base+8}_layer_{i}_post_ffn_norm.weights",
            read_cactus_fp16(src / f"layer_{i}_post_ffn_norm.weights"),
        )

    # arg235: output norm
    write_cactus_fp16(
        dst / "arg235_output_norm.weights",
        read_cactus_fp16(src / "output_norm.weights"),
    )

    print("\nDone.")
    print(f"Derived JAX-layout weights in: {dst}")


if __name__ == "__main__":
    main()