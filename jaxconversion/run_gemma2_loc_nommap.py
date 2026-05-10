#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import struct
import time
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from parse import parse_mlir
from lower_to_cactus import lower_to_cactus, decode_stablehlo_const

HEADER_SIZE = 84
CACT_MAGIC = 0x54434143


def align_offset(offset, alignment):
    rem = offset % alignment
    return offset if rem == 0 else offset + (alignment - rem)


def read_cactus_fp16(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        header = f.read(HEADER_SIZE)
        magic, _flags, alignment, ndim = struct.unpack_from("<IIII", header, 0)
        if magic != CACT_MAGIC:
            raise ValueError(f"bad magic in {path}")
        off = 16
        raw = []
        for _ in range(4):
            d, = struct.unpack_from("<Q", header, off)
            raw.append(int(d))
            off += 8
        shape = tuple(raw[:ndim])
        precision, = struct.unpack_from("<I", header, off)
        off += 4
        if precision != 1:
            raise ValueError(f"expected FP16 precision in {path}, got {precision}")
        byte_size, = struct.unpack_from("<Q", header, off)
        off += 8
        scales_bytes, = struct.unpack_from("<Q", header, off)
        data_offset = align_offset(HEADER_SIZE, alignment)
        if scales_bytes:
            data_offset = align_offset(data_offset + scales_bytes, alignment)
        f.seek(data_offset)
        arr = np.frombuffer(f.read(byte_size), dtype=np.float16).copy()
    return arr.reshape(shape)


def parse_main_args_with_locs(mlir_text: str):
    m = re.search(r"func\.func public @main\((.*?)\) ->", mlir_text, re.S)
    if not m:
        raise RuntimeError("Could not find public @main signature")
    args_blob = m.group(1)

    parts = []
    cur = ""
    depth = 0
    for ch in args_blob:
        if ch == "," and depth == 0:
            parts.append(cur.strip())
            cur = ""
            continue
        cur += ch
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth -= 1
    if cur.strip():
        parts.append(cur.strip())

    out = []
    for p in parts:
        mm = re.match(r"(%arg\d+):\s*([^\s]+)\s*loc\(\"([^\"]+)\"\)", p)
        if not mm:
            mm = re.match(r"(%arg\d+):\s*([^\s]+)", p)
            if not mm:
                raise RuntimeError(f"Could not parse arg entry: {p}")
            out.append((mm.group(1), mm.group(2), None))
        else:
            out.append((mm.group(1), mm.group(2), mm.group(3)))
    return out


def set_input_by_tensor_dtype(g, t, arr: np.ndarray) -> None:
    dt = int(getattr(t, "dtype", 1))
    if dt == 1:
        g.set_input(t, np.asarray(arr, dtype=np.float16), dtype=1)
    elif dt == 2:
        g.set_input(t, np.asarray(arr, dtype=np.float32), dtype=2)
    else:
        g.set_input(t, np.asarray(arr, dtype=np.int32), dtype=dt)


def _pad_first_dim(arr: np.ndarray, target0: int) -> np.ndarray:
    if arr.shape[0] >= target0:
        return arr
    pad_shape = (target0 - arr.shape[0],) + arr.shape[1:]
    pad = np.zeros(pad_shape, dtype=arr.dtype)
    return np.concatenate([arr, pad], axis=0)


def weight_for_loc(weights_dir: Path, loc: str) -> np.ndarray:
    def norm_unshift(arr: np.ndarray) -> np.ndarray:
        return (arr.astype(np.float32) - np.float32(1.0)).astype(np.float16)

    if loc in ("tokens", "positions", "attention_mask", "attention_mask_public", "positions_public"):
        raise RuntimeError(f"{loc} is runtime input, not weight")

    if loc == "params['embedder']['input_embedding']":
        return read_cactus_fp16(weights_dir / "token_embeddings.weights")
    if loc == "params['final_norm']['scale']":
        return norm_unshift(read_cactus_fp16(weights_dir / "output_norm.weights"))

    m = re.match(r"params\['layer_(\d+)'\]\['([^']+)'\]\['([^']+)'\](?:\['([^']+)'\])?", loc)
    if not m:
        raise RuntimeError(f"Unhandled loc format: {loc}")
    layer = int(m.group(1))
    block = m.group(2)
    a = m.group(3)

    if block == "attn" and a == "attn_vec_einsum":
        o = read_cactus_fp16(weights_dir / f"layer_{layer}_attn_output.weights")  # [D,H*DH]
        d, hd = o.shape
        h = 8
        dh = hd // h
        return o.reshape(d, h, dh).transpose(1, 2, 0)  # [H,DH,D]

    if block == "attn" and a == "kv_einsum":
        k = read_cactus_fp16(weights_dir / f"layer_{layer}_attn_k.weights")  # [KVH*DH,D]
        v = read_cactus_fp16(weights_dir / f"layer_{layer}_attn_v.weights")
        kvh = 4
        dh = k.shape[0] // kvh
        kk = k.reshape(kvh, dh, k.shape[1]).transpose(0, 2, 1)  # [KVH,D,DH]
        vv = v.reshape(kvh, dh, v.shape[1]).transpose(0, 2, 1)
        return np.stack([kk, vv], axis=0)  # [2,KVH,D,DH]

    if block == "attn" and a == "q_einsum":
        q = read_cactus_fp16(weights_dir / f"layer_{layer}_attn_q.weights")  # [H*DH,D]
        h = 8
        dh = q.shape[0] // h
        return q.reshape(h, dh, q.shape[1]).transpose(0, 2, 1)  # [H,D,DH]

    if block == "mlp" and a == "gating_einsum":
        gate = read_cactus_fp16(weights_dir / f"layer_{layer}_ffn_gate.weights")
        up = read_cactus_fp16(weights_dir / f"layer_{layer}_ffn_up.weights")
        return np.stack([gate.T, up.T], axis=0)  # [2,D,FF]

    if block == "mlp" and a == "linear":
        down = read_cactus_fp16(weights_dir / f"layer_{layer}_ffn_down.weights")
        return down.T  # [FF,D]

    if block == "pre_attention_norm" and a == "scale":
        return norm_unshift(read_cactus_fp16(weights_dir / f"layer_{layer}_input_norm.weights"))
    if block == "post_attention_norm" and a == "scale":
        return norm_unshift(read_cactus_fp16(weights_dir / f"layer_{layer}_post_attn_norm.weights"))
    if block == "pre_ffw_norm" and a == "scale":
        return norm_unshift(read_cactus_fp16(weights_dir / f"layer_{layer}_pre_ffn_norm.weights"))
    if block == "post_ffw_norm" and a == "scale":
        return norm_unshift(read_cactus_fp16(weights_dir / f"layer_{layer}_post_ffn_norm.weights"))

    raise RuntimeError(f"Unhandled mapped loc: {loc}")


def set_constants(g, env, ir):
    for ssa, const in ir.constants.items():
        if ssa not in env:
            continue
        t = env[ssa]
        td = getattr(t, "dtype", 1)
        shape = list(const.shape) if const.shape else [1]
        v = float(decode_stablehlo_const(const.value, const.dtype))
        if td == 1:
            max_f16 = float(np.finfo(np.float16).max)
            if not np.isfinite(v) or v < -max_f16:
                v = -10000.0
            elif v > max_f16:
                v = max_f16
            g.set_input(t, np.full(shape, v, dtype=np.float16), dtype=1)
        elif td == 2:
            g.set_input(t, np.full(shape, v, dtype=np.float32), dtype=2)
        else:
            scalar_i = int(round(float(v)))
            g.set_input(t, np.full(shape, scalar_i, dtype=np.int32), dtype=td)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mlir", default="gemma_real_prefill.loc.stablehlo.mlir")
    ap.add_argument("--weights", default="../weights/gemma-2-2b-it")
    ap.add_argument("--tokenizer", default="../weights/gemma-2-2b-it/tokenizer.json")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--pad-side", choices=["right", "left"], default="right")
    ap.add_argument("--pos-base", type=int, default=11)
    args = ap.parse_args()

    mlir_text = Path(args.mlir).read_text()
    main_args = parse_main_args_with_locs(mlir_text)

    mlir_for_parse = re.sub(r' loc\("[^"]*"\)', '', mlir_text)
    mlir_for_parse = re.sub(r' loc\(#loc[0-9]+\)', '', mlir_for_parse)
    mlir_for_parse = re.sub(r'^#loc.*$', '', mlir_for_parse, flags=re.M)
    ir = parse_mlir(mlir_for_parse)

    def resolver(g, ssa, shape, dtype):
        return g.input(list(shape), dtype=dtype)

    t0 = time.perf_counter()
    g, env = lower_to_cactus(ir, patterns=["default"], verbose=False, input_resolver=resolver)
    print(f"Build: {(time.perf_counter() - t0) * 1000:.2f}ms")

    weights_dir = Path(args.weights)
    for ssa, _type, loc in main_args:
        if ssa not in env or loc in (None, "tokens", "positions", "attention_mask", "attention_mask_public", "positions_public"):
            continue
        arr = weight_for_loc(weights_dir, loc)
        expected = tuple(int(x) for x in getattr(env[ssa], "shape", ()))
        if expected and tuple(arr.shape) != expected:
            if len(arr.shape) == len(expected) and arr.shape[1:] == expected[1:] and arr.shape[0] < expected[0]:
                arr = _pad_first_dim(arr, expected[0])
            else:
                raise ValueError(f"shape mismatch {ssa} loc={loc}: got {arr.shape} expected {expected}")
        set_input_by_tensor_dtype(g, env[ssa], arr)

    set_constants(g, env, ir)

    tok = Tokenizer.from_file(args.tokenizer)
    ids = tok.encode(args.prompt).ids

    T = int(getattr(env["%arg236"], "shape", (1, 8))[1])

    def build_window(seq):
        total = len(seq)
        start = max(0, total - T)
        ctx = seq[start:total]
        valid = len(ctx)
        if valid < T:
            pad = [0] * (T - valid)
            if args.pad_side == "left":
                ctx = pad + ctx
            else:
                ctx = ctx + pad
        tokens_np = np.array([ctx], dtype=np.float32)
        pos_np = np.zeros((1, T), dtype=np.float32)
        pos_np[0, :valid] = np.arange(start + int(args.pos_base), start + int(args.pos_base) + valid, dtype=np.float32)
        mask_np = np.zeros((1, T, T), dtype=np.float32)
        for r in range(valid):
            for c in range(valid):
                if c <= r:
                    mask_np[0, r, c] = 1.0
        return tokens_np, mask_np, pos_np, valid - 1

    print("Generating...")
    for step in range(args.steps):
        t_in, m_in, p_in, last = build_window(ids)
        set_input_by_tensor_dtype(g, env["%arg236"], t_in)
        set_input_by_tensor_dtype(g, env["%arg237"], m_in)
        set_input_by_tensor_dtype(g, env["%arg238"], p_in)

        t0 = time.perf_counter()
        g.execute()
        ms = (time.perf_counter() - t0) * 1000

        out = env[ir.outputs[0]].numpy().astype(np.float32)
        logits = out[0, last, :]
        next_id = int(np.argmax(logits))
        ids.append(next_id)

        top = np.argsort(-logits)[:10]
        print(f"[{step}] next={next_id} logit={float(logits[next_id]):.4f} time={ms:.2f}ms piece={tok.decode([next_id])!r}")
        print("top:", [(int(i), tok.decode([int(i)])) for i in top])
        print("text:", tok.decode(ids))


if __name__ == "__main__":
    main()
