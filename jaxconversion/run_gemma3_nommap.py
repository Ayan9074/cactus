import argparse
import re
import struct
import time
import os
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


def weight_for_loc(weights_dir: Path, loc: str) -> np.ndarray:
    def norm_unshift(arr: np.ndarray) -> np.ndarray:
        # Cactus Gemma weight conversion stores norm scales as (raw_scale + 1).
        # StableHLO graphs add +1 in-graph, so feed raw scale here.
        return (arr.astype(np.float32) - np.float32(1.0)).astype(np.float16)

    # Non-param runtime input
    if loc == "tokens":
        raise RuntimeError("tokens is runtime input, not weight")

    # Global weights
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
    b = m.group(4)

    # Attention subparts
    if block == "attn":
        if a == "_key_norm" and b == "scale":
            return norm_unshift(read_cactus_fp16(weights_dir / f"layer_{layer}_attn_k_norm.weights"))
        if a == "_query_norm" and b == "scale":
            return norm_unshift(read_cactus_fp16(weights_dir / f"layer_{layer}_attn_q_norm.weights"))
        if a == "q_einsum" and b == "w":
            q = read_cactus_fp16(weights_dir / f"layer_{layer}_attn_q.weights")  # [1024,640]
            return q.reshape(4, 256, 640).transpose(0, 2, 1)  # [4,640,256]
        if a == "kv_einsum" and b == "w":
            k = read_cactus_fp16(weights_dir / f"layer_{layer}_attn_k.weights")  # [256,640]
            v = read_cactus_fp16(weights_dir / f"layer_{layer}_attn_v.weights")  # [256,640]
            k = k.T.reshape(1, 640, 256)
            v = v.T.reshape(1, 640, 256)
            return np.stack([k, v], axis=0)  # [2,1,640,256]
        if a == "attn_vec_einsum" and b == "w":
            o = read_cactus_fp16(weights_dir / f"layer_{layer}_attn_output.weights")  # [640,1024]
            return o.reshape(640, 4, 256).transpose(1, 2, 0)  # [4,256,640]

    # MLP subparts
    if block == "mlp":
        if a == "gating_einsum":
            gate = read_cactus_fp16(weights_dir / f"layer_{layer}_ffn_gate.weights")  # [2048,640]
            up = read_cactus_fp16(weights_dir / f"layer_{layer}_ffn_up.weights")      # [2048,640]
            return np.stack([gate, up], axis=0)  # [2,2048,640]
        if a == "linear":
            down = read_cactus_fp16(weights_dir / f"layer_{layer}_ffn_down.weights")  # [640,2048]
            return down.T  # [2048,640]

    # Norms can arrive as:
    #   params['layer_X']['post_attention_norm']['scale']  -> block=post_attention_norm, a=scale
    # or nested variants handled above.
    if block == "post_attention_norm" and a == "scale":
        return norm_unshift(read_cactus_fp16(weights_dir / f"layer_{layer}_post_attn_norm.weights"))
    if block == "post_ffw_norm" and a == "scale":
        return norm_unshift(read_cactus_fp16(weights_dir / f"layer_{layer}_post_ffn_norm.weights"))
    if block == "pre_attention_norm" and a == "scale":
        return norm_unshift(read_cactus_fp16(weights_dir / f"layer_{layer}_input_norm.weights"))
    if block == "pre_ffw_norm" and a == "scale":
        return norm_unshift(read_cactus_fp16(weights_dir / f"layer_{layer}_pre_ffn_norm.weights"))

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

def set_input_by_tensor_dtype(g, t, arr: np.ndarray) -> None:
    dt = int(getattr(t, "dtype", 1))
    if dt == 1:
        g.set_input(t, np.asarray(arr, dtype=np.float16), dtype=1)
    elif dt == 2:
        g.set_input(t, np.asarray(arr, dtype=np.float32), dtype=2)
    else:
        g.set_input(t, np.asarray(arr, dtype=np.int32), dtype=dt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mlir", default="gemma_stablehlo.mlir")
    ap.add_argument("--weights", default="../weights/gemma-3-270m")
    ap.add_argument("--tokenizer", default="../weights/gemma-3-270m/tokenizer.json")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--exact-math", type=int, default=1, help="1: disable heuristic numeric rewrites for exact StableHLO parity")
    ap.add_argument("--use-cactus-tokenizer", type=int, default=1, help="1: tokenize with native Cactus tokenizer (matches runtime); 0: use tokenizer.json directly")
    ap.add_argument("--force-bos", type=int, default=0, help="1: prepend BOS token id 2")
    ap.add_argument("--pad-side", choices=["right", "left"], default="right")
    args = ap.parse_args()
    if int(args.exact_math) == 1:
        os.environ["CACTUS_EXACT_MATH"] = "1"

    mlir_text = Path(args.mlir).read_text()
    main_args = parse_main_args_with_locs(mlir_text)

    # Parser-friendly variant: drop loc annotations/directives.
    mlir_for_parse = re.sub(r' loc\("[^"]*"\)', '', mlir_text)
    mlir_for_parse = re.sub(r' loc\(#loc[0-9]+\)', '', mlir_for_parse)
    mlir_for_parse = re.sub(r'^#loc.*$', '', mlir_for_parse, flags=re.M)
    ir = parse_mlir(mlir_for_parse)
    print(f"Parsed inputs: {len(ir.inputs)}")

    def resolver(g, ssa, shape, dtype):
        return g.input(list(shape), dtype=dtype)

    t0 = time.perf_counter()
    g, env = lower_to_cactus(
        ir,
        patterns=["default"],
        verbose=False,
        input_resolver=resolver,
        strict_math=False,
    )
    print(f"Build: {(time.perf_counter() - t0) * 1000:.2f}ms")

    weights_dir = Path(args.weights)

    # load all param args except runtime tokens
    for ssa, _type, loc in main_args:
        if ssa not in env:
            continue
        if loc == "tokens":
            continue
        arr = weight_for_loc(weights_dir, loc)
        t = env[ssa]
        expected = tuple(getattr(t, "shape", ()))
        if tuple(arr.shape) != expected:
            raise ValueError(f"Shape mismatch for {ssa} loc={loc}: got {arr.shape}, expected {expected}")
        set_input_by_tensor_dtype(g, t, arr)

    set_constants(g, env, ir)

    tok = Tokenizer.from_file(args.tokenizer)
    ids = None
    if int(args.use_cactus_tokenizer) == 1:
        try:
            import sys
            sys.path.insert(0, str((Path(__file__).resolve().parents[1] / "python" / "src")))
            import cactus as cactus_py
            m = cactus_py.cactus_init(str(weights_dir), None, False)
            try:
                ids = [int(x) for x in cactus_py.cactus_tokenize(m, args.prompt)]
            finally:
                cactus_py.cactus_destroy(m)
        except Exception:
            ids = None
    if ids is None:
        ids = tok.encode(args.prompt).ids
    if int(args.force_bos) == 1:
        ids = [2] + ids

    T = 32

    def build_tokens(seq):
        ctx = seq[-T:]
        if len(ctx) < T:
            pad = [0] * (T - len(ctx))
            if args.pad_side == "left":
                ctx = pad + ctx
            else:
                ctx = ctx + pad
        return np.array([ctx], dtype=np.int32), min(len(seq), T) - 1

    print("Generating...")
    for step in range(args.steps):
        t_in, last = build_tokens(ids)
        set_input_by_tensor_dtype(g, env["%arg200"], t_in)

        t0 = time.perf_counter()
        g.execute()
        ms = (time.perf_counter() - t0) * 1000

        out = env[ir.outputs[0]].numpy().astype(np.float32)
        if out.ndim == 3:
            logits = out[0, last, :262144]
        elif out.ndim == 2:
            logits = out[0, :262144]
        else:
            raise RuntimeError(f"Unexpected output rank={out.ndim} shape={out.shape}")
        next_id = int(np.argmax(logits))
        ids.append(next_id)

        top = np.argsort(-logits)[:10]
        print(f"[{step}] next={next_id} logit={float(logits[next_id]):.4f} time={ms:.2f}ms piece={tok.decode([next_id])!r}")
        print("top:", [(int(i), tok.decode([int(i)])) for i in top])
        print("text:", tok.decode(ids))


if __name__ == "__main__":
    main()
