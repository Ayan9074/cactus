import argparse
import glob
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


def read_cactus_fp16(path):
    with open(path, "rb") as f:
        header = f.read(HEADER_SIZE)
        magic, flags, alignment, ndim = struct.unpack_from("<IIII", header, 0)
        if magic != CACT_MAGIC:
            raise ValueError(f"bad magic in {path}")
        off = 16
        raw_shape = []
        for _ in range(4):
            dim, = struct.unpack_from("<Q", header, off)
            off += 8
            raw_shape.append(dim)
        shape = tuple(int(raw_shape[i]) for i in range(ndim))
        precision, = struct.unpack_from("<I", header, off)
        off += 4
        if precision != 1:
            raise ValueError(f"{path} is not FP16 precision={precision}")
        byte_size, = struct.unpack_from("<Q", header, off)
        off += 8
        scales_bytes, = struct.unpack_from("<Q", header, off)
        data_offset = align_offset(HEADER_SIZE, alignment)
        if scales_bytes:
            data_offset = align_offset(data_offset + scales_bytes, alignment)
        f.seek(data_offset)
        data = np.frombuffer(f.read(byte_size), dtype=np.float16).copy()
    return data.reshape(shape)


def find_arg_file(weight_dir, idx):
    exact = weight_dir / f"arg{idx}.weights"
    if exact.exists():
        return exact
    matches = sorted(glob.glob(str(weight_dir / f"arg{idx}_*.weights")))
    if len(matches) != 1:
        raise FileNotFoundError(f"arg{idx} file missing/ambiguous in {weight_dir}")
    return Path(matches[0])


def set_input_by_tensor_dtype(g, t, arr: np.ndarray) -> None:
    dt = int(getattr(t, "dtype", 1))
    if dt == 1:
        g.set_input(t, np.asarray(arr, dtype=np.float16), dtype=1)
    elif dt == 2:
        g.set_input(t, np.asarray(arr, dtype=np.float32), dtype=2)
    else:
        g.set_input(t, np.asarray(arr, dtype=np.int32), dtype=dt)


def load_named_gemma2_param(weight_dir: Path, idx: int) -> np.ndarray:
    def norm_unshift(x: np.ndarray) -> np.ndarray:
        # StableHLO applies (1 + scale) for Gemma RMSNorm.
        # HF checkpoints store the final multiplier, so provide (w - 1) here.
        return (x.astype(np.float32) - np.float32(1.0)).astype(np.float16)

    if idx == 0:
        return read_cactus_fp16(weight_dir / "token_embeddings.weights")
    if idx == 1:
        return norm_unshift(read_cactus_fp16(weight_dir / "output_norm.weights"))

    # Per-layer layout in this StableHLO:
    # [attn_output, attn_kv, attn_q, ffn_gate_up, ffn_down, input_norm, post_attn_norm, pre_ffn_norm, post_ffn_norm]
    j = idx - 2
    layer = j // 9
    slot = j % 9

    if slot == 0:
        o = read_cactus_fp16(weight_dir / f"layer_{layer}_attn_output.weights")  # [D, H*DH]
        d, hd = o.shape
        h = 8
        dh = hd // h
        return o.reshape(d, h, dh).transpose(1, 2, 0)  # [H, DH, D]
    if slot == 1:
        k = read_cactus_fp16(weight_dir / f"layer_{layer}_attn_k.weights")  # [KVH*DH, D]
        v = read_cactus_fp16(weight_dir / f"layer_{layer}_attn_v.weights")  # [KVH*DH, D]
        kvh = 4
        dh = k.shape[0] // kvh
        kk = k.reshape(kvh, dh, k.shape[1]).transpose(0, 2, 1)  # [KVH, D, DH]
        vv = v.reshape(kvh, dh, v.shape[1]).transpose(0, 2, 1)  # [KVH, D, DH]
        return np.stack([kk, vv], axis=0)  # [2, KVH, D, DH]
    if slot == 2:
        q = read_cactus_fp16(weight_dir / f"layer_{layer}_attn_q.weights")  # [H*DH, D]
        h = 8
        dh = q.shape[0] // h
        return q.reshape(h, dh, q.shape[1]).transpose(0, 2, 1)  # [H, D, DH]
    if slot == 3:
        gate = read_cactus_fp16(weight_dir / f"layer_{layer}_ffn_gate.weights")  # [FF, D]
        up = read_cactus_fp16(weight_dir / f"layer_{layer}_ffn_up.weights")      # [FF, D]
        return np.stack([gate.T, up.T], axis=0)  # [2, D, FF]
    if slot == 4:
        down = read_cactus_fp16(weight_dir / f"layer_{layer}_ffn_down.weights")  # [D, FF]
        return down.T  # [FF, D]
    if slot == 5:
        return norm_unshift(read_cactus_fp16(weight_dir / f"layer_{layer}_input_norm.weights"))
    if slot == 6:
        return norm_unshift(read_cactus_fp16(weight_dir / f"layer_{layer}_post_attn_norm.weights"))
    if slot == 7:
        return norm_unshift(read_cactus_fp16(weight_dir / f"layer_{layer}_pre_ffn_norm.weights"))
    if slot == 8:
        return norm_unshift(read_cactus_fp16(weight_dir / f"layer_{layer}_post_ffn_norm.weights"))

    raise RuntimeError(f"unreachable slot={slot} idx={idx}")


def _pad_first_dim(arr: np.ndarray, target0: int) -> np.ndarray:
    if arr.shape[0] >= target0:
        return arr
    pad_shape = (target0 - arr.shape[0],) + arr.shape[1:]
    pad = np.zeros(pad_shape, dtype=arr.dtype)
    return np.concatenate([arr, pad], axis=0)


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
    ap.add_argument("--mlir", default="gemma_real_prefill.stablehlo.mlir")
    ap.add_argument("--weights", default="../weights/gemma-2-2b-jax-layout")
    ap.add_argument("--tokenizer", default="../weights/gemma-2-2b/tokenizer.json")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--mask", choices=["causal", "ones", "zeros"], default="causal")
    ap.add_argument("--logit-index", choices=["last_real", "last_slot"], default="last_real")
    ap.add_argument("--use-bos", type=int, default=1)
    ap.add_argument("--weights-format", choices=["arg", "named"], default="named")
    ap.add_argument("--pos-base", type=int, default=11)
    args = ap.parse_args()

    ir = parse_mlir(Path(args.mlir).read_text())

    # create explicit inputs for all args
    def resolver(g, ssa, shape, dtype):
        return g.input(list(shape), dtype=dtype)

    t0 = time.perf_counter()
    g, env = lower_to_cactus(ir, patterns=["default"], verbose=False, input_resolver=resolver)
    print(f"Build: {(time.perf_counter()-t0)*1000:.2f}ms")

    # load param args 0..235 directly (non-mmap)
    wdir = Path(args.weights)
    for i in range(236):
        ssa = f"%arg{i}"
        if args.weights_format == "arg":
            arr = read_cactus_fp16(find_arg_file(wdir, i))
        else:
            arr = load_named_gemma2_param(wdir, i)
        expected = tuple(int(x) for x in getattr(env[ssa], "shape", ()))
        if expected and tuple(arr.shape) != expected:
            if len(arr.shape) == len(expected) and arr.shape[1:] == expected[1:] and arr.shape[0] < expected[0]:
                arr = _pad_first_dim(arr, expected[0])
            else:
                raise ValueError(f"shape mismatch {ssa}: got {arr.shape}, expected {expected}")
        set_input_by_tensor_dtype(g, env[ssa], arr)

    set_constants(g, env, ir)

    tok = Tokenizer.from_file(args.tokenizer)
    ids = tok.encode(args.prompt).ids
    if args.use_bos and (not ids or ids[0] != 2):
        ids = [2] + ids

    def window_inputs(tokens):
        T = 8
        total = len(tokens)
        start = max(0, total - T)
        ctx = tokens[start:total]
        valid = len(ctx)
        if valid < T:
            ctx = ctx + [0] * (T - valid)
        tokens_np = np.array([ctx], dtype=np.int32)
        pos_np = np.zeros((1, T), dtype=np.int32)
        pos_np[0, :valid] = np.arange(start + int(args.pos_base), start + int(args.pos_base) + valid, dtype=np.int32)
        if args.mask == "ones":
            mask = np.ones((1, T, T), dtype=np.int32)
        elif args.mask == "zeros":
            mask = np.zeros((1, T, T), dtype=np.int32)
        else:
            mask = np.zeros((1, T, T), dtype=np.int32)
            for r in range(T):
                for c in range(T):
                    mask[0, r, c] = 1 if c <= r else 0
        return tokens_np, pos_np, mask, valid - 1

    print("Generating...")
    for step in range(args.steps):
        tokens_np, pos_np, mask_np, last = window_inputs(ids)
        set_input_by_tensor_dtype(g, env["%arg236"], tokens_np)
        set_input_by_tensor_dtype(g, env["%arg237"], pos_np)
        set_input_by_tensor_dtype(g, env["%arg238"], mask_np)
        t0 = time.perf_counter()
        g.execute()
        ms = (time.perf_counter() - t0) * 1000
        out = env[ir.outputs[0]].numpy().astype(np.float32)
        pick = last if args.logit_index == "last_real" else 7
        logits = out[0, pick, :256000]
        next_id = int(np.argmax(logits))
        top = np.argsort(-logits)[:10]
        ids.append(next_id)
        print(f"[{step}] next={next_id} logit={logits[next_id]:.4f} time={ms:.2f}ms piece={tok.decode([next_id])!r}")
        print("top:", [(int(i), tok.decode([int(i)])) for i in top])
        print("text:", tok.decode(ids))


if __name__ == "__main__":
    main()
