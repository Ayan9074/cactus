from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from lower_to_cactus import lower_to_cactus
from parse import parse_mlir
from run_gemma3_nommap import set_constants


def decode_npy_arr(a: np.ndarray) -> np.ndarray:
    # JAX bfloat16 in .npy often appears as raw 2-byte void dtype (|V2).
    if a.dtype.kind == "V" and a.dtype.itemsize == 2:
        u16 = a.view(np.uint16)
        u32 = (u16.astype(np.uint32) << 16)
        return u32.view(np.float32)
    return a


def set_input_by_tensor_dtype(g, t, arr: np.ndarray) -> None:
    dt = int(getattr(t, "dtype", 1))
    if dt == 1:
        g.set_input(t, np.asarray(arr, dtype=np.float16), dtype=1)
    elif dt == 2:
        g.set_input(t, np.asarray(arr, dtype=np.float32), dtype=2)
    else:
        g.set_input(t, np.asarray(arr, dtype=np.float32), dtype=dt)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32).reshape(-1)
    b = b.astype(np.float32).reshape(-1)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict-math", action="store_true")
    args = ap.parse_args()

    bundle = Path("gemma3_bundle")
    mlir_text = (bundle / "gemma_stablehlo.mlir").read_text()
    ir = parse_mlir(mlir_text)

    g, env = lower_to_cactus(
        ir,
        patterns=["default"],
        verbose=False,
        input_resolver=lambda g, ssa, shape, dtype: g.input(list(shape), dtype=dtype),
        strict_math=bool(args.strict_math),
    )

    # Feed arg0..arg199 from bundle (arg200 is runtime tokens).
    for i in range(200):
        ssa = f"%arg{i}"
        if ssa not in env:
            continue
        a = decode_npy_arr(np.load(bundle / f"arg{i}.npy"))
        set_input_by_tensor_dtype(g, env[ssa], a)

    tokens = np.load(bundle / "tokens.npy")
    set_input_by_tensor_dtype(g, env["%arg200"], decode_npy_arr(tokens))
    set_constants(g, env, ir)

    g.execute()
    out = env[ir.outputs[0]].numpy().astype(np.float32)
    ref = decode_npy_arr(np.load(bundle / "jax_logits.npy")).astype(np.float32)

    # Canonicalize shape to [B, V]
    if out.ndim == 3:
        out_last = out[:, -1, :]
    else:
        out_last = out
    if ref.ndim == 3:
        ref_last = ref[:, -1, :]
    else:
        ref_last = ref

    diff = np.abs(out_last - ref_last)
    print("Cactus shape:", out_last.shape, "Ref shape:", ref_last.shape)
    print("all_finite:", bool(np.isfinite(out_last).all()))
    print("max_abs_diff:", float(diff.max()))
    print("mean_abs_diff:", float(diff.mean()))
    print("cosine:", cosine(out_last, ref_last))

    top_c = np.argsort(-out_last[0])[:10]
    top_r = np.argsort(-ref_last[0])[:10]
    overlap = len(set(top_c.tolist()) & set(top_r.tolist()))
    print("top10_overlap:", overlap)
    print("top10_cactus:", top_c.tolist())
    print("top10_ref:", top_r.tolist())


if __name__ == "__main__":
    main()
