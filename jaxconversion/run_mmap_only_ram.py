#!/usr/bin/env python3
import argparse
import gc
import os
import sys
import time
from pathlib import Path

import numpy as np

from parse import parse_mlir
from lower_to_cactus import lower_to_cactus, decode_stablehlo_const


def mem_info():
    """
    Returns memory in MiB.

    RSS = resident set size. Includes file-backed mmap pages once touched.
    USS = unique set size. Better estimate of memory private to this process.
    """
    try:
        import psutil

        p = psutil.Process(os.getpid())
        mi = p.memory_info()
        full = p.memory_full_info()

        return {
            "rss": mi.rss / 1024 / 1024,
            "vms": mi.vms / 1024 / 1024,
            "uss": getattr(full, "uss", 0) / 1024 / 1024,
            "pss": getattr(full, "pss", 0) / 1024 / 1024,
        }

    except Exception:
        import resource

        val = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        # macOS reports bytes; Linux reports KiB.
        if sys.platform == "darwin":
            rss = val / 1024 / 1024
        else:
            rss = val / 1024

        return {
            "rss": rss,
            "vms": 0.0,
            "uss": 0.0,
            "pss": 0.0,
        }


def print_mem(label):
    gc.collect()
    m = mem_info()

    if m["uss"] > 0:
        print(
            f"[MEM] {label:<38} "
            f"RSS={m['rss']:9.2f} MiB  "
            f"USS={m['uss']:9.2f} MiB  "
            f"PSS={m['pss']:9.2f} MiB"
        )
    else:
        print(f"[MEM] {label:<38} RSS={m['rss']:9.2f} MiB")


def tensor_dtype(t):
    try:
        return int(t.dtype)
    except Exception:
        return 1


def decode_const(value, dtype):
    try:
        return decode_stablehlo_const(value, dtype)
    except Exception:
        pass

    if isinstance(value, np.generic):
        value = value.item()

    if isinstance(value, str):
        s = value.strip()
        low = s.lower()

        if low in ("inf", "+inf", "infinity", "+infinity"):
            return np.inf
        if low in ("-inf", "-infinity"):
            return -np.inf
        if low in ("nan", "+nan", "-nan"):
            return np.nan

        if low.startswith("0x"):
            bits = int(low, 16)
            dtype_s = str(dtype).lower()

            if "f16" in dtype_s:
                return np.array([bits & 0xFFFF], dtype=np.uint16).view(np.float16)[0].item()

            if "bf16" in dtype_s:
                return np.array([(bits & 0xFFFF) << 16], dtype=np.uint32).view(np.float32)[0].item()

            if "f32" in dtype_s:
                return np.array([bits & 0xFFFFFFFF], dtype=np.uint32).view(np.float32)[0].item()

        return float(s)

    return value


def set_one_input(g, tensor, data):
    dt = tensor_dtype(tensor)

    if dt == 1:
        arr = np.asarray(data, dtype=np.float16)
        g.set_input(tensor, arr, dtype=1)

    elif dt == 2:
        arr = np.asarray(data, dtype=np.float32)
        g.set_input(tensor, arr, dtype=2)

    else:
        arr = np.asarray(data, dtype=np.float32)
        g.set_input(tensor, arr, dtype=dt)


def set_constants(g, env, ir):
    for ssa, const in ir.constants.items():
        if ssa not in env:
            continue

        tensor = env[ssa]
        dt = tensor_dtype(tensor)
        shape = list(const.shape) if const.shape else [1]
        scalar = decode_const(const.value, const.dtype)

        if dt == 1:
            scalar_f = float(scalar)
            max_f16 = float(np.finfo(np.float16).max)

            if np.isnan(scalar_f):
                scalar_f = -10000.0
            elif np.isneginf(scalar_f) or scalar_f < -max_f16:
                scalar_f = -10000.0
            elif np.isposinf(scalar_f) or scalar_f > max_f16:
                scalar_f = max_f16

            arr = np.full(shape, scalar_f, dtype=np.float16)
            g.set_input(tensor, arr, dtype=1)

        elif dt == 2:
            scalar_f = float(scalar)

            if np.isnan(scalar_f):
                scalar_f = -10000.0
            elif np.isneginf(scalar_f):
                scalar_f = -10000.0
            elif np.isposinf(scalar_f):
                scalar_f = float(np.finfo(np.float32).max)

            arr = np.full(shape, scalar_f, dtype=np.float32)
            g.set_input(tensor, arr, dtype=2)

        else:
            arr = np.full(shape, float(scalar), dtype=np.float16)
            g.set_input(tensor, arr, dtype=dt)


def make_mmap_resolver(weights_dir):
    weights_dir = Path(weights_dir)

    def resolver(graph, ssa, shape, dtype):
        if not ssa.startswith("%arg"):
            return None

        try:
            idx = int(ssa[len("%arg"):])
        except Exception:
            return None

        # Small-Qwen2 test convention:
        # %arg0 = input_ids
        # %arg1 = position_ids
        # %arg2+ = weights
        if idx < 2:
            return None

        path = weights_dir / f"arg{idx}.weights"

        if not path.exists():
            raise FileNotFoundError(f"Missing mmap file for {ssa}: {path}")

        t = graph.mmap_weights(str(path))

        got = tuple(int(x) for x in t.shape)
        want = tuple(int(x) for x in shape)

        if got != want:
            raise ValueError(f"{ssa} shape mismatch: mmap={got}, mlir={want}, file={path}")

        return t

    return resolver


def summarize_bench(name, times):
    times = np.asarray(times, dtype=np.float64)
    print(f"\n{name}:")
    print(f"  runs:   {len(times)}")
    print(f"  mean:   {times.mean():.4f} ms")
    print(f"  median: {np.median(times):.4f} ms")
    print(f"  p95:    {np.percentile(times, 95):.4f} ms")
    print(f"  min:    {times.min():.4f} ms")
    print(f"  max:    {times.max():.4f} ms")


def bench(name, fn, warmup, iters):
    for _ in range(warmup):
        fn()

    times = []

    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000.0)

    summarize_bench(name, times)


def compare_optional(ref_path, out):
    if not ref_path:
        return

    p = Path(ref_path)
    if not p.exists():
        print(f"Reference not found, skipping compare: {p}")
        return

    ref = np.load(p).astype(np.float32)
    got = np.asarray(out, dtype=np.float32)

    print("\nCOMPARE:")
    print("  ref:", ref.shape, ref.dtype)
    print("  got:", got.shape, got.dtype)

    finite = np.isfinite(ref) & np.isfinite(got)
    diff = np.abs(ref - got)

    denom = np.linalg.norm(ref[finite]) * np.linalg.norm(got[finite])
    cos = float(np.dot(ref[finite], got[finite]) / denom) if denom else float("nan")

    print("  max diff:", float(np.max(diff[finite])))
    print("  mean diff:", float(np.mean(diff[finite])))
    print("  p99 diff:", float(np.percentile(diff[finite], 99)))
    print("  cosine:", cos)
    print("  argmax ref:", int(np.argmax(ref[0, -1])))
    print("  argmax got:", int(np.argmax(got[0, -1])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mlir", default="small_qwen2_mmap_test.stablehlo.mlir")
    ap.add_argument("--weights-dir", default="small_qwen2_mmap_weights")
    ap.add_argument("--input-ids", default="small_qwen2_france_input_ids.npy")
    ap.add_argument("--position-ids", default="small_qwen2_france_position_ids.npy")
    ap.add_argument("--ref", default="small_qwen2_france_cactus_logits.npy")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--readback", action="store_true")
    args = ap.parse_args()

    print("MMAP-only RAM test")
    print("mlir:", args.mlir)
    print("weights:", args.weights_dir)
    print("input_ids:", args.input_ids)
    print("position_ids:", args.position_ids)

    print_mem("process start")

    input_ids = np.load(args.input_ids)
    position_ids = np.load(args.position_ids)

    print("input_ids shape:", input_ids.shape, input_ids.dtype)
    print("position_ids shape:", position_ids.shape, position_ids.dtype)

    print_mem("after loading tiny runtime inputs")

    mlir_text = Path(args.mlir).read_text()
    ir = parse_mlir(mlir_text)

    print("IR:")
    print("  inputs:", len(ir.inputs))
    print("  outputs:", ir.outputs)
    print("  nodes:", len(ir.order))
    print("  constants:", len(ir.constants))

    print_mem("after parse MLIR")

    t0 = time.perf_counter()
    g, env = lower_to_cactus(
        ir,
        patterns=["default"],
        verbose=False,
        strict_math=False,
        input_resolver=make_mmap_resolver(args.weights_dir),
    )
    build_ms = (time.perf_counter() - t0) * 1000.0

    print(f"build: {build_ms:.3f} ms")
    print_mem("after mmap graph build")

    set_one_input(g, env["%arg0"], input_ids)
    set_one_input(g, env["%arg1"], position_ids)
    set_constants(g, env, ir)

    print_mem("after runtime inputs/constants")

    t0 = time.perf_counter()
    g.execute()
    first_ms = (time.perf_counter() - t0) * 1000.0

    print(f"first execute: {first_ms:.3f} ms")
    print_mem("after first execute")

    if args.readback:
        out = env[ir.outputs[0]].numpy()
        print("output shape:", out.shape, out.dtype)
        print_mem("after output numpy readback")
        compare_optional(args.ref, out)
    else:
        # Do one readback for correctness but don't count it as part of the memory
        # benchmark unless --readback is passed.
        out = env[ir.outputs[0]].numpy()
        compare_optional(args.ref, out)
        del out
        gc.collect()
        print_mem("after temp compare readback freed")

    bench(
        "mmap execute only",
        lambda: g.execute(),
        warmup=args.warmup,
        iters=args.iters,
    )

    print_mem("after execute-only bench")

    if args.readback:
        bench(
            "mmap execute + readback",
            lambda: (g.execute(), env[ir.outputs[0]].numpy())[1],
            warmup=args.warmup,
            iters=args.iters,
        )
        print_mem("after execute+readback bench")

    print("\nDONE")
    print("Note:")
    print("  RSS includes mapped file-backed pages after execution touches weights.")
    print("  USS is usually the better number for private process memory.")
    print("  Run this in a fresh terminal process, not after HF/JAX scripts.")


if __name__ == "__main__":
    main()
