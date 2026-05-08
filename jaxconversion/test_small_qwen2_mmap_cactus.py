#!/usr/bin/env python3
import argparse
import gc
import json
import math
import os
import shutil
import struct
import time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from test import jax_to_mlir
from parse import parse_mlir
from lower_to_cactus import lower_to_cactus, decode_stablehlo_const

# Reuse your already-working Small-Qwen2 implementation.
from test_small_qwen2_cactus import (
    DEFAULT_MODEL_ID,
    extract_params,
    make_qwen2_forward,
    make_inputs,
    compare,
    bench_ms,
    op_counts,
)


# ================================================================
# Cactus .weights writer
# ================================================================

CACT_MAGIC = 0x54434143
PREC_FP16 = 1
HEADER_SIZE = 84
ALIGNMENT = 32


def align_offset(offset, alignment):
    rem = offset % alignment
    return offset if rem == 0 else offset + (alignment - rem)


def write_cactus_fp16_weight(path, arr):
    """
    Write a Cactus FP16 .weights file compatible with graph.mmap_weights.
    Header is 84 bytes, data starts aligned to 32 bytes.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    arr = np.ascontiguousarray(np.asarray(arr, dtype=np.float16))
    shape = list(arr.shape)

    if len(shape) > 4:
        raise ValueError(f"Cactus header supports <=4D tensors, got {shape} for {path}")

    ndim = len(shape)
    raw_shape = shape + [0] * (4 - ndim)

    flags = 0
    precision = PREC_FP16
    byte_size = arr.size * 2
    scales_bytes = 0
    group_size = 0
    num_groups = 0
    original_N = shape[0] if shape else 1

    header = bytearray(HEADER_SIZE)
    off = 0

    struct.pack_into("<I", header, off, CACT_MAGIC); off += 4
    struct.pack_into("<I", header, off, flags); off += 4
    struct.pack_into("<I", header, off, ALIGNMENT); off += 4
    struct.pack_into("<I", header, off, ndim); off += 4

    for dim in raw_shape:
        struct.pack_into("<Q", header, off, int(dim))
        off += 8

    struct.pack_into("<I", header, off, precision); off += 4
    struct.pack_into("<Q", header, off, byte_size); off += 8
    struct.pack_into("<Q", header, off, scales_bytes); off += 8
    struct.pack_into("<I", header, off, group_size); off += 4
    struct.pack_into("<I", header, off, num_groups); off += 4
    struct.pack_into("<Q", header, off, int(original_N)); off += 8

    assert off == HEADER_SIZE, off

    data_offset = align_offset(HEADER_SIZE, ALIGNMENT)
    padding = b"\x00" * (data_offset - HEADER_SIZE)

    with open(path, "wb") as f:
        f.write(header)
        f.write(padding)
        f.write(arr.tobytes())

    return path


def export_mmap_weights(weights_dir, params_np):
    """
    params_np corresponds to JAX params, but in StableHLO args:
      %arg0 = input_ids
      %arg1 = position_ids
      %arg2 onward = params_np[0 onward]
    """
    weights_dir = Path(weights_dir)

    if weights_dir.exists():
        shutil.rmtree(weights_dir)

    weights_dir.mkdir(parents=True, exist_ok=True)

    manifest = []

    for i, arr in enumerate(params_np):
        arg_idx = i + 2
        path = weights_dir / f"arg{arg_idx}.weights"
        write_cactus_fp16_weight(path, arr)

        manifest.append(
            {
                "arg": arg_idx,
                "file": path.name,
                "shape": list(np.asarray(arr).shape),
                "dtype": "fp16",
                "bytes": int(np.asarray(arr, dtype=np.float16).size * 2),
            }
        )

    manifest_path = weights_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    total_bytes = sum(x["bytes"] for x in manifest)

    print(f"\nExported mmap weights -> {weights_dir}")
    print(f"  tensors: {len(manifest)}")
    print(f"  raw fp16 bytes: {total_bytes:,} ({total_bytes / 1024 / 1024:.2f} MiB)")
    print(f"  manifest: {manifest_path}")

    return manifest


# ================================================================
# RSS / memory helpers
# ================================================================

def get_rss_bytes():
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss
    except Exception:
        # macOS/Linux fallback. On macOS ru_maxrss is bytes; on Linux it is KiB.
        import resource
        val = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            return int(val)
        return int(val) * 1024


def rss_mib():
    return get_rss_bytes() / 1024 / 1024


def print_mem(label):
    gc.collect()
    print(f"[RSS] {label:<40} {rss_mib():9.2f} MiB")


# ================================================================
# Constants/runtime setting
# ================================================================

def tensor_dtype(t):
    try:
        return int(t.dtype)
    except Exception:
        return 1


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


def set_all_inputs_and_constants(g, env, ir, all_args_np):
    for i, data in enumerate(all_args_np):
        ssa = f"%arg{i}"
        if ssa not in env:
            raise RuntimeError(f"missing {ssa}")
        set_one_input(g, env[ssa], data)

    set_constants(g, env, ir)


def set_runtime_inputs_and_constants_only(g, env, ir, input_ids_np, position_ids_np):
    set_one_input(g, env["%arg0"], input_ids_np)
    set_one_input(g, env["%arg1"], position_ids_np)
    set_constants(g, env, ir)


# ================================================================
# mmap resolver
# ================================================================

def make_mmap_resolver(weights_dir):
    weights_dir = Path(weights_dir)

    def resolver(graph, ssa, shape, dtype):
        if not ssa.startswith("%arg"):
            return None

        try:
            idx = int(ssa[len("%arg"):])
        except Exception:
            return None

        # arg0 input_ids and arg1 position_ids stay runtime inputs.
        if idx < 2:
            return None

        path = weights_dir / f"arg{idx}.weights"

        if not path.exists():
            raise FileNotFoundError(f"Missing mmap weight for {ssa}: {path}")

        t = graph.mmap_weights(str(path))

        got_shape = tuple(int(x) for x in t.shape)
        want_shape = tuple(int(x) for x in shape)

        if got_shape != want_shape:
            raise ValueError(
                f"mmap shape mismatch for {ssa}: file {path.name} has {got_shape}, MLIR wants {want_shape}"
            )

        return t

    return resolver


# ================================================================
# Benchmark
# ================================================================

def summarize_bench(name, times_ms):
    times = np.asarray(times_ms, dtype=np.float64)
    print(f"\n{name}:")
    print(f"  runs:   {len(times)}")
    print(f"  mean:   {times.mean():.4f} ms")
    print(f"  median: {np.median(times):.4f} ms")
    print(f"  p95:    {np.percentile(times, 95):.4f} ms")
    print(f"  min:    {times.min():.4f} ms")
    print(f"  max:    {times.max():.4f} ms")


def bench_ms(name, fn, warmup=5, iters=30):
    for _ in range(warmup):
        fn()

    times = []

    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000.0)

    summarize_bench(name, times)
    return times


# ================================================================
# Main
# ================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--max-length", type=int, default=16)
    ap.add_argument("--weights-dir", default="small_qwen2_mmap_weights")
    ap.add_argument("--bench-iters", type=int, default=30)
    args = ap.parse_args()

    print("Loading:", args.model_id)
    print_mem("start")

    tok = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(args.model_id)
    model.eval()

    cfg = model.config

    print("config:")
    print("  vocab_size:", cfg.vocab_size)
    print("  hidden_size:", cfg.hidden_size)
    print("  intermediate_size:", cfg.intermediate_size)
    print("  num_hidden_layers:", cfg.num_hidden_layers)
    print("  num_attention_heads:", cfg.num_attention_heads)
    print("  num_key_value_heads:", getattr(cfg, "num_key_value_heads", cfg.num_attention_heads))

    input_ids_np, position_ids_np = make_inputs(tok, cfg, args.prompt, args.max_length)

    print("\nprompt:", repr(args.prompt))
    print("input_ids:", input_ids_np.tolist())
    print("decoded:", repr(tok.decode(input_ids_np[0].tolist())))

    print_mem("after loading HF model")

    print("\nExtracting params...")
    params_np = extract_params(model)

    print("param tensors:", len(params_np))
    param_bytes = sum(np.asarray(p, dtype=np.float16).nbytes for p in params_np)
    print(f"param fp16 bytes: {param_bytes:,} ({param_bytes / 1024 / 1024:.2f} MiB)")

    print_mem("after extracting params")

    forward = make_qwen2_forward(cfg)

    all_args_np = (input_ids_np, position_ids_np, *params_np)
    all_args_jax = tuple(jnp.asarray(x) for x in all_args_np)

    print("\nRunning JAX reference...")
    jitted = jax.jit(forward)

    t0 = time.perf_counter()
    jax_logits = jitted(*all_args_jax)
    jax.block_until_ready(jax_logits)
    jax_first_ms = (time.perf_counter() - t0) * 1000.0

    jax_logits_np = np.asarray(jax_logits)

    print(f"JAX first run: {jax_first_ms:.2f}ms")
    print("JAX logits:", jax_logits_np.shape, jax_logits_np.dtype)
    print_mem("after JAX reference")

    print("\nExporting StableHLO...")
    mlir = jax_to_mlir(forward, all_args_jax)
    ir = parse_mlir(mlir)

    print("IR:")
    print("  inputs:", len(ir.inputs))
    print("  outputs:", ir.outputs)
    print("  nodes:", len(ir.order))
    print("  constants:", len(ir.constants))

    counts = op_counts(ir)
    print("\nOps:")
    for op, n in sorted(counts.items()):
        print(f"  {op:<36} {n}")

    Path("small_qwen2_mmap_test.stablehlo.mlir").write_text(mlir)

    print_mem("after StableHLO export")

    # ------------------------------------------------------------
    # Normal set_input Cactus path
    # ------------------------------------------------------------
    print("\n" + "=" * 80)
    print("NORMAL SET_INPUT CACTUS PATH")
    print("=" * 80)

    print_mem("before normal build")

    t0 = time.perf_counter()
    g_normal, env_normal = lower_to_cactus(
        ir,
        patterns=["default"],
        verbose=False,
        strict_math=False,
    )
    normal_build_ms = (time.perf_counter() - t0) * 1000.0

    print(f"normal build: {normal_build_ms:.2f}ms")
    print_mem("after normal build before set_input")

    set_all_inputs_and_constants(g_normal, env_normal, ir, all_args_np)
    print_mem("after normal set_input all weights")

    t0 = time.perf_counter()
    g_normal.execute()
    normal_exec_ms = (time.perf_counter() - t0) * 1000.0

    normal_logits = env_normal[ir.outputs[0]].numpy()

    print(f"normal first execute: {normal_exec_ms:.2f}ms")
    print_mem("after normal first execute")

    compare("normal Cactus vs JAX", jax_logits_np, normal_logits)

    bench_ms(
        "normal Cactus warmed execute",
        lambda: g_normal.execute(),
        warmup=5,
        iters=args.bench_iters,
    )

    print_mem("after normal warmed bench")

    # ------------------------------------------------------------
    # Export .weights
    # ------------------------------------------------------------
    print("\n" + "=" * 80)
    print("EXPORT MMAP WEIGHTS")
    print("=" * 80)

    export_mmap_weights(args.weights_dir, params_np)
    print_mem("after writing mmap weights")

    # Optional: drop Python params before mmap build to show memory behavior.
    # But keep JAX logits and all_args input ids. We no longer need params_np/all_args_jax.
    del params_np
    del all_args_jax
    gc.collect()

    print_mem("after deleting params_np/all_args_jax")

    # ------------------------------------------------------------
    # mmap Cactus path
    # ------------------------------------------------------------
    print("\n" + "=" * 80)
    print("MMAP CACTUS PATH")
    print("=" * 80)

    print_mem("before mmap build")

    t0 = time.perf_counter()
    g_mmap, env_mmap = lower_to_cactus(
        ir,
        patterns=["default"],
        verbose=False,
        strict_math=False,
        input_resolver=make_mmap_resolver(args.weights_dir),
    )
    mmap_build_ms = (time.perf_counter() - t0) * 1000.0

    print(f"mmap build: {mmap_build_ms:.2f}ms")
    print_mem("after mmap build before runtime inputs")

    set_runtime_inputs_and_constants_only(
        g_mmap,
        env_mmap,
        ir,
        input_ids_np=input_ids_np,
        position_ids_np=position_ids_np,
    )

    print_mem("after setting runtime inputs/constants only")

    t0 = time.perf_counter()
    g_mmap.execute()
    mmap_first_exec_ms = (time.perf_counter() - t0) * 1000.0

    mmap_logits = env_mmap[ir.outputs[0]].numpy()

    print(f"mmap first execute: {mmap_first_exec_ms:.2f}ms")
    print_mem("after mmap first execute")

    compare("mmap Cactus vs JAX", jax_logits_np, mmap_logits)
    compare("mmap Cactus vs normal Cactus", normal_logits, mmap_logits)

    bench_ms(
        "mmap Cactus warmed execute",
        lambda: g_mmap.execute(),
        warmup=5,
        iters=args.bench_iters,
    )

    bench_ms(
        "mmap Cactus warmed execute + readback",
        lambda: (g_mmap.execute(), env_mmap[ir.outputs[0]].numpy())[1],
        warmup=5,
        iters=args.bench_iters,
    )

    print_mem("after mmap warmed bench")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"normal build:          {normal_build_ms:.2f}ms")
    print(f"normal first execute:  {normal_exec_ms:.2f}ms")
    print(f"mmap build:            {mmap_build_ms:.2f}ms")
    print(f"mmap first execute:    {mmap_first_exec_ms:.2f}ms")
    print("\nExpected interpretation:")
    print("  - mmap should reduce Python-side weight copies / set_input memory pressure.")
    print("  - warmed execute may be similar or slightly slower/faster.")
    print("  - first mmap execute can include page faults.")
    print("  - RSS on macOS includes resident mapped pages after they are touched.")


if __name__ == "__main__":
    main()
