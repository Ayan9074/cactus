#!/usr/bin/env python3
import argparse
import gc
import time
import traceback
from collections import Counter
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from test import jax_to_mlir
from parse import parse_mlir
from lower_to_cactus import lower_to_cactus, decode_stablehlo_const
from pattern_registry import Pattern

from test_small_qwen2_cactus import (
    DEFAULT_MODEL_ID,
    extract_params,
    make_qwen2_forward,
    make_inputs,
    compare,
)

import patterns.default as default_patterns_module

# ================================================================
# Timing / memory helpers
# ================================================================

def mem_mib():
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1024 / 1024
    except Exception:
        return 0.0


def print_mem(label):
    gc.collect()
    m = mem_mib()
    if m:
        print(f"[RSS] {label:<35} {m:9.2f} MiB")


def summarize_times(name, times_ms):
    times = np.asarray(times_ms, dtype=np.float64)
    print(f"\n{name}:")
    print(f"  runs:   {len(times)}")
    print(f"  mean:   {times.mean():.4f} ms")
    print(f"  median: {np.median(times):.4f} ms")
    print(f"  p95:    {np.percentile(times, 95):.4f} ms")
    print(f"  min:    {times.min():.4f} ms")
    print(f"  max:    {times.max():.4f} ms")
    return {
        "mean": float(times.mean()),
        "median": float(np.median(times)),
        "p95": float(np.percentile(times, 95)),
        "min": float(times.min()),
        "max": float(times.max()),
    }


def bench(name, fn, warmup=10, iters=50):
    for _ in range(warmup):
        fn()

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000.0)

    return summarize_times(name, times)


# ================================================================
# Pattern counting wrapper
# ================================================================

def wrap_patterns(patterns, counts, print_fusions=False):
    wrapped = []

    for p in patterns:
        original_handler = p.handler

        def make_handler(pattern_name, handler):
            def wrapped_handler(ctx, nodes, idx):
                result = handler(ctx, nodes, idx)

                if result is not None:
                    counts[pattern_name] += 1

                    if print_fusions:
                        consumed = result[1]
                        lead = nodes[idx]
                        print(
                            f"FUSED {pattern_name:<16} "
                            f"idx={idx:<5} consumed={consumed:<3} "
                            f"lead={lead.id:<12} op={lead.op:<30} "
                            f"outputs={lead.outputs}"
                        )

                return result
            return wrapped_handler

        wrapped.append(
            Pattern(
                name=p.name,
                handler=make_handler(p.name, original_handler),
                trigger_ops=p.trigger_ops,
            )
        )

    return wrapped


def op_counts(ir):
    c = Counter()
    for nid in ir.order:
        c[ir.nodes[nid].op] += 1
    return c


# ================================================================
# Cactus input helpers
# ================================================================

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


def set_inputs_and_constants(g, env, ir, args_np):
    for i, data in enumerate(args_np):
        ssa = f"%arg{i}"
        if ssa not in env:
            raise RuntimeError(f"missing {ssa}")
        set_one_input(g, env[ssa], data)

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


# ================================================================
# Mode runner
# ================================================================

def run_cactus_mode(
    mode_name,
    ir,
    args_np,
    jax_logits_np,
    patterns,
    bench_iters,
    bench_warmup,
    print_fusions=False,
):
    print("\n" + "=" * 90)
    print(f"MODE: {mode_name}")
    print("=" * 90)

    fusion_counts = Counter()

    if patterns is None:
        lowering_patterns = []
    else:
        lowering_patterns = wrap_patterns(patterns, fusion_counts, print_fusions=print_fusions)

    print_mem(f"{mode_name}: before build")

    t0 = time.perf_counter()
    g, env = lower_to_cactus(
        ir,
        patterns=lowering_patterns,
        verbose=False,
        strict_math=False,
    )
    build_ms = (time.perf_counter() - t0) * 1000.0

    print(f"{mode_name} build: {build_ms:.4f} ms")
    print_mem(f"{mode_name}: after build")

    t0 = time.perf_counter()
    set_inputs_and_constants(g, env, ir, args_np)
    set_inputs_ms = (time.perf_counter() - t0) * 1000.0

    print(f"{mode_name} set inputs/constants: {set_inputs_ms:.4f} ms")
    print_mem(f"{mode_name}: after set inputs")

    t0 = time.perf_counter()
    g.execute()
    first_exec_ms = (time.perf_counter() - t0) * 1000.0

    print(f"{mode_name} first execute: {first_exec_ms:.4f} ms")

    t0 = time.perf_counter()
    logits = env[ir.outputs[0]].numpy()
    readback_ms = (time.perf_counter() - t0) * 1000.0

    print(f"{mode_name} first readback: {readback_ms:.4f} ms")
    print_mem(f"{mode_name}: after first execute/readback")

    ok = compare(f"{mode_name} vs JAX", jax_logits_np, logits)

    exec_stats = bench(
        f"{mode_name} warmed execute only",
        lambda: g.execute(),
        warmup=bench_warmup,
        iters=bench_iters,
    )

    exec_read_stats = bench(
        f"{mode_name} warmed execute + readback",
        lambda: (g.execute(), env[ir.outputs[0]].numpy())[1],
        warmup=bench_warmup,
        iters=bench_iters,
    )

    print("\nFusion counts:")
    if fusion_counts:
        for name, n in fusion_counts.most_common():
            print(f"  {name:<16} {n}")
    else:
        print("  none")

    return {
        "name": mode_name,
        "ok": bool(ok),
        "logits": logits,
        "fusion_counts": dict(fusion_counts),
        "build_ms": build_ms,
        "set_inputs_ms": set_inputs_ms,
        "first_exec_ms": first_exec_ms,
        "first_readback_ms": readback_ms,
        "exec_stats": exec_stats,
        "exec_read_stats": exec_read_stats,
    }


def compare_modes(base, fused):
    a = np.asarray(base["logits"], dtype=np.float32)
    b = np.asarray(fused["logits"], dtype=np.float32)

    print("\n" + "=" * 90)
    print("BASELINE VS FUSED LOGITS")
    print("=" * 90)

    finite = np.isfinite(a) & np.isfinite(b)
    diff = np.abs(a - b)

    denom = np.linalg.norm(a[finite]) * np.linalg.norm(b[finite])
    cos = float(np.dot(a[finite], b[finite]) / denom) if denom else float("nan")

    print("shape:", a.shape)
    print("max diff:", float(np.max(diff[finite])))
    print("mean diff:", float(np.mean(diff[finite])))
    print("p95 diff:", float(np.percentile(diff[finite], 95)))
    print("p99 diff:", float(np.percentile(diff[finite], 99)))
    print("cosine:", cos)
    print("baseline argmax:", int(np.argmax(a[0, -1])))
    print("fused argmax:   ", int(np.argmax(b[0, -1])))


def pct_speedup(old, new):
    if old == 0:
        return 0.0
    return (old - new) / old * 100.0


def print_summary(base, fused):
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)

    print(f"baseline correct: {base['ok']}")
    print(f"fused correct:    {fused['ok']}")

    print("\nFusion counts:")
    for name, n in sorted(fused["fusion_counts"].items()):
        print(f"  {name:<16} {n}")

    rows = [
        ("build", base["build_ms"], fused["build_ms"]),
        ("set inputs", base["set_inputs_ms"], fused["set_inputs_ms"]),
        ("first execute", base["first_exec_ms"], fused["first_exec_ms"]),
        ("first readback", base["first_readback_ms"], fused["first_readback_ms"]),
        ("warmed execute mean", base["exec_stats"]["mean"], fused["exec_stats"]["mean"]),
        ("warmed execute median", base["exec_stats"]["median"], fused["exec_stats"]["median"]),
        ("warmed exec+read mean", base["exec_read_stats"]["mean"], fused["exec_read_stats"]["mean"]),
        ("warmed exec+read median", base["exec_read_stats"]["median"], fused["exec_read_stats"]["median"]),
    ]

    print("\nTiming comparison:")
    print(f"{'metric':<28} {'baseline ms':>14} {'fused ms':>14} {'speedup':>12}")
    print("-" * 72)

    for label, old, new in rows:
        print(f"{label:<28} {old:14.4f} {new:14.4f} {pct_speedup(old, new):11.2f}%")

    print("\nInterpretation:")
    print("  Positive speedup means fused was faster.")
    print("  Small models can be noisy; run with --bench-iters 100 or 300 for cleaner numbers.")
    print("  If correctness passes and timings are similar, fusion still helps graph size/readability.")


# ================================================================
# Main
# ================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--max-length", type=int, default=16)
    ap.add_argument("--bench-iters", type=int, default=100)
    ap.add_argument("--bench-warmup", type=int, default=20)
    ap.add_argument("--save-prefix", default="small_qwen2_fusion_bench")
    ap.add_argument("--print-fusions", action="store_true")
    args = ap.parse_args()

    print("Qwen2 before/after default.py fusion benchmark")
    print("model:", args.model_id)
    print("prompt:", repr(args.prompt))
    print("max_length:", args.max_length)
    print("bench_iters:", args.bench_iters)

    print_mem("start")

    print("\nLoading HF model...")
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

    print_mem("after HF load")

    input_ids_np, position_ids_np = make_inputs(tok, cfg, args.prompt, args.max_length)

    print("\nInput:")
    print("  ids:", input_ids_np.tolist())
    print("  decoded:", repr(tok.decode(input_ids_np[0].tolist())))

    print("\nExtracting params...")
    params_np = extract_params(model)

    params_bytes = sum(np.asarray(p, dtype=np.float16).nbytes for p in params_np)
    print(f"params: {len(params_np)} tensors, {params_bytes / 1024 / 1024:.2f} MiB fp16")

    print_mem("after params")

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

    print(f"JAX first run: {jax_first_ms:.4f} ms")
    print("JAX logits:", jax_logits_np.shape, jax_logits_np.dtype)
    print("JAX argmax:", int(np.argmax(jax_logits_np[0, -1])))

    print_mem("after JAX reference")

    print("\nExporting StableHLO...")
    mlir = jax_to_mlir(forward, all_args_jax)

    mlir_path = f"{args.save_prefix}.stablehlo.mlir"
    Path(mlir_path).write_text(mlir)
    print("saved:", mlir_path)

    ir = parse_mlir(mlir)

    print("\nIR summary:")
    print("  inputs:", len(ir.inputs))
    print("  outputs:", ir.outputs)
    print("  nodes:", len(ir.order))
    print("  constants:", len(ir.constants))

    print("\nOp counts:")
    for op, n in sorted(op_counts(ir).items()):
        print(f"  {op:<36} {n}")

    print("\nDefault patterns loaded from:")
    print(" ", default_patterns_module.__file__)
    print("Patterns:")
    for p in default_patterns_module.DEFAULT_PATTERNS:
        print(f"  {p.name:<16} triggers={sorted(p.trigger_ops)}")

    # Baseline: no default.py fusions.
    # The internal mean_keepdims canonicalizer still runs in lower_to_cactus.
    baseline = run_cactus_mode(
        mode_name="baseline_no_default_patterns",
        ir=ir,
        args_np=all_args_np,
        jax_logits_np=jax_logits_np,
        patterns=None,
        bench_iters=args.bench_iters,
        bench_warmup=args.bench_warmup,
        print_fusions=args.print_fusions,
    )

    # Fused: current DEFAULT_PATTERNS.
    fused = run_cactus_mode(
        mode_name="fused_default_patterns",
        ir=ir,
        args_np=all_args_np,
        jax_logits_np=jax_logits_np,
        patterns=default_patterns_module.DEFAULT_PATTERNS,
        bench_iters=args.bench_iters,
        bench_warmup=args.bench_warmup,
        print_fusions=args.print_fusions,
    )

    compare_modes(baseline, fused)
    print_summary(baseline, fused)

    np.save(f"{args.save_prefix}_jax_logits.npy", jax_logits_np)
    np.save(f"{args.save_prefix}_baseline_logits.npy", baseline["logits"])
    np.save(f"{args.save_prefix}_fused_logits.npy", fused["logits"])

    print("\nSaved:")
    print(f"  {args.save_prefix}_jax_logits.npy")
    print(f"  {args.save_prefix}_baseline_logits.npy")
    print(f"  {args.save_prefix}_fused_logits.npy")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\nFAILED")
        print(type(e).__name__ + ":", str(e))
        traceback.print_exc(limit=40)
        raise
