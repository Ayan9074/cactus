#!/usr/bin/env python3
import argparse
import gc
import os
import time
import traceback
from collections import Counter
from pathlib import Path

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import jax
import jax.numpy as jnp
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from test import jax_to_mlir
from parse import parse_mlir
from lower_to_cactus import lower_to_cactus, decode_stablehlo_const

from test_small_qwen2_cactus import (
    DEFAULT_MODEL_ID,
    extract_params,
    make_inputs,
)


# ================================================================
# Memory helpers
# ================================================================

def mem_info():
    try:
        import psutil
        p = psutil.Process(os.getpid())
        mi = p.memory_info()
        full = p.memory_full_info()
        return {
            "rss": mi.rss / 1024 / 1024,
            "uss": getattr(full, "uss", 0) / 1024 / 1024,
            "pss": getattr(full, "pss", 0) / 1024 / 1024,
        }
    except Exception:
        return {"rss": 0.0, "uss": 0.0, "pss": 0.0}


def print_mem(label):
    gc.collect()
    m = mem_info()
    print(
        f"[MEM] {label:<38} "
        f"RSS={m['rss']:9.2f} MiB  "
        f"USS={m['uss']:9.2f} MiB  "
        f"PSS={m['pss']:9.2f} MiB"
    )


# ================================================================
# Model helpers
# ================================================================

def silu(x):
    return x * jax.nn.sigmoid(x)


def rms_norm(x, weight, eps):
    x32 = x.astype(jnp.float32)
    var = jnp.mean(x32 * x32, axis=-1, keepdims=True)
    y = x32 * jax.lax.rsqrt(var + jnp.asarray(eps, dtype=jnp.float32))
    return y.astype(x.dtype) * weight.astype(x.dtype)


def rotate_half(x):
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return jnp.concatenate((-x2, x1), axis=-1)


def apply_rope(q, k, position_ids, rope_theta):
    """
    q: [B,H,T,DH]
    k: [B,KVH,T,DH]
    position_ids: [B,T]
    """
    head_dim = q.shape[-1]

    freq_i = jnp.arange(0, head_dim, 2, dtype=jnp.float32)
    inv_freq = jnp.asarray(1.0, dtype=jnp.float32) / (
        jnp.asarray(rope_theta, dtype=jnp.float32)
        ** (freq_i / jnp.asarray(head_dim, dtype=jnp.float32))
    )

    freqs = position_ids.astype(jnp.float32)[:, :, None] * inv_freq[None, None, :]
    emb = jnp.concatenate((freqs, freqs), axis=-1)

    cos = jnp.cos(emb)[:, None, :, :].astype(q.dtype)
    sin = jnp.sin(emb)[:, None, :, :].astype(q.dtype)

    q2 = q * cos + rotate_half(q) * sin
    k2 = k * cos + rotate_half(k) * sin
    return q2, k2


def split_heads(x, n_heads, head_dim):
    b, t, _ = x.shape
    x = jnp.reshape(x, (b, t, n_heads, head_dim))
    return jnp.transpose(x, (0, 2, 1, 3))


def merge_heads(x):
    x = jnp.transpose(x, (0, 2, 1, 3))
    b, t, h, d = x.shape
    return jnp.reshape(x, (b, t, h * d))


def repeat_kv(x, n_heads, n_kv_heads):
    reps = n_heads // n_kv_heads
    pieces = []
    for kv_i in range(n_kv_heads):
        for _ in range(reps):
            pieces.append(x[:, kv_i:kv_i + 1, :, :])
    return jnp.concatenate(tuple(pieces), axis=1)


def layer_param_count():
    return 16


def unpack_layer(params, p):
    return (
        params[p + 0],   # input_norm_w
        params[p + 1],   # q_w
        params[p + 2],   # q_b
        params[p + 3],   # k_w
        params[p + 4],   # k_b
        params[p + 5],   # v_w
        params[p + 6],   # v_b
        params[p + 7],   # o_w
        params[p + 8],   # o_b
        params[p + 9],   # post_norm_w
        params[p + 10],  # gate_w
        params[p + 11],  # gate_b
        params[p + 12],  # up_w
        params[p + 13],  # up_b
        params[p + 14],  # down_w
        params[p + 15],  # down_b
    )


def make_qwen2_prefill_and_decode(cfg):
    n_layer = int(cfg.num_hidden_layers)
    hidden = int(cfg.hidden_size)
    n_heads = int(cfg.num_attention_heads)
    n_kv_heads = int(getattr(cfg, "num_key_value_heads", n_heads))
    head_dim = int(getattr(cfg, "head_dim", hidden // n_heads))
    eps = float(getattr(cfg, "rms_norm_eps", 1e-6))
    rope_theta = float(getattr(cfg, "rope_theta", 10000.0) or 10000.0)

    def prefill_forward(input_ids, position_ids, *params):
        """
        Returns:
          logits: [B,T,V]
          k_cache: [L,B,KVH,T,DH]
          v_cache: [L,B,KVH,T,DH]
        """
        embed_w = params[0]
        final_norm_w = params[1]
        lm_head_w = params[2]

        x = embed_w[input_ids].astype(jnp.float16)

        p = 3
        k_layers = []
        v_layers = []

        for _layer in range(n_layer):
            (
                input_norm_w,
                q_w, q_b,
                k_w, k_b,
                v_w, v_b,
                o_w, o_b,
                post_norm_w,
                gate_w, gate_b,
                up_w, up_b,
                down_w, down_b,
            ) = unpack_layer(params, p)
            p += layer_param_count()

            residual = x
            h = rms_norm(x, input_norm_w, eps)

            q = jnp.einsum("btd,od->bto", h, q_w) + q_b
            k = jnp.einsum("btd,od->bto", h, k_w) + k_b
            v = jnp.einsum("btd,od->bto", h, v_w) + v_b

            q = split_heads(q, n_heads, head_dim)
            k = split_heads(k, n_kv_heads, head_dim)
            v = split_heads(v, n_kv_heads, head_dim)

            q, k = apply_rope(q, k, position_ids, rope_theta)

            # Store RoPE-applied k and raw v in cache.
            k_layers.append(k)
            v_layers.append(v)

            k_rep = repeat_kv(k, n_heads, n_kv_heads)
            v_rep = repeat_kv(v, n_heads, n_kv_heads)

            scores = jnp.einsum("bhtd,bhsd->bhts", q, k_rep)
            scores = scores * jnp.asarray(1.0 / np.sqrt(head_dim), dtype=scores.dtype)

            t = input_ids.shape[1]
            causal = jnp.tril(jnp.ones((t, t), dtype=bool))

            scores = jnp.where(
                causal[None, None, :, :],
                scores,
                jnp.asarray(-10000.0, dtype=scores.dtype),
            )

            probs = jax.nn.softmax(scores, axis=-1)
            ctx = jnp.einsum("bhts,bhsd->bhtd", probs, v_rep)
            ctx = merge_heads(ctx)

            attn_out = jnp.einsum("btd,od->bto", ctx, o_w) + o_b
            x = residual + attn_out

            residual = x
            h = rms_norm(x, post_norm_w, eps)

            gate = jnp.einsum("btd,od->bto", h, gate_w) + gate_b
            up = jnp.einsum("btd,od->bto", h, up_w) + up_b
            hidden_mlp = silu(gate) * up
            down = jnp.einsum("btd,od->bto", hidden_mlp, down_w) + down_b

            x = residual + down

        x = rms_norm(x, final_norm_w, eps)
        logits = jnp.einsum("btd,vd->btv", x, lm_head_w)

        k_cache = jnp.stack(tuple(k_layers), axis=0)
        v_cache = jnp.stack(tuple(v_layers), axis=0)

        return logits.astype(jnp.float16), k_cache.astype(jnp.float16), v_cache.astype(jnp.float16)

    def decode_forward(input_id, position_id, k_cache, v_cache, *params):
        """
        One-token decode.

        input_id: [B,1]
        position_id: [B,1]
        k_cache: [L,B,KVH,P,DH]
        v_cache: [L,B,KVH,P,DH]

        Returns:
          logits: [B,1,V]
          new_k_cache: [L,B,KVH,P+1,DH]
          new_v_cache: [L,B,KVH,P+1,DH]
        """
        embed_w = params[0]
        final_norm_w = params[1]
        lm_head_w = params[2]

        x = embed_w[input_id].astype(jnp.float16)

        p = 3
        new_k_layers = []
        new_v_layers = []

        for layer in range(n_layer):
            (
                input_norm_w,
                q_w, q_b,
                k_w, k_b,
                v_w, v_b,
                o_w, o_b,
                post_norm_w,
                gate_w, gate_b,
                up_w, up_b,
                down_w, down_b,
            ) = unpack_layer(params, p)
            p += layer_param_count()

            old_k = k_cache[layer]
            old_v = v_cache[layer]

            residual = x
            h = rms_norm(x, input_norm_w, eps)

            q = jnp.einsum("btd,od->bto", h, q_w) + q_b
            k = jnp.einsum("btd,od->bto", h, k_w) + k_b
            v = jnp.einsum("btd,od->bto", h, v_w) + v_b

            q = split_heads(q, n_heads, head_dim)
            k = split_heads(k, n_kv_heads, head_dim)
            v = split_heads(v, n_kv_heads, head_dim)

            q, k = apply_rope(q, k, position_id, rope_theta)

            k_cat = jnp.concatenate((old_k, k), axis=2)
            v_cat = jnp.concatenate((old_v, v), axis=2)

            new_k_layers.append(k_cat)
            new_v_layers.append(v_cat)

            k_rep = repeat_kv(k_cat, n_heads, n_kv_heads)
            v_rep = repeat_kv(v_cat, n_heads, n_kv_heads)

            scores = jnp.einsum("bhtd,bhsd->bhts", q, k_rep)
            scores = scores * jnp.asarray(1.0 / np.sqrt(head_dim), dtype=scores.dtype)

            # One-token decode attends to all cached tokens including itself.
            probs = jax.nn.softmax(scores, axis=-1)

            ctx = jnp.einsum("bhts,bhsd->bhtd", probs, v_rep)
            ctx = merge_heads(ctx)

            attn_out = jnp.einsum("btd,od->bto", ctx, o_w) + o_b
            x = residual + attn_out

            residual = x
            h = rms_norm(x, post_norm_w, eps)

            gate = jnp.einsum("btd,od->bto", h, gate_w) + gate_b
            up = jnp.einsum("btd,od->bto", h, up_w) + up_b
            hidden_mlp = silu(gate) * up
            down = jnp.einsum("btd,od->bto", hidden_mlp, down_w) + down_b

            x = residual + down

        x = rms_norm(x, final_norm_w, eps)
        logits = jnp.einsum("btd,vd->btv", x, lm_head_w)

        new_k_cache = jnp.stack(tuple(new_k_layers), axis=0)
        new_v_cache = jnp.stack(tuple(new_v_layers), axis=0)

        return logits.astype(jnp.float16), new_k_cache.astype(jnp.float16), new_v_cache.astype(jnp.float16)

    def full_forward(input_ids, position_ids, *params):
        logits, _, _ = prefill_forward(input_ids, position_ids, *params)
        return logits

    return prefill_forward, decode_forward, full_forward


# ================================================================
# StableHLO/Cactus helpers
# ================================================================

def op_counts(ir):
    c = Counter()
    for nid in ir.order:
        c[ir.nodes[nid].op] += 1
    return c


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
        g.set_input(tensor, np.asarray(data, dtype=np.float16), dtype=1)
    elif dt == 2:
        g.set_input(tensor, np.asarray(data, dtype=np.float32), dtype=2)
    else:
        g.set_input(tensor, np.asarray(data, dtype=np.float32), dtype=dt)


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


def set_args(g, env, values):
    for i, x in enumerate(values):
        ssa = f"%arg{i}"
        if ssa not in env:
            raise RuntimeError(f"missing {ssa}")
        set_one_input(g, env[ssa], x)


def compare_arrays(name, ref, got, atol=0.02, p99_limit=0.02, cos_limit=0.999):
    ref = np.asarray(ref, dtype=np.float32)
    got = np.asarray(got, dtype=np.float32)

    print(f"\nCOMPARE {name}")
    print("  ref:", ref.shape, ref.dtype)
    print("  got:", got.shape, got.dtype)

    finite = np.isfinite(ref) & np.isfinite(got)
    diff = np.abs(ref - got)

    denom = np.linalg.norm(ref[finite]) * np.linalg.norm(got[finite])
    cos = float(np.dot(ref[finite], got[finite]) / denom) if denom else float("nan")

    max_diff = float(np.max(diff[finite]))
    mean_diff = float(np.mean(diff[finite]))
    p99 = float(np.percentile(diff[finite], 99))

    print("  max diff:", max_diff)
    print("  mean diff:", mean_diff)
    print("  p99 diff:", p99)
    print("  cosine:", cos)

    ok = max_diff < atol or (p99 < p99_limit and cos > cos_limit)
    print("  OK:", ok)
    return ok


def summarize_bench(name, times_ms):
    times = np.asarray(times_ms, dtype=np.float64)
    print(f"\n{name}:")
    print(f"  runs:   {len(times)}")
    print(f"  mean:   {times.mean():.4f} ms")
    print(f"  median: {np.median(times):.4f} ms")
    print(f"  p95:    {np.percentile(times, 95):.4f} ms")
    print(f"  min:    {times.min():.4f} ms")
    print(f"  max:    {times.max():.4f} ms")


def bench(name, fn, warmup=5, iters=30):
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
    ap.add_argument("--prefill-len", type=int, default=5)
    ap.add_argument("--decode-steps", type=int, default=8)
    ap.add_argument("--bench-iters", type=int, default=30)
    ap.add_argument("--save-prefix", default="small_qwen2_kvcache")
    args = ap.parse_args()

    print("KV-cache Small-Qwen2 test")
    print("model:", args.model_id)
    print_mem("start")

    tok = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(args.model_id)
    model.eval()
    cfg = model.config

    print("config:")
    print("  vocab_size:", cfg.vocab_size)
    print("  hidden_size:", cfg.hidden_size)
    print("  layers:", cfg.num_hidden_layers)
    print("  heads:", cfg.num_attention_heads)
    print("  kv_heads:", getattr(cfg, "num_key_value_heads", cfg.num_attention_heads))

    input_ids_np, _ = make_inputs(tok, cfg, args.prompt, args.prefill_len)
    if input_ids_np.shape[1] > args.prefill_len:
        input_ids_np = input_ids_np[:, :args.prefill_len]

    position_ids_np = np.arange(input_ids_np.shape[1], dtype=np.int32)[None, :]

    print("\nprompt:", repr(args.prompt))
    print("prefill input_ids:", input_ids_np.tolist())
    print("decoded:", repr(tok.decode(input_ids_np[0].tolist())))

    print_mem("after HF load")

    params_np = extract_params(model)
    params_bytes = sum(np.asarray(p, dtype=np.float16).nbytes for p in params_np)
    print(f"params: {len(params_np)} tensors, {params_bytes/1024/1024:.2f} MiB fp16")

    print_mem("after extracting params")

    prefill_forward, decode_forward, full_forward = make_qwen2_prefill_and_decode(cfg)

    params_jax = tuple(jnp.asarray(x) for x in params_np)

    prefill_args_np = (input_ids_np, position_ids_np, *params_np)
    prefill_args_jax = (jnp.asarray(input_ids_np), jnp.asarray(position_ids_np), *params_jax)

    # ---------------- JAX prefill ----------------
    print("\nJAX prefill...")
    jax_prefill = jax.jit(prefill_forward)

    t0 = time.perf_counter()
    jax_logits, jax_k, jax_v = jax_prefill(*prefill_args_jax)
    jax.block_until_ready(jax_logits)
    jax.block_until_ready(jax_k)
    jax.block_until_ready(jax_v)
    jax_prefill_first_ms = (time.perf_counter() - t0) * 1000.0

    jax_logits_np = np.asarray(jax_logits)
    jax_k_np = np.asarray(jax_k)
    jax_v_np = np.asarray(jax_v)

    print(f"JAX prefill first: {jax_prefill_first_ms:.3f} ms")
    print("prefill logits:", jax_logits_np.shape, jax_logits_np.dtype)
    print("k cache:", jax_k_np.shape, jax_k_np.dtype)
    print("v cache:", jax_v_np.shape, jax_v_np.dtype)

    # Choose first decode token greedily from JAX prefill.
    next_id = int(np.argmax(jax_logits_np[0, -1]))
    next_token_np = np.array([[next_id]], dtype=np.int32)
    next_pos_np = np.array([[input_ids_np.shape[1]]], dtype=np.int32)

    print("first next_id:", next_id, repr(tok.decode([next_id])))

    decode_args_np = (next_token_np, next_pos_np, jax_k_np, jax_v_np, *params_np)
    decode_args_jax = (
        jnp.asarray(next_token_np),
        jnp.asarray(next_pos_np),
        jnp.asarray(jax_k_np),
        jnp.asarray(jax_v_np),
        *params_jax,
    )

    # ---------------- JAX decode ----------------
    print("\nJAX decode one step...")
    jax_decode = jax.jit(decode_forward)

    t0 = time.perf_counter()
    jax_dec_logits, jax_k2, jax_v2 = jax_decode(*decode_args_jax)
    jax.block_until_ready(jax_dec_logits)
    jax.block_until_ready(jax_k2)
    jax.block_until_ready(jax_v2)
    jax_decode_first_ms = (time.perf_counter() - t0) * 1000.0

    jax_dec_logits_np = np.asarray(jax_dec_logits)
    jax_k2_np = np.asarray(jax_k2)
    jax_v2_np = np.asarray(jax_v2)

    print(f"JAX decode first: {jax_decode_first_ms:.3f} ms")
    print("decode logits:", jax_dec_logits_np.shape, jax_dec_logits_np.dtype)
    print("new k cache:", jax_k2_np.shape, jax_k2_np.dtype)

    # ---------------- Full recompute reference ----------------
    full_ids_np = np.concatenate((input_ids_np, next_token_np), axis=1)
    full_pos_np = np.arange(full_ids_np.shape[1], dtype=np.int32)[None, :]

    jax_full = jax.jit(full_forward)
    full_logits = jax_full(jnp.asarray(full_ids_np), jnp.asarray(full_pos_np), *params_jax)
    full_logits_np = np.asarray(full_logits)

    compare_arrays(
        "JAX cached decode vs JAX full recompute last token",
        full_logits_np[:, -1:, :],
        jax_dec_logits_np,
        atol=0.04,
        p99_limit=0.03,
        cos_limit=0.998,
    )

    print_mem("after JAX refs")

    # ---------------- Export StableHLO ----------------
    print("\nExporting prefill StableHLO...")
    prefill_mlir = jax_to_mlir(prefill_forward, prefill_args_jax)
    prefill_path = f"{args.save_prefix}_prefill.stablehlo.mlir"
    Path(prefill_path).write_text(prefill_mlir)

    prefill_ir = parse_mlir(prefill_mlir)
    print("prefill IR inputs:", len(prefill_ir.inputs), "outputs:", prefill_ir.outputs, "nodes:", len(prefill_ir.order))

    print("\nExporting decode StableHLO...")
    decode_mlir = jax_to_mlir(decode_forward, decode_args_jax)
    decode_path = f"{args.save_prefix}_decode.stablehlo.mlir"
    Path(decode_path).write_text(decode_mlir)

    decode_ir = parse_mlir(decode_mlir)
    print("decode IR inputs:", len(decode_ir.inputs), "outputs:", decode_ir.outputs, "nodes:", len(decode_ir.order))

    print_mem("after MLIR export")

    # ---------------- Cactus prefill ----------------
    print("\nBuilding Cactus prefill graph...")
    t0 = time.perf_counter()
    g_prefill, env_prefill = lower_to_cactus(
        prefill_ir,
        patterns=[],
        verbose=False,
        strict_math=False,
    )
    prefill_build_ms = (time.perf_counter() - t0) * 1000.0
    print(f"prefill build: {prefill_build_ms:.3f} ms")

    set_args(g_prefill, env_prefill, prefill_args_np)
    set_constants(g_prefill, env_prefill, prefill_ir)

    t0 = time.perf_counter()
    g_prefill.execute()
    cactus_prefill_ms = (time.perf_counter() - t0) * 1000.0

    cactus_logits = env_prefill[prefill_ir.outputs[0]].numpy()
    cactus_k = env_prefill[prefill_ir.outputs[1]].numpy()
    cactus_v = env_prefill[prefill_ir.outputs[2]].numpy()

    print(f"Cactus prefill first execute: {cactus_prefill_ms:.3f} ms")
    print("Cactus prefill logits:", cactus_logits.shape)
    print("Cactus k cache:", cactus_k.shape)

    compare_arrays("Cactus prefill logits vs JAX", jax_logits_np, cactus_logits)
    compare_arrays("Cactus prefill k cache vs JAX", jax_k_np, cactus_k, atol=0.04, p99_limit=0.03, cos_limit=0.998)
    compare_arrays("Cactus prefill v cache vs JAX", jax_v_np, cactus_v, atol=0.04, p99_limit=0.03, cos_limit=0.998)

    print_mem("after Cactus prefill")

    # ---------------- Cactus decode ----------------
    print("\nBuilding Cactus decode graph...")

    cactus_next_id = int(np.argmax(cactus_logits[0, -1]))
    cactus_next_np = np.array([[cactus_next_id]], dtype=np.int32)
    cactus_pos_np = np.array([[input_ids_np.shape[1]]], dtype=np.int32)

    cactus_decode_args_np = (cactus_next_np, cactus_pos_np, cactus_k, cactus_v, *params_np)

    t0 = time.perf_counter()
    g_decode, env_decode = lower_to_cactus(
        decode_ir,
        patterns=["default"],
        verbose=False,
        strict_math=False,
    )
    decode_build_ms = (time.perf_counter() - t0) * 1000.0
    print(f"decode build: {decode_build_ms:.3f} ms")

    set_args(g_decode, env_decode, cactus_decode_args_np)
    set_constants(g_decode, env_decode, decode_ir)

    t0 = time.perf_counter()
    g_decode.execute()
    cactus_decode_ms = (time.perf_counter() - t0) * 1000.0

    cactus_dec_logits = env_decode[decode_ir.outputs[0]].numpy()
    cactus_k2 = env_decode[decode_ir.outputs[1]].numpy()
    cactus_v2 = env_decode[decode_ir.outputs[2]].numpy()

    print(f"Cactus decode first execute: {cactus_decode_ms:.3f} ms")
    print("Cactus decode logits:", cactus_dec_logits.shape)
    print("Cactus new k cache:", cactus_k2.shape)

    compare_arrays("Cactus decode logits vs JAX", jax_dec_logits_np, cactus_dec_logits)
    compare_arrays("Cactus decode k cache vs JAX", jax_k2_np, cactus_k2, atol=0.04, p99_limit=0.03, cos_limit=0.998)
    compare_arrays("Cactus decode v cache vs JAX", jax_v2_np, cactus_v2, atol=0.04, p99_limit=0.03, cos_limit=0.998)

    print_mem("after Cactus decode")

    # ---------------- Benchmarks ----------------
    print("\nBenchmarking warmed prefill/decode...")

    bench("JAX prefill warmed", lambda: jax_prefill(*prefill_args_jax)[0].block_until_ready(), iters=args.bench_iters)
    bench("JAX decode warmed", lambda: jax_decode(*decode_args_jax)[0].block_until_ready(), iters=args.bench_iters)

    bench("Cactus prefill warmed", lambda: g_prefill.execute(), iters=args.bench_iters)
    bench("Cactus decode warmed", lambda: g_decode.execute(), iters=args.bench_iters)

    # ---------------- Multi-step Cactus decode loop ----------------
    print("\nMulti-step greedy Cactus decode loop...")

    cur_ids = input_ids_np.copy()
    cur_k = cactus_k
    cur_v = cactus_v
    generated = []

    decode_times = []

    for step in range(args.decode_steps):
        if step == 0:
            tok_id = int(np.argmax(cactus_logits[0, -1]))
        else:
            tok_id = int(np.argmax(step_logits[0, -1]))

        generated.append(tok_id)

        one_tok = np.array([[tok_id]], dtype=np.int32)
        one_pos = np.array([[cur_ids.shape[1]]], dtype=np.int32)

        set_args(g_decode, env_decode, (one_tok, one_pos, cur_k, cur_v, *params_np))
        set_constants(g_decode, env_decode, decode_ir)

        t0 = time.perf_counter()
        g_decode.execute()
        dt = (time.perf_counter() - t0) * 1000.0
        decode_times.append(dt)

        step_logits = env_decode[decode_ir.outputs[0]].numpy()
        cur_k = env_decode[decode_ir.outputs[1]].numpy()
        cur_v = env_decode[decode_ir.outputs[2]].numpy()
        cur_ids = np.concatenate((cur_ids, one_tok), axis=1)

        print(f"  step {step:02d}: token={tok_id:<8} piece={repr(tok.decode([tok_id])):<20} time={dt:.3f} ms cache_len={cur_k.shape[3]}")

    avg_decode_ms = float(np.mean(decode_times))
    decode_tps = 1000.0 / avg_decode_ms

    print("\nGenerated ids:", generated)
    print("Generated text:", repr(tok.decode(input_ids_np[0].tolist() + generated)))
    print(f"Average decode: {avg_decode_ms:.3f} ms/token")
    print(f"Decode throughput: {decode_tps:.2f} tok/s")
    print_mem("after multi-step decode")

    print("\nSUMMARY")
    print(f"prefill length:          {input_ids_np.shape[1]} tokens")
    print(f"Cactus prefill first:    {cactus_prefill_ms:.3f} ms")
    print(f"Cactus decode first:     {cactus_decode_ms:.3f} ms/token")
    print(f"decode avg loop:         {avg_decode_ms:.3f} ms/token")
    print(f"decode TPS:              {decode_tps:.2f} tok/s")
    print(f"prefill build:           {prefill_build_ms:.3f} ms")
    print(f"decode build:            {decode_build_ms:.3f} ms")
    print("\nNote:")
    print("  This uses cache concat each decode step, so it is correct but not yet optimal.")
    print("  A production KV cache should preallocate [max_seq] cache and write in-place.")
    print("  Still, this proves the model math/cache wiring before C++ cache optimization.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\nFAILED")
        print(type(e).__name__ + ":", str(e))
        traceback.print_exc(limit=40)
        raise
