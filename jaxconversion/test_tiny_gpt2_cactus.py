#!/usr/bin/env python3
import argparse
import os
import time
import traceback
from collections import Counter

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


DEFAULT_MODEL_ID = "trl-internal-testing/tiny-random-LlamaForCausalLM"


def npy(x, dtype=np.float16):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=dtype)


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


def apply_llama_rope(q, k, position_ids, rope_theta):
    """
    q: [B,H,T,DH]
    k: [B,KVH,T,DH]
    position_ids: [B,T]

    Mirrors HF Llama rotary:
      inv_freq = 1 / theta ** (arange(0, dim, 2) / dim)
      emb = concat(freqs, freqs)
      q = q*cos + rotate_half(q)*sin
    """
    head_dim = q.shape[-1]
    half = head_dim // 2

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
    # [B,T,H*DH] -> [B,H,T,DH]
    b, t, _ = x.shape
    x = jnp.reshape(x, (b, t, n_heads, head_dim))
    return jnp.transpose(x, (0, 2, 1, 3))


def merge_heads(x):
    # [B,H,T,DH] -> [B,T,H*DH]
    x = jnp.transpose(x, (0, 2, 1, 3))
    b, t, h, d = x.shape
    return jnp.reshape(x, (b, t, h * d))


def repeat_kv(x, n_heads, n_kv_heads):
    # [B,KVH,T,DH] -> [B,H,T,DH]
    reps = n_heads // n_kv_heads
    pieces = []
    for kv_i in range(n_kv_heads):
        for _ in range(reps):
            pieces.append(x[:, kv_i:kv_i + 1, :, :])
    return jnp.concatenate(tuple(pieces), axis=1)


def make_llama_forward(cfg):
    n_layer = int(cfg.num_hidden_layers)
    hidden = int(cfg.hidden_size)
    n_heads = int(cfg.num_attention_heads)
    n_kv_heads = int(getattr(cfg, "num_key_value_heads", n_heads))
    head_dim = int(getattr(cfg, "head_dim", hidden // n_heads))
    intermediate = int(cfg.intermediate_size)
    eps = float(getattr(cfg, "rms_norm_eps", 1e-6))
    rope_theta = float(getattr(cfg, "rope_theta", 10000.0))

    def tiny_llama_forward(input_ids, position_ids, *params):
        """
        params:
          0 embed_tokens.weight [vocab, hidden]
          1 final_norm.weight   [hidden]
          2 lm_head.weight      [vocab, hidden]

          each layer:
            input_norm_w
            q_w, q_b
            k_w, k_b
            v_w, v_b
            o_w, o_b
            post_norm_w
            gate_w, gate_b
            up_w, up_b
            down_w, down_b
        """
        embed_w = params[0]
        final_norm_w = params[1]
        lm_head_w = params[2]

        x = embed_w[input_ids]
        x = x.astype(jnp.float16)

        p = 3

        for _ in range(n_layer):
            input_norm_w = params[p + 0]
            q_w = params[p + 1]
            q_b = params[p + 2]
            k_w = params[p + 3]
            k_b = params[p + 4]
            v_w = params[p + 5]
            v_b = params[p + 6]
            o_w = params[p + 7]
            o_b = params[p + 8]
            post_norm_w = params[p + 9]
            gate_w = params[p + 10]
            gate_b = params[p + 11]
            up_w = params[p + 12]
            up_b = params[p + 13]
            down_w = params[p + 14]
            down_b = params[p + 15]
            p += 16

            # Attention.
            residual = x
            h = rms_norm(x, input_norm_w, eps)

            q = jnp.einsum("btd,od->bto", h, q_w) + q_b
            k = jnp.einsum("btd,od->bto", h, k_w) + k_b
            v = jnp.einsum("btd,od->bto", h, v_w) + v_b

            q = split_heads(q, n_heads, head_dim)
            k = split_heads(k, n_kv_heads, head_dim)
            v = split_heads(v, n_kv_heads, head_dim)

            q, k = apply_llama_rope(q, k, position_ids, rope_theta)

            k = repeat_kv(k, n_heads, n_kv_heads)
            v = repeat_kv(v, n_heads, n_kv_heads)

            scores = jnp.einsum("bhtd,bhsd->bhts", q, k)
            scores = scores * jnp.asarray(1.0 / np.sqrt(head_dim), dtype=scores.dtype)

            t = input_ids.shape[1]
            causal = jnp.tril(jnp.ones((t, t), dtype=bool))
            scores = jnp.where(
                causal[None, None, :, :],
                scores,
                jnp.asarray(-10000.0, dtype=scores.dtype),
            )

            probs = jax.nn.softmax(scores, axis=-1)
            ctx = jnp.einsum("bhts,bhsd->bhtd", probs, v)
            ctx = merge_heads(ctx)

            attn_out = jnp.einsum("btd,od->bto", ctx, o_w) + o_b
            x = residual + attn_out

            # MLP.
            residual = x
            h = rms_norm(x, post_norm_w, eps)

            gate = jnp.einsum("btd,od->bto", h, gate_w) + gate_b
            up = jnp.einsum("btd,od->bto", h, up_w) + up_b
            hidden_mlp = silu(gate) * up
            down = jnp.einsum("btd,od->bto", hidden_mlp, down_w) + down_b

            x = residual + down

        x = rms_norm(x, final_norm_w, eps)

        logits = jnp.einsum("btd,vd->btv", x, lm_head_w)
        return logits.astype(jnp.float16)

    return tiny_llama_forward


def get_or_zero(sd, key, shape):
    if key in sd:
        return npy(sd[key])
    return np.zeros(shape, dtype=np.float16)


def extract_params(model):
    sd = model.state_dict()
    cfg = model.config

    params = [
        npy(sd["model.embed_tokens.weight"]),
        npy(sd["model.norm.weight"]),
        npy(sd["lm_head.weight"]),
    ]

    for i in range(int(cfg.num_hidden_layers)):
        prefix = f"model.layers.{i}"
        hidden = int(cfg.hidden_size)
        n_heads = int(cfg.num_attention_heads)
        n_kv_heads = int(getattr(cfg, "num_key_value_heads", n_heads))
        head_dim = int(getattr(cfg, "head_dim", hidden // n_heads))
        intermediate = int(cfg.intermediate_size)

        q_out = n_heads * head_dim
        kv_out = n_kv_heads * head_dim

        params.extend(
            [
                npy(sd[f"{prefix}.input_layernorm.weight"]),

                npy(sd[f"{prefix}.self_attn.q_proj.weight"]),
                get_or_zero(sd, f"{prefix}.self_attn.q_proj.bias", (q_out,)),

                npy(sd[f"{prefix}.self_attn.k_proj.weight"]),
                get_or_zero(sd, f"{prefix}.self_attn.k_proj.bias", (kv_out,)),

                npy(sd[f"{prefix}.self_attn.v_proj.weight"]),
                get_or_zero(sd, f"{prefix}.self_attn.v_proj.bias", (kv_out,)),

                npy(sd[f"{prefix}.self_attn.o_proj.weight"]),
                get_or_zero(sd, f"{prefix}.self_attn.o_proj.bias", (hidden,)),

                npy(sd[f"{prefix}.post_attention_layernorm.weight"]),

                npy(sd[f"{prefix}.mlp.gate_proj.weight"]),
                get_or_zero(sd, f"{prefix}.mlp.gate_proj.bias", (intermediate,)),

                npy(sd[f"{prefix}.mlp.up_proj.weight"]),
                get_or_zero(sd, f"{prefix}.mlp.up_proj.bias", (intermediate,)),

                npy(sd[f"{prefix}.mlp.down_proj.weight"]),
                get_or_zero(sd, f"{prefix}.mlp.down_proj.bias", (hidden,)),
            ]
        )

    return params


def op_counts(ir):
    c = Counter()
    for nid in ir.order:
        c[ir.nodes[nid].op] += 1
    return c


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


def summarize_bench(name, times_ms):
    times = np.asarray(times_ms, dtype=np.float64)
    print(f"\n{name}:")
    print(f"  runs:   {len(times)}")
    print(f"  mean:   {times.mean():.4f} ms")
    print(f"  median: {np.median(times):.4f} ms")
    print(f"  p95:    {np.percentile(times, 95):.4f} ms")
    print(f"  min:    {times.min():.4f} ms")
    print(f"  max:    {times.max():.4f} ms")


def bench_ms(name, fn, warmup=10, iters=100, block=None):
    for _ in range(warmup):
        y = fn()
        if block is not None:
            block(y)

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        y = fn()
        if block is not None:
            block(y)
        times.append((time.perf_counter() - t0) * 1000.0)

    summarize_bench(name, times)
    return times


def compare(name, ref, got):
    ref = np.asarray(ref, dtype=np.float32)
    got = np.asarray(got, dtype=np.float32)

    print(f"\n================ {name} ================")
    print("ref shape:", ref.shape)
    print("got shape:", got.shape)
    print("ref finite:", bool(np.isfinite(ref).all()))
    print("got finite:", bool(np.isfinite(got).all()))

    if ref.shape != got.shape:
        print("SHAPE MISMATCH")
        return False

    finite = np.isfinite(ref) & np.isfinite(got)
    diff = np.abs(ref - got)

    max_diff = float(np.max(diff[finite]))
    mean_diff = float(np.mean(diff[finite]))
    p95 = float(np.percentile(diff[finite], 95))
    p99 = float(np.percentile(diff[finite], 99))

    denom = np.linalg.norm(ref[finite]) * np.linalg.norm(got[finite])
    cos = float(np.dot(ref[finite], got[finite]) / denom) if denom else float("nan")

    print("max diff:", max_diff)
    print("mean diff:", mean_diff)
    print("p95 diff:", p95)
    print("p99 diff:", p99)
    print("cosine:", cos)

    last_ref = ref[0, -1]
    last_got = got[0, -1]

    print("last ref first16:", last_ref[:16])
    print("last got first16:", last_got[:16])

    ref_top = np.argsort(-last_ref)[:10]
    got_top = np.argsort(-last_got)[:10]

    print("\nref top10:")
    for i in ref_top:
        print(f"  {int(i):<8} {float(last_ref[i]):>10.5f}")

    print("\ngot top10:")
    for i in got_top:
        print(f"  {int(i):<8} {float(last_got[i]):>10.5f}")

    overlap = set(ref_top.tolist()) & set(got_top.tolist())
    print("top10 overlap:", len(overlap), "/ 10", sorted(int(x) for x in overlap))

    ok = (
        np.isfinite(ref).all()
        and np.isfinite(got).all()
        and cos > 0.998
        and mean_diff < 0.08
        and p99 < 0.50
    )

    print("OK:", ok)
    return ok


def make_inputs(tok, cfg, prompt, max_length):
    try:
        enc = tok(prompt, return_tensors="pt")
        input_ids_np = enc["input_ids"].detach().cpu().numpy().astype(np.int32)
    except Exception:
        vocab = int(cfg.vocab_size)
        raw = [1, 150, 160, 170, 180, 190]
        input_ids_np = np.array([[x % vocab for x in raw]], dtype=np.int32)

    if input_ids_np.shape[1] > max_length:
        input_ids_np = input_ids_np[:, :max_length]

    position_ids_np = np.arange(input_ids_np.shape[1], dtype=np.int32)[None, :]
    return input_ids_np, position_ids_np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--max-length", type=int, default=16)
    ap.add_argument("--save-prefix", default="tiny_llama")
    ap.add_argument("--bench-iters", type=int, default=100)
    args = ap.parse_args()

    print("Loading:", args.model_id)

    tok = None
    try:
        tok = AutoTokenizer.from_pretrained(args.model_id)
    except Exception as e:
        print("Tokenizer load failed; using raw synthetic token ids.")
        print("Tokenizer error:", repr(e))

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
    print("  head_dim:", getattr(cfg, "head_dim", int(cfg.hidden_size) // int(cfg.num_attention_heads)))
    print("  rms_norm_eps:", getattr(cfg, "rms_norm_eps", None))
    print("  rope_theta:", getattr(cfg, "rope_theta", None))

    input_ids_np, position_ids_np = make_inputs(tok, cfg, args.prompt, args.max_length)
    input_ids_pt = torch.tensor(input_ids_np, dtype=torch.long)
    position_ids_pt = torch.tensor(position_ids_np, dtype=torch.long)

    print("\nprompt:", repr(args.prompt))
    print("input_ids:", input_ids_np.tolist())
    if tok is not None:
        try:
            print("decoded:", repr(tok.decode(input_ids_np[0].tolist())))
        except Exception:
            pass

    print("\nRunning official PyTorch model...")
    with torch.no_grad():
        t0 = time.perf_counter()
        torch_logits_t = model(input_ids=input_ids_pt, position_ids=position_ids_pt).logits
        torch_logits = torch_logits_t.detach().cpu().numpy().astype(np.float32)
        torch_first_ms = (time.perf_counter() - t0) * 1000.0

    print(f"PyTorch first run: {torch_first_ms:.2f}ms")
    print("torch logits:", torch_logits.shape, torch_logits.dtype)
    print("torch last argmax:", int(np.argmax(torch_logits[0, -1])))

    print("\nExtracting weights...")
    params_np = extract_params(model)
    print("num params tensors:", len(params_np))
    for i, p in enumerate(params_np[:12]):
        print(f"  param {i}: shape={p.shape} dtype={p.dtype}")

    forward = make_llama_forward(cfg)

    all_args_np = (input_ids_np, position_ids_np, *params_np)
    all_args_jax = tuple(jnp.asarray(x) for x in all_args_np)

    print("\nRunning manual JAX tiny Llama...")
    jitted = jax.jit(forward)

    t0 = time.perf_counter()
    jax_logits = jitted(*all_args_jax)
    jax.block_until_ready(jax_logits)
    jax_first_ms = (time.perf_counter() - t0) * 1000.0

    jax_logits_np = np.asarray(jax_logits)

    print(f"JAX first run: {jax_first_ms:.2f}ms")
    print("JAX logits:", jax_logits_np.shape, jax_logits_np.dtype)

    print("\nBenchmarking warmed PyTorch/JAX...")

    bench_ms(
        "PyTorch warmed execute",
        lambda: model(input_ids=input_ids_pt, position_ids=position_ids_pt).logits,
        warmup=10,
        iters=args.bench_iters,
        block=lambda y: None,
    )

    bench_ms(
        "JAX warmed execute",
        lambda: jitted(*all_args_jax),
        warmup=10,
        iters=args.bench_iters,
        block=lambda y: y.block_until_ready(),
    )

    ok_jax_torch = compare("JAX vs official PyTorch", torch_logits, jax_logits_np)

    print("\nExporting StableHLO...")
    mlir = jax_to_mlir(forward, all_args_jax)

    mlir_path = f"{args.save_prefix}.stablehlo.mlir"
    with open(mlir_path, "w") as f:
        f.write(mlir)
    print("saved:", mlir_path)

    ir = parse_mlir(mlir)

    print("\nParsed IR:")
    print("  inputs:", len(ir.inputs))
    print("  outputs:", ir.outputs)
    print("  nodes:", len(ir.order))
    print("  constants:", len(ir.constants))

    counts = op_counts(ir)
    print("\nOps:")
    for op, n in sorted(counts.items()):
        print(f"  {op:<36} {n}")

    print("\nBuilding Cactus graph...")
    t0 = time.perf_counter()
    g, env = lower_to_cactus(
        ir,
        patterns=["default"],
        verbose=False,
        strict_math=False,
    )
    build_ms = (time.perf_counter() - t0) * 1000.0
    print(f"Build: {build_ms:.2f}ms")

    set_inputs_and_constants(g, env, ir, all_args_np)

    print("Executing Cactus...")
    t0 = time.perf_counter()
    g.execute()
    exec_ms = (time.perf_counter() - t0) * 1000.0
    print(f"Execute: {exec_ms:.2f}ms")

    print("\nBenchmarking warmed Cactus...")

    bench_ms(
        "Cactus warmed execute only",
        lambda: g.execute(),
        warmup=10,
        iters=args.bench_iters,
    )

    bench_ms(
        "Cactus warmed execute + numpy readback",
        lambda: (g.execute(), env[ir.outputs[0]].numpy())[1],
        warmup=10,
        iters=args.bench_iters,
    )

    cactus_logits = env[ir.outputs[0]].numpy()

    np.save(f"{args.save_prefix}_torch_logits.npy", torch_logits)
    np.save(f"{args.save_prefix}_jax_logits.npy", jax_logits_np)
    np.save(f"{args.save_prefix}_cactus_logits.npy", cactus_logits)
    np.save(f"{args.save_prefix}_input_ids.npy", input_ids_np)
    np.save(f"{args.save_prefix}_position_ids.npy", position_ids_np)

    print("Saved npys with prefix:", args.save_prefix)

    ok_cactus_jax = compare("Cactus vs manual JAX", jax_logits_np, cactus_logits)
    ok_cactus_torch = compare("Cactus vs official PyTorch", torch_logits, cactus_logits)

    last = cactus_logits[0, -1].astype(np.float32)
    next_id = int(np.argmax(last))

    print("\nGreedy next token from Cactus:")
    print("  id:", next_id)

    if tok is not None:
        try:
            print("  piece:", repr(tok.decode([next_id])))
            print("  text:", repr(tok.decode(input_ids_np[0].tolist() + [next_id])))
        except Exception:
            pass

    print("\n================ FINAL ================")
    print("JAX/PyTorch close:", ok_jax_torch)
    print("JAX/Cactus matched:", ok_cactus_jax)
    print("PyTorch/Cactus close:", ok_cactus_torch)

    if ok_cactus_jax:
        print("TINY-LLAMA CACTUS TEST PASSED ✅")
        print("This proves a RoPE/RMSNorm/Llama-style external model can run through JAX -> StableHLO -> Cactus.")
    else:
        print("TINY-LLAMA CACTUS TEST FAILED ❌")
        print("Use saved MLIR/NPYs to audit the first mismatch.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\nFAILED")
        print(type(e).__name__ + ":", str(e))
        traceback.print_exc(limit=30)
        raise