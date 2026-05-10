#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from safetensors import safe_open

from convert_any import _compare_arrays, _execute_graph
from lower_to_cactus import lower_to_cactus
from parse import parse_mlir
from test import jax_to_mlir


def build_gemmaish_fn(seq: int):
    B = 1
    T = int(seq)
    VOCAB = 4096
    D = 2304
    H = 8
    KVH = 4
    DH = 256
    FF = 9216
    EPS = 1e-6

    def silu(x):
        return x * jax.nn.sigmoid(x)

    def rms_norm(x, w):
        x32 = x.astype(jnp.float32)
        ms = jnp.mean(jnp.square(x32), axis=-1, keepdims=True)
        y = x32 * jax.lax.rsqrt(ms + jnp.asarray(EPS, dtype=jnp.float32))
        return y.astype(x.dtype) * w.astype(x.dtype)

    def head_rms_norm(x, w):
        # x:[B,H,T,DH], w:[H,DH] or [KVH,DH]
        x32 = x.astype(jnp.float32)
        ms = jnp.mean(jnp.square(x32), axis=-1, keepdims=True)
        y = x32 * jax.lax.rsqrt(ms + jnp.asarray(EPS, dtype=jnp.float32))
        return y.astype(x.dtype) * w[None, :, None, :].astype(x.dtype)

    def fn(
        tokens, positions, attention_mask, token_embeddings, output_norm,
        attn_output, attn_kv, attn_q, ffn_gate_up, ffn_down,
        input_norm, post_attn_norm, pre_ffn_norm, post_ffn_norm, q_norm, k_norm
    ):
        x = token_embeddings[tokens].astype(jnp.float16) * jnp.asarray(np.sqrt(D), dtype=jnp.float16)
        token_valid = tokens != jnp.asarray(0, dtype=tokens.dtype)
        x = jnp.where(token_valid[:, :, None], x, jnp.asarray(0.0, dtype=x.dtype))

        xn = rms_norm(x, input_norm)
        q = jnp.einsum("btd,hdf->bthf", xn, attn_q)
        q = jnp.transpose(q, (0, 2, 1, 3))
        kv = jnp.einsum("btd,zkdf->bztkf", xn, attn_kv)
        k = jnp.transpose(kv[:, 0, :, :, :], (0, 2, 1, 3))
        v = jnp.transpose(kv[:, 1, :, :, :], (0, 2, 1, 3))

        q = head_rms_norm(q, q_norm)
        k = head_rms_norm(k, k_norm)

        reps = H // KVH
        k = jnp.concatenate([k[:, i : i + 1, :, :] for i in range(KVH) for _ in range(reps)], axis=1)
        v = jnp.concatenate([v[:, i : i + 1, :, :] for i in range(KVH) for _ in range(reps)], axis=1)

        scores = jnp.einsum("bhtd,bhsd->bhts", q, k) * jnp.asarray(1.0 / np.sqrt(DH), dtype=q.dtype)
        pos = positions.astype(jnp.float16)
        am = attention_mask.astype(jnp.float16)
        causal = pos[:, :, None] >= pos[:, None, :]
        runtime_allowed = am > jnp.asarray(0.0, dtype=am.dtype)
        mask = causal & runtime_allowed & token_valid[:, :, None] & token_valid[:, None, :]
        scores = jnp.where(mask[:, None, :, :], scores, jnp.asarray(-10000.0, dtype=scores.dtype))
        probs = jax.nn.softmax(scores, axis=-1)

        ctx = jnp.einsum("bhts,bhsd->bhtd", probs, v)
        ctx = jnp.transpose(ctx, (0, 2, 1, 3))
        attn = jnp.einsum("bthd,hdm->btm", ctx, attn_output)

        x = x + attn
        x = x + jnp.asarray(0.03125, dtype=x.dtype) * rms_norm(x, post_attn_norm)
        xf = rms_norm(x, pre_ffn_norm)
        gate_up = jnp.einsum("btd,zdf->bztf", xf, ffn_gate_up)
        hidden = silu(gate_up[:, 0, :, :]) * gate_up[:, 1, :, :]
        down = jnp.einsum("btf,fd->btd", hidden, ffn_down)
        x = x + down
        x = x + jnp.asarray(0.03125, dtype=x.dtype) * rms_norm(x, post_ffn_norm)
        y = rms_norm(x, output_norm)
        logits = jnp.einsum("btd,vd->btv", y, token_embeddings)
        return logits.astype(jnp.float16)

    def make_random_args():
        rng = np.random.default_rng(12345)

        def f16(x):
            return np.asarray(x, dtype=np.float16)

        def rn(shape, s=1.0):
            return f16(rng.standard_normal(shape).astype(np.float32) * s)

        def one(shape, s=0.02):
            return f16(1.0 + rng.standard_normal(shape).astype(np.float32) * s)

        base = np.array([2, 651, 603, 576, 608, 283, 0, 0], dtype=np.int32)
        if T <= base.shape[0]:
            tok = base[:T]
        else:
            pad = np.zeros((T - base.shape[0],), dtype=np.int32)
            tok = np.concatenate([base, pad], axis=0)
        tokens = tok[None, :]
        positions = np.array([list(range(11, 11 + T))], dtype=np.float16)
        attention_mask = np.zeros((B, T, T), dtype=np.float16)
        for q in range(min(6, T)):
            for kk in range(min(6, T)):
                if kk <= q:
                    attention_mask[0, q, kk] = 1.0

        return (
            tokens,
            positions,
            attention_mask,
            rn((VOCAB, D), 0.024),
            one((D,), 0.02),
            rn((H, DH, D), 0.012),
            rn((2, KVH, D, DH), 0.014),
            rn((H, D, DH), 0.014),
            rn((2, D, FF), 0.010),
            rn((FF, D), 0.010),
            one((D,), 0.02),
            one((D,), 0.02),
            one((D,), 0.02),
            one((D,), 0.02),
            one((H, DH), 0.02),
            one((KVH, DH), 0.02),
        )

    return fn, make_random_args


def _load_real_layer0_args(seq: int, hf_dir: Path, prompt: str | None = None):
    cfg = json.loads((hf_dir / "config.json").read_text())
    D = int(cfg["hidden_size"])
    FF = int(cfg["intermediate_size"])
    H = int(cfg["num_attention_heads"])
    KVH = int(cfg["num_key_value_heads"])
    VOCAB = int(cfg["vocab_size"])

    idx = json.loads((hf_dir / "model.safetensors.index.json").read_text())["weight_map"]
    files = {}

    def get_tensor(name: str):
        fn = idx[name]
        if fn not in files:
            files[fn] = safe_open(str(hf_dir / fn), framework="pt", device="cpu")
        return files[fn].get_tensor(name).float().numpy()

    token_embeddings_full = get_tensor("model.embed_tokens.weight")  # [V,D]
    output_norm = get_tensor("model.norm.weight")  # [D]

    q_w = get_tensor("model.layers.0.self_attn.q_proj.weight")  # [H*DH,D]
    k_w = get_tensor("model.layers.0.self_attn.k_proj.weight")  # [KVH*DH,D]
    v_w = get_tensor("model.layers.0.self_attn.v_proj.weight")  # [KVH*DH,D]
    o_w = get_tensor("model.layers.0.self_attn.o_proj.weight")  # [D,H*DH]

    gate_w = get_tensor("model.layers.0.mlp.gate_proj.weight")  # [FF,D]
    up_w = get_tensor("model.layers.0.mlp.up_proj.weight")  # [FF,D]
    down_w = get_tensor("model.layers.0.mlp.down_proj.weight")  # [D,FF]

    input_norm = get_tensor("model.layers.0.input_layernorm.weight")
    post_attn_norm = get_tensor("model.layers.0.post_attention_layernorm.weight")
    pre_ffn_norm = get_tensor("model.layers.0.pre_feedforward_layernorm.weight")
    post_ffn_norm = get_tensor("model.layers.0.post_feedforward_layernorm.weight")

    # Infer head dimensions from actual tensor shapes (more robust than config assumptions).
    q_out = int(q_w.shape[0])
    k_out = int(k_w.shape[0])
    v_out = int(v_w.shape[0])
    if q_out % H != 0:
        raise ValueError(f"q_proj out dim {q_out} not divisible by num_heads {H}")
    DH_q = q_out // H
    if k_out % KVH != 0 or v_out % KVH != 0:
        raise ValueError(f"k/v out dims {k_out}/{v_out} not divisible by num_kv_heads {KVH}")
    DH_k = k_out // KVH
    DH_v = v_out // KVH
    if DH_k != DH_q or DH_v != DH_q:
        raise ValueError(f"head dims mismatch q={DH_q}, k={DH_k}, v={DH_v}")

    # Map HF linear weight layout to gemmaish function layout.
    attn_q = q_w.reshape(H, DH_q, D).transpose(0, 2, 1)  # [H,D,DH]
    attn_k = k_w.reshape(KVH, DH_q, D).transpose(0, 2, 1)  # [KVH,D,DH]
    attn_v = v_w.reshape(KVH, DH_q, D).transpose(0, 2, 1)  # [KVH,D,DH]
    attn_kv = np.stack([attn_k, attn_v], axis=0)  # [2,KVH,D,DH]
    attn_output = o_w.reshape(D, H, DH_q).transpose(1, 2, 0)  # [H,DH,D]

    ffn_gate_up = np.stack([gate_w.T, up_w.T], axis=0)  # [2,D,FF]
    ffn_down = down_w.T  # [FF,D]

    # Gemma2 HF weights do not expose q_norm/k_norm tensors like this path expects.
    q_norm = np.ones((H, DH_q), dtype=np.float32)
    k_norm = np.ones((KVH, DH_q), dtype=np.float32)

    T = int(seq)
    if prompt:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(str(hf_dir), local_files_only=True)
        enc = tok(prompt, return_tensors="np", add_special_tokens=True)
        ids = enc["input_ids"][0].astype(np.int32)
        ids = ids[:T]
        if ids.shape[0] < T:
            pad_id = int(tok.pad_token_id if tok.pad_token_id is not None else 0)
            ids = np.concatenate([ids, np.full((T - ids.shape[0],), pad_id, dtype=np.int32)], axis=0)
        tokens = ids[None, :]
    else:
        base = np.array([2, 651, 603, 576, 608, 283, 0, 0], dtype=np.int32)
        tokv = base[:T] if T <= base.shape[0] else np.concatenate([base, np.zeros((T - base.shape[0],), np.int32)], axis=0)
        tokens = tokv[None, :]
    positions = np.array([list(range(11, 11 + T))], dtype=np.float32)
    attention_mask = np.zeros((1, T, T), dtype=np.float32)
    valid = int(np.count_nonzero(tokens[0] != 0))
    valid = valid if valid > 0 else T
    for q in range(valid):
        for kk in range(valid):
            if kk <= q:
                attention_mask[0, q, kk] = 1.0

    # Keep full vocab for true text quality.
    token_embeddings = token_embeddings_full

    return (
        tokens,
        positions,
        attention_mask,
        token_embeddings.astype(np.float16),
        output_norm.astype(np.float16),
        attn_output.astype(np.float16),
        attn_kv.astype(np.float16),
        attn_q.astype(np.float16),
        ffn_gate_up.astype(np.float16),
        ffn_down.astype(np.float16),
        input_norm.astype(np.float16),
        post_attn_norm.astype(np.float16),
        pre_ffn_norm.astype(np.float16),
        post_ffn_norm.astype(np.float16),
        q_norm.astype(np.float16),
        k_norm.astype(np.float16),
    )


def save_npz(path: Path, args_np):
    payload = {f"arg{i}": np.asarray(v) for i, v in enumerate(args_np)}
    np.savez(path, **payload)

def _greedy_generate_preview(fn, args_np, max_new_tokens: int = 8):
    args = [np.asarray(x).copy() for x in args_np]
    tokens = args[0].astype(np.int32)  # [1, T]
    positions = args[1].astype(np.float32)  # [1, T]
    T = tokens.shape[1]

    valid = int(np.count_nonzero(tokens[0] != 0))
    if valid <= 0:
        valid = 1

    for _ in range(max_new_tokens):
        if valid >= T:
            break

        attn = np.zeros((1, T, T), dtype=np.float32)
        for q in range(valid):
            for k in range(valid):
                if k <= q:
                    attn[0, q, k] = 1.0
        args[2] = attn.astype(args[2].dtype, copy=False)

        logits = np.asarray(
            jax.device_get(fn(*tuple(jnp.asarray(x) for x in args)))
        )  # [1, T, V]
        next_id = int(np.argmax(logits[0, valid - 1, :]))
        tokens[0, valid] = next_id
        valid += 1

        # Keep positions monotonic as we extend.
        positions[0, valid - 1] = positions[0, valid - 2] + 1.0

    args[0] = tokens
    args[1] = positions.astype(args[1].dtype, copy=False)
    return tokens[0].tolist()


def main() -> None:
    ap = argparse.ArgumentParser(description="Export Gemma-like StableHLO MLIR + argN NPZ bundle.")
    ap.add_argument("--out-prefix", default="gemma_bundle", help="Output prefix")
    ap.add_argument("--seq", type=int, default=8, help="Sequence length")
    ap.add_argument("--test", action="store_true", help="Run JAX vs Cactus compare after export.")
    ap.add_argument("--atol", type=float, default=1e-3, help="allclose atol for --test")
    ap.add_argument("--rtol", type=float, default=1e-3, help="allclose rtol for --test")
    ap.add_argument("--show-text", action="store_true", help="Print token prediction preview.")
    ap.add_argument("--tokenizer-model-id", default="", help="Optional HF tokenizer id for text decode.")
    ap.add_argument(
        "--real-hf-dir",
        default="",
        help="Optional local HF Gemma dir (with safetensors) to use real weights, e.g. ~/.jaxgarden/hf_models/google/gemma-2-2b-it",
    )
    ap.add_argument("--prompt", default="", help="Optional prompt text used to build input_ids for --real-hf-dir mode.")
    args = ap.parse_args()

    fn, make_random_args = build_gemmaish_fn(args.seq)
    if args.real_hf_dir:
        args_np = _load_real_layer0_args(
            args.seq,
            Path(args.real_hf_dir).expanduser(),
            prompt=(args.prompt or None),
        )
    else:
        args_np = make_random_args()
    args_jax = tuple(jnp.asarray(x) for x in args_np)

    mlir_text = jax_to_mlir(fn, args_jax)

    out_prefix = Path(args.out_prefix)
    mlir_path = out_prefix.with_suffix(".stablehlo.mlir")
    npz_path = out_prefix.with_suffix(".inputs_weights.npz")
    meta_path = out_prefix.with_suffix(".meta.json")

    mlir_path.write_text(mlir_text)
    save_npz(npz_path, args_np)
    meta = {
        "schema": "cactus.export.gemmaish.v1",
        "mlir": str(mlir_path),
        "npz": str(npz_path),
        "num_args": len(args_np),
        "seq": int(args.seq),
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")

    print(f"saved mlir: {mlir_path}")
    print(f"saved npz : {npz_path}")
    print(f"saved meta: {meta_path}")

    if args.test:
        ir = parse_mlir(mlir_text)
        g, env = lower_to_cactus(ir, patterns=["default"], verbose=False)
        cactus_out = _execute_graph(g, env, ir, args_np)[0]
        jax_out = np.asarray(jax.device_get(fn(*args_jax)))
        cmp = _compare_arrays(jax_out, cactus_out, atol=args.atol, rtol=args.rtol)
        test_path = out_prefix.with_suffix(".test.json")
        test_path.write_text(json.dumps(cmp, indent=2, sort_keys=True) + "\n")
        print(f"saved test: {test_path}")
        print("test:", json.dumps(cmp, sort_keys=True))
        if args.show_text:
            in_ids = np.asarray(args_np[0])[0].tolist()
            gen_ids = _greedy_generate_preview(fn, args_np, max_new_tokens=max(1, min(8, int(args.seq) - 1)))
            print("input token ids :", in_ids)
            print("gen token ids   :", gen_ids)

            if args.tokenizer_model_id:
                try:
                    from transformers import AutoTokenizer

                    tok = AutoTokenizer.from_pretrained(args.tokenizer_model_id)
                    print("input text      :", tok.decode(in_ids))
                    print("gen text        :", tok.decode(gen_ids))
                except Exception as e:
                    print(f"tokenizer decode failed: {e}")
                    print("Hint: run with internet access for HF tokenizer download, or pre-cache it.")
            else:
                print("No tokenizer set. Pass --tokenizer-model-id google/gemma-2-2b-it to decode text.")

    print("\nnext:")
    print(
        "python jaxconversion/convert_any.py "
        f"{mlir_path} --inputs-npz {npz_path} --run --graph-out {out_prefix}.cactus"
    )


if __name__ == "__main__":
    main()
