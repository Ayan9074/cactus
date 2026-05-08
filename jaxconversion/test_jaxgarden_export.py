#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

# Make sure local third_party checkout is importable even if pip -e failed.
ROOT = Path(__file__).resolve().parent
JAXGARDEN_ROOT = ROOT / "third_party" / "jaxgarden"
if JAXGARDEN_ROOT.exists():
    sys.path.insert(0, str(JAXGARDEN_ROOT))

from jaxgarden import (
    LlamaConfig,
    LlamaForCausalLM,
    Gemma2Config,
    Gemma2ForCausalLM,
    Gemma3Config,
    Gemma3ForCausalLM,
)

from test import jax_to_mlir


def summarize(name, x):
    arr = np.asarray(x)
    print(f"\n{name}:")
    print("  shape:", arr.shape)
    print("  dtype:", arr.dtype)
    if np.issubdtype(arr.dtype, np.number):
        print("  finite:", bool(np.isfinite(arr).all()))
        print("  min/max:", float(np.nanmin(arr)), float(np.nanmax(arr)))
        print("  mean/std:", float(np.nanmean(arr)), float(np.nanstd(arr)))
        print("  sample:", arr.reshape(-1)[:12])


def make_ids(batch=1, seq=8, vocab=320):
    ids = np.array([[2, 11, 277, 256, 1, 283, 44, 91]], dtype=np.int32)
    ids = ids[:, :seq]
    ids = np.mod(ids, vocab).astype(np.int32)
    if batch != 1:
        ids = np.broadcast_to(ids, (batch, seq)).copy()
    return jnp.asarray(ids)


def export_llama(seq=8, out_prefix="jaxgarden_llama_tiny"):
    print("\n================ JAXGARDEN LLAMA TINY ================")

    config = LlamaConfig(
        dim=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        head_dim=16,
        intermediate_size=128,
        vocab_size=320,
        norm_eps=1e-5,
        rope_theta=10000.0,
    )

    model = LlamaForCausalLM(
        config,
        dtype=jnp.float16,
        param_dtype=jnp.float16,
        rngs=nnx.Rngs(0),
    )

    input_ids = make_ids(seq=seq, vocab=config.vocab_size)

    causal = np.tril(np.ones((seq, seq), dtype=bool))
    attention_mask = jnp.asarray(causal)

    # Split NNX module into static graphdef + mutable state.
    graphdef, state = nnx.split(model)

    def forward(state, input_ids, attention_mask):
        m = nnx.merge(graphdef, state)
        return m(input_ids, attention_mask=attention_mask)

    logits = forward(state, input_ids, attention_mask)
    jax.block_until_ready(logits)

    summarize("llama logits", logits)
    print("argmax last:", int(np.argmax(np.asarray(logits)[0, -1])))

    print("exporting StableHLO...")
    mlir = jax_to_mlir(forward, (state, input_ids, attention_mask))
    path = f"{out_prefix}.stablehlo.mlir"
    Path(path).write_text(mlir)
    print("saved:", path)
    return path


def export_gemma2(seq=8, out_prefix="jaxgarden_gemma2_tiny"):
    print("\n================ JAXGARDEN GEMMA2 TINY ================")

    config = Gemma2Config(
        vocab_size=320,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        attn_logits_soft_cap=None,
        final_logit_soft_cap=None,
        sliding_window_size=None,
        context_length=128,
        dtype=jnp.float16,
        param_dtype=jnp.float16,
    )

    model = Gemma2ForCausalLM(config, rngs=nnx.Rngs(0))

    input_ids = make_ids(seq=seq, vocab=config.vocab_size)
    position_ids = jnp.arange(seq, dtype=jnp.int32)[None, :]
    attention_mask = jnp.ones((1, seq), dtype=jnp.bool_)

    def forward(input_ids, position_ids, attention_mask):
        logits, _cache = model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            cache=None,
            deterministic=True,
        )
        return logits

    logits = forward(input_ids, position_ids, attention_mask)
    jax.block_until_ready(logits)

    summarize("gemma2 logits", logits)
    print("argmax last:", int(np.argmax(np.asarray(logits)[0, -1])))

    print("exporting StableHLO...")
    mlir = jax_to_mlir(forward, (input_ids, position_ids, attention_mask))
    path = f"{out_prefix}.stablehlo.mlir"
    Path(path).write_text(mlir)
    print("saved:", path)
    return path


def export_gemma3(seq=8, out_prefix="jaxgarden_gemma3_tiny"):
    print("\n================ JAXGARDEN GEMMA3 TINY ================")

    config = Gemma3Config(
        vocab_size=320,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        rope_local_base_freq=10000.0,
        max_position_embeddings=128,
        sliding_window=None,
        sliding_window_pattern=2,
        final_logit_soft_cap=None,
        attn_logit_soft_cap=None,
        tie_word_embeddings=True,
        dtype=jnp.float16,
        param_dtype=jnp.float16,
    )

    model = Gemma3ForCausalLM(config, rngs=nnx.Rngs(0))

    input_ids = make_ids(seq=seq, vocab=config.vocab_size)
    position_ids = jnp.arange(seq, dtype=jnp.int32)[None, :]
    attention_mask = jnp.ones((1, seq), dtype=jnp.bool_)

    def forward(input_ids, position_ids, attention_mask):
        out = model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            cache=None,
            deterministic=True,
        )

        # Some implementations return logits directly, others return tuple.
        if isinstance(out, tuple):
            return out[0]
        return out

    logits = forward(input_ids, position_ids, attention_mask)
    jax.block_until_ready(logits)

    summarize("gemma3 logits", logits)
    print("argmax last:", int(np.argmax(np.asarray(logits)[0, -1])))

    print("exporting StableHLO...")
    mlir = jax_to_mlir(forward, (input_ids, position_ids, attention_mask))
    path = f"{out_prefix}.stablehlo.mlir"
    Path(path).write_text(mlir)
    print("saved:", path)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["llama", "gemma2", "gemma3", "all"], default="all")
    ap.add_argument("--seq", type=int, default=8)
    args = ap.parse_args()

    paths = []

    if args.model in ("llama", "all"):
        paths.append(export_llama(seq=args.seq))

    if args.model in ("gemma2", "all"):
        paths.append(export_gemma2(seq=args.seq))

    if args.model in ("gemma3", "all"):
        paths.append(export_gemma3(seq=args.seq))

    print("\n================ DONE ================")
    for p in paths:
        print(p)
        print(f"  next: python count_default_fusions.py {p}")


if __name__ == "__main__":
    main()
