#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax.numpy as jnp
from flax import nnx
from jaxgarden import Gemma3Config, Gemma3ForCausalLM, Tokenizer


def config_from_hf_dir(hf_dir: Path) -> Gemma3Config:
    cfg = json.loads((hf_dir / "config.json").read_text())
    swp = cfg.get("sliding_window_pattern")
    if swp is None:
        # Gemma3 HF often stores explicit layer_types instead.
        layer_types = cfg.get("layer_types")
        if isinstance(layer_types, list) and len(layer_types) > 0:
            first_full = next((i for i, t in enumerate(layer_types) if t == "full_attention"), None)
            if first_full is not None:
                swp = first_full + 1
    if swp is None:
        swp = 2

    return Gemma3Config(
        vocab_size=int(cfg["vocab_size"]),
        hidden_size=int(cfg["hidden_size"]),
        intermediate_size=int(cfg["intermediate_size"]),
        num_hidden_layers=int(cfg["num_hidden_layers"]),
        num_attention_heads=int(cfg["num_attention_heads"]),
        num_key_value_heads=int(cfg["num_key_value_heads"]),
        head_dim=int(cfg.get("head_dim", cfg["hidden_size"] // cfg["num_attention_heads"])),
        rms_norm_eps=float(cfg.get("rms_norm_eps", 1e-6)),
        rope_theta=float(cfg.get("rope_theta", 1_000_000.0)),
        query_pre_attn_scalar=float(cfg.get("query_pre_attn_scalar", 256.0)),
        sliding_window=int(cfg.get("sliding_window", 512)),
        sliding_window_pattern=int(swp),
        max_position_embeddings=int(cfg.get("max_position_embeddings", 32768)),
        final_logit_soft_cap=cfg.get("final_logit_soft_cap"),
        attn_logit_soft_cap=cfg.get("attn_logit_soft_cap"),
        dtype=jnp.bfloat16,
        param_dtype=jnp.bfloat16,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Gemma3 HF->JAXgarden smoke test with auto HF config.")
    ap.add_argument("--model-id", default="google/gemma-3-270m")
    ap.add_argument("--hf-dir", default="", help="Optional local HF dir; defaults to ~/.jaxgarden/hf_models/<model-id>")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--max-length", type=int, default=48)
    ap.add_argument("--do-sample", action="store_true")
    args = ap.parse_args()

    if args.hf_dir:
        hf_dir = Path(args.hf_dir).expanduser()
    else:
        hf_dir = Path.home() / ".jaxgarden" / "hf_models" / Path(args.model_id)
    if not hf_dir.exists():
        raise SystemExit(f"HF dir not found: {hf_dir}")

    cfg = config_from_hf_dir(hf_dir)
    model = Gemma3ForCausalLM(cfg, rngs=nnx.Rngs(0))
    model.from_hf(
        args.model_id,
        force_download=False,
        save_in_orbax=False,
        remove_hf_after_conversion=False,
    )

    tok = Tokenizer.from_pretrained(args.model_id)
    model_inputs = tok.encode(args.prompt)
    out = model.generate(
        **model_inputs,
        max_length=int(args.max_length),
        do_sample=bool(args.do_sample),
    )
    print(tok.decode(out))


if __name__ == "__main__":
    main()
