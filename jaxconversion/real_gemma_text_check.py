#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    ap = argparse.ArgumentParser(description="Quick coherent-text check using real Gemma weights.")
    ap.add_argument(
        "--model-dir",
        default="/Users/ayan/.jaxgarden/hf_models/google/gemma-2-2b-it",
        help="Local HF model directory (already downloaded).",
    )
    ap.add_argument("--prompt", default="Write a short cheerful message about coding:", help="Prompt text")
    ap.add_argument("--max-new-tokens", type=int, default=48, help="Tokens to generate")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        raise SystemExit(f"Model dir not found: {model_dir}")

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        local_files_only=True,
        dtype=torch.float16,
    )
    model.eval()
    model.to("cpu")

    inputs = tokenizer(args.prompt, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(
            **inputs,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
            max_new_tokens=int(args.max_new_tokens),
            pad_token_id=tokenizer.eos_token_id,
        )

    text = tokenizer.decode(out[0], skip_special_tokens=True)
    print("PROMPT:")
    print(args.prompt)
    print("\nGENERATED:")
    print(text)


if __name__ == "__main__":
    main()
