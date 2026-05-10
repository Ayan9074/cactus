#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(ROOT.parent))


def pipeline_gemma2_manual(args) -> None:
    out_prefix = Path(args.out_prefix)
    export_cmd = [
        sys.executable,
        str(ROOT / "export_gemma_bundle.py"),
        "--out-prefix",
        str(out_prefix),
        "--seq",
        str(args.seq),
    ]
    if args.real_hf_dir:
        export_cmd += ["--real-hf-dir", str(Path(args.real_hf_dir).expanduser())]
    if args.prompt:
        export_cmd += ["--prompt", args.prompt]
    if args.tokenizer_model_id:
        export_cmd += ["--tokenizer-model-id", args.tokenizer_model_id]
    if args.show_text:
        export_cmd += ["--show-text"]
    if args.test:
        export_cmd += ["--test"]

    run(export_cmd)

    mlir = str(out_prefix.with_suffix(".stablehlo.mlir"))
    npz = str(out_prefix.with_suffix(".inputs_weights.npz"))
    graph_out = str(out_prefix.with_suffix(".cactus"))
    report_json = str(out_prefix.with_suffix(".convert_report.json"))

    run([
        sys.executable,
        str(ROOT / "convert_any.py"),
        mlir,
        "--inputs-npz",
        npz,
        "--run",
        "--graph-out",
        graph_out,
        "--report-json",
        report_json,
    ])

def pipeline_llama_jaxgarden(args) -> None:
    run([
        sys.executable,
        str(ROOT / "text_jaxgarden_llama_cactus.py"),
        "--seq",
        str(args.seq),
        "--hf-model-id",
        args.hf_model_id,
        "--tokenizer-id",
        args.tokenizer_model_id if args.tokenizer_model_id else args.default_llama_tokenizer,
        "--prompt",
        args.prompt,
        "--mlir-out",
        str(Path(args.out_prefix).with_suffix(".stablehlo.mlir")),
    ])


def main() -> None:
    ap = argparse.ArgumentParser(description="One-shot JAXgarden->StableHLO + HF-weight conversion pipeline")
    ap.add_argument("--model", choices=["gemma2", "llama"], default="gemma2")
    ap.add_argument("--out-prefix", default="/private/tmp/gemma2_pipeline")
    ap.add_argument("--seq", type=int, default=8)
    ap.add_argument("--real-hf-dir", default="", help="Local HF model dir (Gemma2 manual mapper path)")
    ap.add_argument("--hf-model-id", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0", help="HF model id for llama path")
    ap.add_argument("--default-llama-tokenizer", default="/Users/ayan/.jaxgarden/hf_models/TinyLlama/TinyLlama-1.1B-Chat-v1.0", help="Local tokenizer/model dir for llama decode display")
    ap.add_argument("--prompt", default="The meaning of life is")
    ap.add_argument("--tokenizer-model-id", default="")
    ap.add_argument("--show-text", action="store_true")
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()

    if args.model == "gemma2":
        pipeline_gemma2_manual(args)
    else:
        pipeline_llama_jaxgarden(args)


if __name__ == "__main__":
    main()
