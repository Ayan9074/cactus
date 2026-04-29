from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from transformers import AutoTokenizer, FlaxAutoModelForCausalLM

from src.transpile.stablehlo.compile_mlir_to_cgraph import compile_mlir_to_cgraph
from src.IR.stablehlo_ir import parse_stablehlo_ops


def _topk_overlap(a: np.ndarray, b: np.ndarray, k: int = 5) -> int:
    aa = set(np.argpartition(a, -k)[-k:].tolist())
    bb = set(np.argpartition(b, -k)[-k:].tolist())
    return len(aa & bb)


def main() -> None:
    ap = argparse.ArgumentParser(description="Import FlaxAutoModelForCausalLM -> StableHLO -> Cactus .cgraph")
    ap.add_argument("--model", default="hf-internal-testing/tiny-random-gpt2")
    ap.add_argument("--prompt", default="Hello from Cactus")
    ap.add_argument("--seq-len", type=int, default=16)
    ap.add_argument("--out", required=True, help="Output .cgraph")
    ap.add_argument("--meta", default=None, help="Output metadata json")
    ap.add_argument("--save-mlir", default=None, help="Optional path to save exported StableHLO")
    ap.add_argument("--weight-policy", default="inputs", choices=["inputs", "mmap", "mixed"])
    ap.add_argument("--weight-arg-regex", default=None)
    ap.add_argument("--test", action="store_true", help="Run accuracy + timing check vs JAX")
    args = ap.parse_args()

    out_path = Path(args.out).resolve()
    meta_path = Path(args.meta).resolve() if args.meta else Path(str(out_path) + ".meta.json")

    t_load0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = FlaxAutoModelForCausalLM.from_pretrained(args.model)
    t_load_ms = (time.perf_counter() - t_load0) * 1000.0

    enc = tok(
        args.prompt,
        return_tensors="np",
        truncation=True,
        padding="max_length",
        max_length=args.seq_len,
    )
    input_ids = enc["input_ids"].astype(np.int32)
    attention_mask = enc["attention_mask"].astype(np.int32)

    # Some tiny/random checkpoints use reduced vocab tables while inheriting a
    # tokenizer whose ids can exceed model vocab size. Canonicalize ids into
    # the model embedding range to avoid out-of-bounds gather in exported
    # StableHLO helper functions like @_take.
    vocab_size = int(getattr(model.config, "vocab_size", 0) or 0)
    if vocab_size > 0:
        min_id = int(input_ids.min())
        max_id = int(input_ids.max())
        if min_id < 0 or max_id >= vocab_size:
            print(
                f"[flax_import] canonicalizing input_ids to vocab range [0,{vocab_size-1}] "
                f"(saw min={min_id}, max={max_id})"
            )
            input_ids = np.mod(input_ids, vocab_size).astype(np.int32)

    def forward_fn(ids, mask):
        out = model(input_ids=ids, attention_mask=mask, params=model.params, train=False)
        return out.logits

    # 1) export stablehlo
    t0 = time.perf_counter()
    lowered = jax.jit(forward_fn).lower(jnp.asarray(input_ids), jnp.asarray(attention_mask))
    stablehlo_text = str(lowered.compiler_ir(dialect="stablehlo"))
    t_export_ms = (time.perf_counter() - t0) * 1000.0

    # 2) parse
    t1 = time.perf_counter()
    nodes = parse_stablehlo_ops(stablehlo_text)
    t_parse_ms = (time.perf_counter() - t1) * 1000.0

    if args.save_mlir:
        stablehlo_path = Path(args.save_mlir).resolve()
        stablehlo_path.write_text(stablehlo_text)
    else:
        tmp = tempfile.NamedTemporaryFile(prefix="flax_causallm_", suffix=".stablehlo.mlir", delete=False)
        stablehlo_path = Path(tmp.name)
        tmp.close()
        stablehlo_path.write_text(stablehlo_text)

    npz_path = Path(str(out_path) + ".inputs.npz")
    np.savez(npz_path, arg0=input_ids, arg1=attention_mask)

    # 3) build+save cgraph
    t2 = time.perf_counter()
    compile_mlir_to_cgraph(
        stablehlo_path=stablehlo_path,
        inputs_path=npz_path,
        output_cgraph_path=out_path,
        output_meta_path=meta_path,
        output_name=None,
        weight_policy=args.weight_policy,
        weight_arg_regex=args.weight_arg_regex,
    )
    t_buildsave_ms = (time.perf_counter() - t2) * 1000.0

    result = {
        "model": args.model,
        "load_model_tokenizer_ms": t_load_ms,
        "stablehlo_export_ms": t_export_ms,
        "stablehlo_parse_ms": t_parse_ms,
        "cgraph_build_save_ms": t_buildsave_ms,
        "num_nodes": len(nodes),
        "out_cgraph": str(out_path),
        "out_meta": str(meta_path),
        "saved_inputs_npz": str(npz_path),
        "weight_policy": args.weight_policy,
    }

    if args.test:
        # JAX reference
        _ = jax.jit(forward_fn)(jnp.asarray(input_ids), jnp.asarray(attention_mask)).block_until_ready()
        tj0 = time.perf_counter()
        jax_logits = jax.jit(forward_fn)(jnp.asarray(input_ids), jnp.asarray(attention_mask)).block_until_ready()
        jax_ms = (time.perf_counter() - tj0) * 1000.0
        jax_logits = np.asarray(jax_logits, dtype=np.float32)

        # Cactus runner
        out_npy = Path(str(out_path) + ".out.npy")
        cmd = [
            sys.executable,
            "-m",
            "src.IR.run_cgraph_generic",
            "--cgraph",
            str(out_path),
            "--meta",
            str(meta_path),
            "--inputs",
            str(npz_path),
            "--out",
            str(out_npy),
        ]
        env = dict(os.environ)
        py_root = str(Path(__file__).resolve().parents[2])
        env["PYTHONPATH"] = py_root + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

        tc0 = time.perf_counter()
        subprocess.check_call(cmd, env=env)
        cactus_total_ms = (time.perf_counter() - tc0) * 1000.0
        cactus_logits = np.load(out_npy).astype(np.float32)

        d = cactus_logits - jax_logits
        mae = float(np.mean(np.abs(d)))
        max_abs = float(np.max(np.abs(d)))
        rmse = float(np.sqrt(np.mean(d * d)))
        cosine = float(np.dot(cactus_logits.ravel(), jax_logits.ravel()) / (
            np.linalg.norm(cactus_logits.ravel()) * np.linalg.norm(jax_logits.ravel()) + 1e-12
        ))

        # top-5 overlap on final token logits
        c_last = cactus_logits[0, -1]
        j_last = jax_logits[0, -1]
        top5 = _topk_overlap(c_last, j_last, k=5)

        result.update({
            "jax_ref_ms": jax_ms,
            "cactus_run_total_ms": cactus_total_ms,
            "mae": mae,
            "max_abs": max_abs,
            "rmse": rmse,
            "cosine": cosine,
            "top5_overlap_last_token": int(top5),
        })

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
