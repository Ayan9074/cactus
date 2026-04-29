from __future__ import annotations

import argparse
import importlib
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from src.transpile.stablehlo.compile_mlir_to_cgraph import compile_mlir_to_cgraph


def _load_symbol(spec: str):
    if ":" not in spec:
        raise RuntimeError("--callable must be in module.submodule:function_name format")
    mod_name, fn_name = spec.split(":", 1)
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, fn_name, None)
    if fn is None:
        raise RuntimeError(f"Symbol not found: {spec}")
    return fn


def _load_args_npz(npz_path: Path) -> List[np.ndarray]:
    with np.load(npz_path, allow_pickle=False) as z:
        keys = list(z.keys())
        if not keys:
            raise RuntimeError("Input NPZ is empty")

        def key_order(k: str):
            if k.startswith("arg") and k[3:].isdigit():
                return int(k[3:])
            if k.startswith("%arg") and k[4:].isdigit():
                return int(k[4:])
            if k.isdigit():
                return int(k)
            return 10**9

        sorted_keys = sorted(keys, key=key_order)
        arrays: List[np.ndarray] = []
        for k in sorted_keys:
            arrays.append(np.asarray(z[k]))

    return arrays


def export_stablehlo_from_jax(fn, args: Sequence[Any]) -> str:
    jax_args = tuple(jax.tree_util.tree_map(jnp.asarray, args))
    lowered = jax.jit(fn).lower(*jax_args)
    return str(lowered.compiler_ir(dialect="stablehlo"))


def _resolve_fn_and_args(
    callable_spec: str,
    callable_kind: str,
    inputs_npz: Path | None,
) -> Tuple[Any, Sequence[Any], List[np.ndarray]]:
    sym = _load_symbol(callable_spec)
    if callable_kind == "fn":
        if not callable(sym):
            raise RuntimeError(f"Target is not callable function: {callable_spec}")
        if inputs_npz is None:
            raise RuntimeError("--inputs is required when --callable-kind=fn")
        flat = _load_args_npz(inputs_npz)
        return sym, flat, flat

    if not callable(sym):
        raise RuntimeError(f"Factory target is not callable: {callable_spec}")
    out = sym()
    if not isinstance(out, dict) or "fn" not in out or "args" not in out:
        raise RuntimeError(
            "Factory must return dict with keys {'fn','args'}, where args is an iterable "
            "of JAX/NumPy arrays or pytrees."
        )
    fn = out["fn"]
    raw_args = out["args"]
    if not callable(fn):
        raise RuntimeError("Factory return value 'fn' is not callable.")
    call_args = tuple(raw_args)
    flat_args = [np.asarray(a) for a in jax.tree_util.tree_leaves(call_args)]
    if not flat_args:
        raise RuntimeError("Factory returned empty args.")
    return fn, call_args, flat_args


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Export JAX callable -> StableHLO and compile to Cactus .cgraph"
    )
    ap.add_argument("--callable", required=True, help="module.submodule:function_name")
    ap.add_argument(
        "--inputs",
        required=False,
        help="NPZ file with ordered args (arg0,arg1,...) used for lowering + compile sample inputs",
    )
    ap.add_argument(
        "--callable-kind",
        default="fn",
        choices=["fn", "factory"],
        help=(
            "fn: --callable points to function and --inputs supplies args. "
            "factory: --callable points to a zero-arg builder returning {'fn','args'}."
        ),
    )
    ap.add_argument("--out", required=True, help="Output .cgraph path")
    ap.add_argument("--meta", default=None, help="Output metadata JSON path")
    ap.add_argument(
        "--output-name",
        default=None,
        help="Return node name to select from StableHLO return values (default: first)",
    )
    ap.add_argument(
        "--save-mlir",
        default=None,
        help="Optional path to save generated StableHLO MLIR",
    )
    ap.add_argument(
        "--weight-policy",
        default="inputs",
        choices=["inputs", "mmap", "mixed"],
        help="How to bind %%argN tensors in compiled graph.",
    )
    ap.add_argument(
        "--weight-arg-regex",
        default=None,
        help="Regex over arg names to force mmap in mixed mode.",
    )
    ap.add_argument(
        "--primary-runtime-arg",
        default="%arg0",
        help="Argument name to force as runtime input in mmap mode (default: %%arg0).",
    )
    ap.add_argument(
        "--cast-non-primary-fp16",
        type=int,
        default=1,
        help="Cast non-primary float args to FP16 in compiler input materialization (1/0).",
    )
    ap.add_argument("--disable-attention-fusion", action="store_true")
    ap.add_argument("--disable-rmsnorm-fusion", action="store_true")
    args = ap.parse_args()

    inputs_npz = Path(args.inputs).resolve() if args.inputs else None
    out_path = Path(args.out).resolve()
    meta_path = Path(args.meta).resolve() if args.meta else Path(str(out_path) + ".meta.json")

    callable_fn, call_args, np_args = _resolve_fn_and_args(args.callable, args.callable_kind, inputs_npz)
    stablehlo_text = export_stablehlo_from_jax(callable_fn, call_args)

    if inputs_npz is None:
        inputs_npz = Path(str(out_path) + ".inputs.npz")
        np.savez(inputs_npz, **{f"arg{i}": a for i, a in enumerate(np_args)})

    if args.save_mlir:
        mlir_path = Path(args.save_mlir).resolve()
        mlir_path.write_text(stablehlo_text)
        stablehlo_path = mlir_path
    else:
        tmp = tempfile.NamedTemporaryFile(prefix="jax_export_", suffix=".stablehlo.mlir", delete=False)
        tmp_path = Path(tmp.name)
        tmp.close()
        tmp_path.write_text(stablehlo_text)
        stablehlo_path = tmp_path

    compile_mlir_to_cgraph(
        stablehlo_path=stablehlo_path,
        inputs_path=inputs_npz,
        output_cgraph_path=out_path,
        output_meta_path=meta_path,
        output_name=args.output_name,
        weight_policy=args.weight_policy,
        weight_arg_regex=args.weight_arg_regex,
        cast_non_primary_float_to_fp16=bool(args.cast_non_primary_fp16),
        primary_runtime_arg=args.primary_runtime_arg,
        enable_attention_fusion=not args.disable_attention_fusion,
        enable_rmsnorm_fusion=not args.disable_rmsnorm_fusion,
    )

    print(f"Exported from callable: {args.callable}")
    print(f"StableHLO source: {stablehlo_path}")


if __name__ == "__main__":
    main()
