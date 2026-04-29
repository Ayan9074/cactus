from __future__ import annotations

import argparse
import importlib
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import jax
import jax.numpy as jnp
import numpy as np

from src.transpile.stablehlo.compile_mlir_to_cgraph import compile_mlir_to_cgraph


def _load_callable(spec: str):
    if ":" not in spec:
        raise RuntimeError("--callable must be in module.submodule:function_name format")
    mod_name, fn_name = spec.split(":", 1)
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, fn_name, None)
    if fn is None:
        raise RuntimeError(f"Callable not found: {spec}")
    if not callable(fn):
        raise RuntimeError(f"Target is not callable: {spec}")
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


def export_stablehlo_from_jax(fn, args: List[np.ndarray]) -> str:
    jax_args = [jnp.asarray(a) for a in args]
    lowered = jax.jit(fn).lower(*jax_args)
    return str(lowered.compiler_ir(dialect="stablehlo"))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Export JAX callable -> StableHLO and compile to Cactus .cgraph"
    )
    ap.add_argument("--callable", required=True, help="module.submodule:function_name")
    ap.add_argument(
        "--inputs",
        required=True,
        help="NPZ file with ordered args (arg0,arg1,...) used for lowering + compile sample inputs",
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
        help="How to bind %argN tensors in compiled graph.",
    )
    ap.add_argument(
        "--weight-arg-regex",
        default=None,
        help="Regex over arg names to force mmap in mixed mode.",
    )
    args = ap.parse_args()

    callable_fn = _load_callable(args.callable)
    inputs_npz = Path(args.inputs).resolve()
    out_path = Path(args.out).resolve()
    meta_path = Path(args.meta).resolve() if args.meta else Path(str(out_path) + ".meta.json")

    np_args = _load_args_npz(inputs_npz)
    stablehlo_text = export_stablehlo_from_jax(callable_fn, np_args)

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
    )

    print(f"Exported from callable: {args.callable}")
    print(f"StableHLO source: {stablehlo_path}")


if __name__ == "__main__":
    main()
