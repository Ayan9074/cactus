from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

from run_gemma3_nommap import parse_main_args_with_locs, weight_for_loc


def shape_from_type(type_str: str) -> tuple[int, ...]:
    m = re.match(r"tensor<(.+)>", type_str.strip())
    if not m:
        return ()
    parts = m.group(1).split("x")
    return tuple(int(x) for x in parts[:-1])


def dtype_from_type(type_str: str) -> str:
    m = re.match(r"tensor<(.+)>", type_str.strip())
    if not m:
        return type_str.strip()
    return m.group(1).split("x")[-1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mlir", default="gemma_stablehlo.mlir")
    ap.add_argument("--weights", default="../weights/gemma-3-270m")
    args = ap.parse_args()

    mlir_text = Path(args.mlir).read_text()
    main_args = parse_main_args_with_locs(mlir_text)
    weights_dir = Path(args.weights)

    n_ok = 0
    n_fail = 0
    print("arg  dtype   expected_shape        actual_shape          min       max       mean_abs   loc")
    print("-" * 140)
    for ssa, ty, loc in main_args:
        if loc == "tokens":
            continue
        exp_shape = shape_from_type(ty)
        exp_dtype = dtype_from_type(ty)
        try:
            arr = weight_for_loc(weights_dir, loc)
            ok = tuple(arr.shape) == tuple(exp_shape)
            if ok:
                n_ok += 1
            else:
                n_fail += 1
            flat = arr.astype(np.float32).reshape(-1)
            print(
                f"{ssa:>6} {exp_dtype:>6} {str(exp_shape):>20} {str(tuple(arr.shape)):>20} "
                f"{float(np.min(flat)):>9.4f} {float(np.max(flat)):>9.4f} {float(np.mean(np.abs(flat))):>11.4f}  {loc}"
            )
            if not ok:
                print(f"  !! SHAPE_MISMATCH expected={exp_shape} got={tuple(arr.shape)}")
        except Exception as e:
            n_fail += 1
            print(f"{ssa:>6} {exp_dtype:>6} {str(exp_shape):>20} {'<error>':>20} {'':>9} {'':>9} {'':>11}  {loc}")
            print(f"  !! ERROR {type(e).__name__}: {e}")

    print("-" * 140)
    print(f"summary: ok={n_ok} fail={n_fail}")


if __name__ == "__main__":
    main()

