from __future__ import annotations

import argparse
import runpy
import sys


def _run_module(module: str, args: list[str]) -> None:
    old_argv = sys.argv[:]
    try:
        sys.argv = [module, *args]
        runpy.run_module(module, run_name="__main__")
    finally:
        sys.argv = old_argv


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Unified Cactus graph pipeline.\n\n"
            "Path 1: python model/function -> JAX StableHLO -> Cactus\n"
            "Path 2: existing StableHLO (.mlir) -> Cactus\n"
            "Path 3: FlaxAutoModelForCausalLM -> StableHLO -> Cactus\n"
            "Path 4: model-specific optimized bridges (e.g. Gemma)"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("path1-python", help="Path 1: callable -> JAX graph -> Cactus")
    p1.add_argument("args", nargs=argparse.REMAINDER, help="Pass-through args for compile_jax_to_cgraph")

    p2 = sub.add_parser("path2-jax-graph", help="Path 2: StableHLO MLIR -> Cactus")
    p2.add_argument("args", nargs=argparse.REMAINDER, help="Pass-through args for compile_mlir_to_cgraph")

    p3 = sub.add_parser("path3-flax-auto", help="Path 3: FlaxAutoModelForCausalLM -> Cactus")
    p3.add_argument("args", nargs=argparse.REMAINDER, help="Pass-through args for import_flax_causallm")

    p4 = sub.add_parser("path4-gemma", help="Path 4: Gemma-specific optimized path")
    p4.add_argument("args", nargs=argparse.REMAINDER, help="Pass-through args for gemma_kv_cache_compiler_bridge")

    pr = sub.add_parser("run", help="Run compiled Cactus graph artifact")
    pr.add_argument("args", nargs=argparse.REMAINDER, help="Pass-through args for run_cgraph_generic")

    pp = sub.add_parser("profile", help="Profile generic JAX callable path")
    pp.add_argument("args", nargs=argparse.REMAINDER, help="Pass-through args for profile_jax_generic")

    args = ap.parse_args()
    pass_args = args.args
    if pass_args and pass_args[0] == "--":
        pass_args = pass_args[1:]

    if args.cmd == "path1-python":
        _run_module("src.IR.compile_jax_to_cgraph", pass_args)
    elif args.cmd == "path2-jax-graph":
        _run_module("src.IR.compile_mlir_to_cgraph", pass_args)
    elif args.cmd == "path3-flax-auto":
        _run_module("src.IR.import_flax_causallm", pass_args)
    elif args.cmd == "path4-gemma":
        _run_module("src.IR.gemma_kv_cache_compiler_bridge", pass_args)
    elif args.cmd == "run":
        _run_module("src.IR.run_cgraph_generic", pass_args)
    elif args.cmd == "profile":
        _run_module("src.IR.profile_jax_generic", pass_args)
    else:
        raise RuntimeError(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    main()
