#!/usr/bin/env python3
import argparse
from collections import Counter

from parse import parse_mlir
from lower_to_cactus import lower_to_cactus
from pattern_registry import Pattern
from patterns.default import DEFAULT_PATTERNS


def wrap_patterns(patterns, counts):
    wrapped = []

    for p in patterns:
        original_handler = p.handler

        def make_handler(pattern_name, handler):
            def wrapped_handler(ctx, nodes, idx):
                result = handler(ctx, nodes, idx)
                if result is not None:
                    counts[pattern_name] += 1
                    consumed = result[1]
                    lead = nodes[idx]
                    print(
                        f"FUSED {pattern_name:<16} "
                        f"idx={idx:<5} consumed={consumed:<3} "
                        f"lead={lead.id:<12} op={lead.op:<30} "
                        f"outputs={lead.outputs}"
                    )
                return result
            return wrapped_handler

        wrapped.append(
            Pattern(
                name=p.name,
                handler=make_handler(p.name, original_handler),
                trigger_ops=p.trigger_ops,
            )
        )

    return wrapped


def op_counts(ir):
    c = Counter()
    for nid in ir.order:
        c[ir.nodes[nid].op] += 1
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mlir", help="Path to StableHLO MLIR file")
    ap.add_argument("--verbose-lowering", action="store_true")
    args = ap.parse_args()

    print("Reading:", args.mlir)
    mlir = open(args.mlir, "r").read()
    ir = parse_mlir(mlir)

    print("\nIR summary:")
    print("  inputs:", len(ir.inputs))
    print("  outputs:", ir.outputs)
    print("  nodes:", len(ir.order))
    print("  constants:", len(ir.constants))

    print("\nOp counts:")
    counts = op_counts(ir)
    for op, n in sorted(counts.items()):
        print(f"  {op:<36} {n}")

    fusion_counts = Counter()
    wrapped = wrap_patterns(DEFAULT_PATTERNS, fusion_counts)

    print("\nLowering with wrapped DEFAULT_PATTERNS...")
    g, env = lower_to_cactus(
        ir,
        patterns=wrapped,
        verbose=args.verbose_lowering,
        strict_math=False,
    )

    print("\n================ FUSION SUMMARY ================")
    total = sum(fusion_counts.values())

    if total == 0:
        print("No default fusions fired.")
    else:
        for name, n in fusion_counts.most_common():
            print(f"  {name:<16} {n}")
        print(f"  {'TOTAL':<16} {total}")

    print("\nPattern opportunities rough check:")
    print("  stablehlo.maximum:", counts.get("stablehlo.maximum", 0), "possible ReLU-ish but may be masks/clamps")
    print("  stablehlo.reduce:", counts.get("stablehlo.reduce", 0), "possible softmax/RMSNorm reductions")
    print("  stablehlo.exponential:", counts.get("stablehlo.exponential", 0), "softmax usually has exp")
    print("  stablehlo.divide:", counts.get("stablehlo.divide", 0), "softmax/RMSNorm/etc")

    print("\nDone.")


if __name__ == "__main__":
    main()
