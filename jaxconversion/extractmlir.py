# check_gemma_mlir.py
import argparse
from collections import Counter

from parse import parse_mlir


SUPPORTED_NOW = {
    "stablehlo.add",
    "stablehlo.subtract",
    "stablehlo.multiply",
    "stablehlo.divide",
    "stablehlo.negate",
    "stablehlo.abs",
    "stablehlo.sqrt",
    "stablehlo.rsqrt",
    "stablehlo.log",
    "stablehlo.exponential",
    "stablehlo.tanh",
    "stablehlo.maximum",
    "stablehlo.minimum",
    "stablehlo.dot_general",
    "stablehlo.broadcast_in_dim",
    "stablehlo.reshape",
    "stablehlo.transpose",
    "stablehlo.slice",
    "stablehlo.concatenate",
    "stablehlo.reduce",
    "stablehlo.convert",
    "stablehlo.compare",
    "stablehlo.select",
    "stablehlo.and",
    "stablehlo.or",
    "stablehlo.not",
    "stablehlo.constant",
    "stablehlo.return",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mlir")
    args = ap.parse_args()

    text = open("gemma_real_prefill.stablehlo.mlir").read()
    ir = parse_mlir(text)

    counts = Counter()
    examples = {}

    for nid in ir.order:
        n = ir.nodes[nid]
        counts[n.op] += 1
        examples.setdefault(n.op, n)

    print("Parsed node count:", len(ir.order))
    print("\nOps:")
    for op, c in sorted(counts.items()):
        flag = "OK" if op in SUPPORTED_NOW else "CHECK"
        print(f"  {flag:<5} {op:<32} {c}")

    print("\nUnsupported/check examples:")
    for op, n in sorted(examples.items()):
        if op not in SUPPORTED_NOW:
            print(f"\n{op}")
            print("  outputs:", n.outputs)
            print("  inputs:", n.inputs)
            print("  attrs:", n.attrs)
            print("  result_types:", n.result_types)


if __name__ == "__main__":
    main()