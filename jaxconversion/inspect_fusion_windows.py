#!/usr/bin/env python3
import argparse
from parse import parse_mlir


WATCH = {
    "stablehlo.exponential",
    "stablehlo.reduce",
    "stablehlo.negate",
    "stablehlo.divide",
    "stablehlo.multiply",
    "stablehlo.maximum",
    "stablehlo.tanh",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mlir")
    ap.add_argument("--radius", type=int, default=8)
    args = ap.parse_args()

    ir = parse_mlir(open(args.mlir).read())
    nodes = [ir.nodes[nid] for nid in ir.order]

    producer = {}
    consumers = {}

    for i, n in enumerate(nodes):
        for out in n.outputs:
            producer[out] = i
        for inp in n.inputs:
            consumers.setdefault(inp, []).append(i)

    print("MLIR:", args.mlir)
    print("nodes:", len(nodes))
    print("outputs:", ir.outputs)

    for i, n in enumerate(nodes):
        if n.op not in WATCH:
            continue

        print("\n" + "=" * 100)
        print(f"TARGET idx={i} id={n.id} op={n.op}")
        print("  inputs:", n.inputs)
        print("  outputs:", n.outputs)
        print("  attrs:", n.attrs)
        print("  result:", n.result_types)

        print("\nWINDOW:")
        lo = max(0, i - args.radius)
        hi = min(len(nodes), i + args.radius + 1)

        for j in range(lo, hi):
            m = nodes[j]
            mark = ">>" if j == i else "  "
            print(
                f"{mark} [{j:04d}] {m.id:<12} {m.op:<32} "
                f"in={m.inputs} out={m.outputs} attrs={m.attrs}"
            )

        print("\nPRODUCERS:")
        for inp in n.inputs:
            if inp in producer:
                pidx = producer[inp]
                p = nodes[pidx]
                print(f"  {inp} <- [{pidx}] {p.id} {p.op} out={p.outputs} attrs={p.attrs}")
            else:
                print(f"  {inp} <- ARG/CONST/UNKNOWN")

        print("\nCONSUMERS:")
        for out in n.outputs:
            cs = consumers.get(out, [])
            for cidx in cs:
                c = nodes[cidx]
                print(f"  {out} -> [{cidx}] {c.id} {c.op} in={c.inputs} attrs={c.attrs}")

    print("\nDONE")


if __name__ == "__main__":
    main()
