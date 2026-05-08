#!/usr/bin/env python3
import argparse
from parse import parse_mlir


def op(nodes, idx, offset=0):
    i = idx + offset
    return nodes[i] if 0 <= i < len(nodes) else None


def out(n, i=0):
    return n.outputs[i] if n and len(n.outputs) > i else None


def inp(n, i=0):
    return n.inputs[i] if n and len(n.inputs) > i else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mlir")
    ap.add_argument("--radius", type=int, default=20)
    args = ap.parse_args()

    ir = parse_mlir(open(args.mlir).read())
    nodes = [ir.nodes[nid] for nid in ir.order]

    producers = {}
    consumers = {}

    for i, n in enumerate(nodes):
        for o in n.outputs:
            producers[o] = i
        for x in n.inputs:
            consumers.setdefault(x, []).append(i)

    print("MLIR:", args.mlir)
    print("nodes:", len(nodes))

    candidates = []

    for i, n in enumerate(nodes):
        if n.op != "stablehlo.multiply":
            continue

        # Potential square: multiply(a, a)
        if len(n.inputs) == 2 and n.inputs[0] == n.inputs[1]:
            candidates.append(i)

    print("square-like multiply candidates:", candidates)
    print()

    for i in candidates:
        n = nodes[i]
        x = n.inputs[0]
        pidx = producers.get(x)
        prod = nodes[pidx] if pidx is not None else None

        print("=" * 100)
        print(f"CANDIDATE idx={i} id={n.id} op={n.op}")
        print("square input:", x)
        if prod:
            print(f"producer of square input: idx={pidx} id={prod.id} op={prod.op} inputs={prod.inputs} outputs={prod.outputs}")
        else:
            print("producer of square input: ARG/CONST/UNKNOWN")

        print("\nWindow:")
        lo = max(0, i - 5)
        hi = min(len(nodes), i + args.radius)

        for j in range(lo, hi):
            m = nodes[j]
            mark = ">>" if j == i else "  "
            print(
                f"{mark} [{j:04d}] {m.id:<12} {m.op:<32} "
                f"in={m.inputs} out={m.outputs} attrs={m.attrs} result={m.result_types}"
            )

        print("\nConsumers:")
        for o in n.outputs:
            for cidx in consumers.get(o, []):
                c = nodes[cidx]
                print(f"  {o} -> idx={cidx} id={c.id} op={c.op} inputs={c.inputs} attrs={c.attrs}")

        print()

    print("DONE")


if __name__ == "__main__":
    main()
