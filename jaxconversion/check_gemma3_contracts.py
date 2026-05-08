from __future__ import annotations

import re
from pathlib import Path

from parse import parse_mlir
from run_gemma3_nommap import parse_main_args_with_locs


def main() -> None:
    mlir_path = Path("gemma_stablehlo.mlir")
    mlir_text = mlir_path.read_text()
    main_args = parse_main_args_with_locs(mlir_text)

    cleaned = re.sub(r' loc\("[^"]*"\)', "", mlir_text)
    cleaned = re.sub(r" loc\(#loc[0-9]+\)", "", cleaned)
    cleaned = re.sub(r"^#loc.*$", "", cleaned, flags=re.M)
    ir = parse_mlir(cleaned)

    consumers: dict[str, list[str]] = {}
    for nid in ir.order:
        node = ir.nodes[nid]
        for inp in node.inputs:
            consumers.setdefault(inp, []).append(nid)

    def first_dot(arg: str):
        for nid in consumers.get(arg, []):
            n = ir.nodes[nid]
            if n.op == "stablehlo.dot_general":
                return nid, n
        return None, None

    print("Gemma3 StableHLO Contract Check")
    print("================================")
    print(f"inputs={len(ir.inputs)} outputs={len(ir.outputs)} nodes={len(ir.order)}")

    # Focus on transformed weights per layer.
    pats = [
        "attn']['q_einsum']['w",
        "attn']['kv_einsum']['w",
        "attn']['attn_vec_einsum']['w",
        "mlp']['gating_einsum",
        "mlp']['linear",
    ]

    rows = []
    for ssa, _ty, loc in main_args:
        if not loc.startswith("params['layer_"):
            continue
        if not any(p in loc for p in pats):
            continue
        nid, dot = first_dot(ssa)
        shape = ir.values[ssa].shape
        if dot is None:
            rows.append((ssa, loc, shape, "NO_DOT", "", "", ""))
            continue
        in_shapes = [ir.values[i].shape for i in dot.inputs]
        rows.append(
            (
                ssa,
                loc,
                shape,
                nid,
                str(dot.attrs.get("contracting_dims")),
                str(in_shapes),
                str([ir.values[o].shape for o in dot.outputs]),
            )
        )

    for r in rows:
        print(
            f"{r[0]:>8}  shape={r[2]}  dot={r[3]}  contract={r[4]}  in={r[5]}  out={r[6]}\n"
            f"         loc={r[1]}"
        )


if __name__ == "__main__":
    main()

