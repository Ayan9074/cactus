from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from parse import IRGraph, parse_mlir


FALLBACK_SUPPORTED_OPS = {
    "stablehlo.dot_general",
    "stablehlo.dot",
    "stablehlo.add",
    "stablehlo.subtract",
    "stablehlo.multiply",
    "stablehlo.divide",
    "stablehlo.maximum",
    "stablehlo.minimum",
    "stablehlo.negate",
    "stablehlo.abs",
    "stablehlo.sqrt",
    "stablehlo.rsqrt",
    "stablehlo.exponential",
    "stablehlo.log",
    "stablehlo.sine",
    "stablehlo.cosine",
    "stablehlo.tanh",
    "stablehlo.logistic",
    "stablehlo.relu",
    "stablehlo.silu",
    "stablehlo.gelu",
    "stablehlo.power",
    "chlo.square",
    "stablehlo.reshape",
    "stablehlo.transpose",
    "stablehlo.broadcast_in_dim",
    "stablehlo.slice",
    "stablehlo.concatenate",
    "stablehlo.reduce",
    "stablehlo.gather",
    "stablehlo.convert",
    "stablehlo.softmax",
    "stablehlo.compare",
    "stablehlo.and",
    "stablehlo.or",
    "stablehlo.not",
    "stablehlo.select",
    "stablehlo.iota",
    "cactus.mean_keepdims",
    "cactus.mean",
}


def _supported_ops() -> set[str]:
    try:
        from op_lowering import OP_LOWERING_TABLE  # type: ignore

        return set(OP_LOWERING_TABLE.keys())
    except Exception:
        return FALLBACK_SUPPORTED_OPS


@dataclass
class CompatibilityReport:
    tier: str
    profile: str
    total_nodes: int
    live_nodes: int
    supported_nodes: int
    unsupported_nodes: int
    op_histogram: dict[str, int]
    unsupported_ops: dict[str, int]
    notes: list[str]


def _producer_map(ir: IRGraph) -> dict[str, Any]:
    out_to_node = {}
    for nid in ir.order:
        n = ir.nodes[nid]
        for out in n.outputs:
            out_to_node[out] = n
    return out_to_node


def _compute_live_nodes(ir: IRGraph) -> set[str]:
    prod = _producer_map(ir)
    live_values = set(ir.outputs)
    live_nodes = set()
    stack = list(ir.outputs)

    while stack:
        ssa = stack.pop()
        node = prod.get(ssa)
        if node is None:
            continue
        if node.id in live_nodes:
            continue
        live_nodes.add(node.id)
        for inp in node.inputs:
            if inp not in live_values:
                live_values.add(inp)
                stack.append(inp)
    return live_nodes


def _has_dynamic_shape(ir: IRGraph) -> bool:
    for v in ir.values.values():
        for d in v.shape:
            if int(d) < 0:
                return True
    return False


def _guess_profile(ir: IRGraph) -> str:
    ops = {ir.nodes[nid].op for nid in ir.order}
    if "stablehlo.gather" in ops and "stablehlo.dot_general" in ops:
        return "transformer"
    if "stablehlo.convolution" in ops:
        return "vision"
    return "generic"


def analyze_ir(ir: IRGraph) -> CompatibilityReport:
    supported_ops = _supported_ops()
    live_nodes = _compute_live_nodes(ir)
    op_hist = Counter()
    unsupported = Counter()
    supported_nodes = 0

    for nid in ir.order:
        node = ir.nodes[nid]
        if nid not in live_nodes:
            continue
        op_hist[node.op] += 1
        if node.op in supported_ops:
            supported_nodes += 1
        else:
            unsupported[node.op] += 1

    live_count = sum(op_hist.values())
    unsupported_count = sum(unsupported.values())
    notes: list[str] = []

    if _has_dynamic_shape(ir):
        notes.append("Dynamic shapes detected; current lowering path expects static shapes.")

    if unsupported_count == 0 and not notes:
        tier = "A-ready"
    elif unsupported_count <= max(3, live_count // 20):
        tier = "B-needs_rewrites"
    else:
        tier = "C-blocked"

    if unsupported_count:
        notes.append("Unsupported ops require new primitive handlers or fusion patterns.")

    return CompatibilityReport(
        tier=tier,
        profile=_guess_profile(ir),
        total_nodes=len(ir.order),
        live_nodes=live_count,
        supported_nodes=supported_nodes,
        unsupported_nodes=unsupported_count,
        op_histogram=dict(sorted(op_hist.items())),
        unsupported_ops=dict(sorted(unsupported.items())),
        notes=notes,
    )


def analyze_mlir_text(mlir_text: str) -> CompatibilityReport:
    ir = parse_mlir(mlir_text)
    return analyze_ir(ir)


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze StableHLO MLIR compatibility with Cactus lowering.")
    ap.add_argument("mlir", help="Path to .mlir file")
    ap.add_argument("--json-out", help="Optional JSON output path")
    args = ap.parse_args()

    mlir_path = Path(args.mlir)
    report = analyze_mlir_text(mlir_path.read_text())

    payload = asdict(report)
    print(json.dumps(payload, indent=2, sort_keys=True))

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
