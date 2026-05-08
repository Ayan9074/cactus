"""
patterns_default.py
===================
Safer built-in model-agnostic fusions.
"""
from __future__ import annotations

from pattern_registry import Pattern, LoweringCtx, is_const, const_value


def _op(nodes, idx, offset=0):
    i = idx + offset
    return nodes[i] if i < len(nodes) else None


def _is(nodes, idx, opcode, offset=0):
    n = _op(nodes, idx, offset)
    return n is not None and n.op == opcode


def _get(ctx, ssa):
    return ctx.env[ssa]


def _norm_axis(axis: int, rank: int) -> int:
    axis = int(axis)
    return axis + rank if axis < 0 else axis


def _dims(node):
    d = node.attrs.get("dimensions", [-1])
    if isinstance(d, int):
        return [int(d)]
    if isinstance(d, str):
        import re
        m = re.search(r"array<[^:>]+:\s*([^>]*)>", d)
        raw = m.group(1) if m else d.strip("[]")
        return [int(x) for x in re.findall(r"-?\d+", raw)]
    return [int(x) for x in d]


def _reduce_kind(node):
    body = str(node.attrs.get("applies", "")).lower()
    if "maximum" in body or body.endswith("max") or ".max" in body:
        return "max"
    if "minimum" in body or body.endswith("min") or ".min" in body:
        return "min"
    if "multiply" in body or ".mul" in body:
        return "mul"
    if "add" in body or "plus" in body or not body:
        return "add"
    return "unknown"


# ---------------------------------------------------------------------------
# ReLU: broadcast_in_dim(zero) + maximum -> relu
# ---------------------------------------------------------------------------

def _relu_pattern(ctx: LoweringCtx, nodes: list, idx: int):
    bcast = _op(nodes, idx)
    maxn = _op(nodes, idx, 1)

    if not (bcast and maxn):
        return None
    if bcast.op != "stablehlo.broadcast_in_dim" or maxn.op != "stablehlo.maximum":
        return None

    bcast_src = bcast.inputs[0]
    if not is_const(ctx, bcast_src):
        return None
    try:
        if float(const_value(ctx, bcast_src)) != 0.0:
            return None
    except Exception:
        return None

    bcast_out = bcast.outputs[0]
    if bcast_out not in maxn.inputs:
        return None

    actual_ssa = next(s for s in maxn.inputs if s != bcast_out)
    if actual_ssa not in ctx.env:
        return None

    x = _get(ctx, actual_ssa)
    out = ctx.graph.relu(x)
    ctx.env[maxn.outputs[0]] = out
    ctx.env[bcast.outputs[0]] = out  # consumed; should not be used outside this pattern
    return [out], 2


RELU = Pattern(name="relu", handler=_relu_pattern, trigger_ops={"stablehlo.broadcast_in_dim"})


# ---------------------------------------------------------------------------
# Softmax: reduce(max)+bcast+sub+exp+reduce(sum)+bcast+div -> softmax
# ---------------------------------------------------------------------------

def _softmax_pattern(ctx: LoweringCtx, nodes: list, idx: int):
    rmax = _op(nodes, idx, 0)
    bmax = _op(nodes, idx, 1)
    sub = _op(nodes, idx, 2)
    exp = _op(nodes, idx, 3)
    rsum = _op(nodes, idx, 4)
    bsum = _op(nodes, idx, 5)
    div = _op(nodes, idx, 6)

    expected = [
        (rmax, "stablehlo.reduce"),
        (bmax, "stablehlo.broadcast_in_dim"),
        (sub, "stablehlo.subtract"),
        (exp, "stablehlo.exponential"),
        (rsum, "stablehlo.reduce"),
        (bsum, "stablehlo.broadcast_in_dim"),
        (div, "stablehlo.divide"),
    ]
    for n, op in expected:
        if n is None or n.op != op:
            return None

    if _reduce_kind(rmax) != "max" or _reduce_kind(rsum) != "add":
        return None

    dims1 = _dims(rmax)
    dims2 = _dims(rsum)
    if dims1 != dims2 or len(dims1) != 1:
        return None
    axis = int(dims1[0])

    x_ssa = rmax.inputs[0]
    if x_ssa not in ctx.env:
        return None

    max_ssa = rmax.outputs[0]
    bmax_ssa = bmax.outputs[0]
    sub_ssa = sub.outputs[0]
    exp_ssa = exp.outputs[0]
    sum_ssa = rsum.outputs[0]
    bsum_ssa = bsum.outputs[0]

    # Verify actual dataflow, not just opcode sequence.
    if bmax.inputs[0] != max_ssa:
        return None
    if sub.inputs[0] != x_ssa or sub.inputs[1] != bmax_ssa:
        return None
    if exp.inputs[0] != sub_ssa:
        return None
    if rsum.inputs[0] != exp_ssa:
        return None
    if bsum.inputs[0] != sum_ssa:
        return None
    if div.inputs[0] != exp_ssa or div.inputs[1] != bsum_ssa:
        return None

    x = _get(ctx, x_ssa)
    rank = len(x.shape)
    actual_axis = _norm_axis(axis, rank)
    # Cactus compute_softmax_node currently ignores params.axis and always uses last axis.
    if actual_axis != rank - 1:
        return None

    out = ctx.graph.softmax(x, axis=axis)

    # Bind final output. Bind consumed intermediates only so future accidental uses don't KeyError;
    # strict dataflow above ensures this is a real softmax chain.
    for n in (rmax, bmax, sub, exp, rsum, bsum, div):
        if n and n.outputs:
            ctx.env[n.outputs[0]] = out

    return [out], 7


SOFTMAX = Pattern(name="softmax", handler=_softmax_pattern, trigger_ops={"stablehlo.reduce"})


DEFAULT_PATTERNS: list[Pattern] = [RELU, SOFTMAX]