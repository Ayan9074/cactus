"""
op_lowering_fixed.py
====================
Semantics-first StableHLO -> Cactus primitive lowering.

This is still not a complete StableHLO implementation. The important rule is:
if a StableHLO op cannot be represented faithfully by the current Cactus Graph
API, this file raises a precise NotImplementedError instead of silently emitting
an approximate graph.

Layering expectation:
  * Layer 1 here handles safe primitive lowerings.
  * Layer 2 patterns should fuse high-level structures that Cactus already has
    native ops for: softmax, rms_norm, rope, attention, MLP/MoE, etc.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
import itertools
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from parse import IRNode
    from pattern_registry import LoweringCtx


# ---------------------------------------------------------------------------
# dtype helpers
# ---------------------------------------------------------------------------

_DTYPE_MAP = {
    "f32": 2,   # Graph.FP32
    "f16": 1,   # Graph.FP16
    "bf16": 1,  # Cactus treats bf16 compute as fp16 today
    "i8": 0,
    "si8": 0,
    "ui8": 0,
    "i4": 3,
    "si4": 3,
    "ui4": 3,
    # Cactus does not have general i32/i64 tensors. These are mapped to FP32
    # only so simple index-like inputs can be carried through the FFI. Boolean
    # or arithmetic integer semantics are NOT implemented by this file.
    "i1": 2,
    "i32": 2,
    "si32": 2,
    "ui32": 2,
    "i64": 2,
    "si64": 2,
    "ui64": 2,
}

_FLOAT_DTYPES = {1, 2}  # Graph.FP16, Graph.FP32


def _cactus_dtype(dtype_str: str) -> int:
    return _DTYPE_MAP.get(dtype_str.strip(), 1)


def _dtype_from_type_str(type_str: str) -> str:
    m = re.match(r"tensor<(.+)>", type_str.strip())
    if m:
        return m.group(1).split("x")[-1]
    return type_str.strip()


def _shape_from_type_str(type_str: str) -> tuple[int, ...]:
    m = re.match(r"tensor<(.+)>", type_str.strip())
    if not m:
        return ()
    parts = m.group(1).split("x")
    try:
        return tuple(int(p) for p in parts[:-1])
    except ValueError:
        # Dynamic dimensions are not supported by this compiler path yet.
        return ()


def _result_shape(node: "IRNode") -> tuple[int, ...]:
    if not node.result_types:
        raise ValueError(f"{node.op} node {node.id} has no result_types")
    return _shape_from_type_str(node.result_types[0])


def _prod(xs) -> int:
    out = 1
    for x in xs:
        out *= int(x)
    return out


def _norm_axis(axis: int, rank: int) -> int:
    axis = int(axis)
    return axis + rank if axis < 0 else axis


def _permute_if_needed(ctx, x, perm: list[int] | tuple[int, ...]):
    perm = tuple(int(p) for p in perm)
    if perm == tuple(range(len(x.shape))):
        return x
    return ctx.graph.permute(x, perm)


def _reshape_if_needed(ctx, x, shape: tuple[int, ...] | list[int]):
    shape = tuple(int(s) for s in shape)
    if tuple(x.shape) == shape:
        return x
    # Cactus represents scalar tensors as shape [1]. StableHLO scalar shape is ().
    if shape == ():
        shape = (1,)
    return ctx.graph.reshape(x, shape)


# ---------------------------------------------------------------------------
# Lazy broadcast values
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BroadcastView:
    """A StableHLO broadcast that Cactus has not materialized.

    Cactus binary ops have implicit NumPy-style broadcasting. We reshape the
    operand into a rank-aligned storage shape with 1s in broadcasted axes, then
    let binary ops do the expansion. Non-binary consumers must not silently use
    the storage tensor because that would drop StableHLO's logical result shape.
    """

    tensor: Any
    logical_shape: tuple[int, ...]
    storage_shape: tuple[int, ...]
    broadcast_dimensions: tuple[int, ...]

    @property
    def shape(self) -> tuple[int, ...]:
        return self.logical_shape

    @property
    def dtype(self) -> int:
        return self.tensor.dtype


def _is_broadcast_view(x: Any) -> bool:
    return isinstance(x, BroadcastView)


def _logical_shape(x: Any) -> tuple[int, ...]:
    return tuple(x.logical_shape if _is_broadcast_view(x) else x.shape)


def _storage_tensor(x: Any) -> Any:
    return x.tensor if _is_broadcast_view(x) else x

_CURRENT_CTX = None


def _materialize_broadcast_view(ctx, bv: BroadcastView, consumer: str):
    storage = bv.tensor
    storage_shape = tuple(int(v) for v in bv.storage_shape)
    logical_shape = tuple(int(v) for v in bv.logical_shape)

    if tuple(storage.shape) != storage_shape:
        storage = _reshape_if_needed(ctx, storage, storage_shape)

    if storage_shape == logical_shape:
        return storage

    if len(storage_shape) != len(logical_shape):
        raise NotImplementedError(
            f"{consumer} needs materialized broadcast with rank mismatch: "
            f"storage_shape={storage_shape}, logical_shape={logical_shape}"
        )

    out = storage

    for axis, (s_dim, l_dim) in enumerate(zip(storage_shape, logical_shape)):
        s_dim = int(s_dim)
        l_dim = int(l_dim)

        if s_dim == l_dim:
            continue

        if s_dim != 1:
            raise NotImplementedError(
                f"{consumer} needs broadcast materialization but axis {axis} has "
                f"storage_dim={s_dim}, logical_dim={l_dim}; only singleton expansion supported."
            )

        # Keep it conservative so accidental huge broadcasts do not explode graph size.
        if l_dim > 64:
            raise NotImplementedError(
                f"{consumer} needs materializing broadcast axis={axis} repeat={l_dim}, "
                "which is too large for concat fallback. Fuse this pattern instead."
            )

        pieces = [out] * l_dim
        out = _cat_many(ctx, pieces, axis=axis)

    if tuple(out.shape) != logical_shape:
        out = _reshape_if_needed(ctx, out, logical_shape)

    return out


def _force_tensor(x: Any, consumer: str) -> Any:
    if not _is_broadcast_view(x):
        return x

    if tuple(x.logical_shape) == tuple(x.storage_shape):
        return x.tensor

    if consumer in ("reshape", "transpose", "slice", "concatenate", "dot_general lhs", "dot_general rhs"):
        if _CURRENT_CTX is None:
            raise NotImplementedError(
                f"{consumer} needs materialized broadcast but no ctx is available."
            )
        return _materialize_broadcast_view(_CURRENT_CTX, x, consumer)

    raise NotImplementedError(
        f"{consumer} consumes the result of a true stablehlo.broadcast_in_dim "
        f"with logical_shape={x.logical_shape} but Cactus only materializes "
        f"broadcasts inside binary ops. Add a Cactus broadcast_to op or fuse this "
        f"producer/consumer pattern."
    )


def _get(ctx, ssa: str) -> Any:
    if ssa not in ctx.env:
        raise KeyError(
            f"SSA value {ssa!r} not in env — its producer has not been lowered yet.\n"
            f"  Available: {sorted(ctx.env.keys())}"
        )
    return ctx.env[ssa]


def _ir_shape(ctx, ssa: str, fallback: tuple[int, ...]) -> tuple[int, ...]:
    try:
        return tuple(ctx.ir.values[ssa].shape)
    except Exception:
        return tuple(fallback)


# ---------------------------------------------------------------------------
# Attribute helpers
# ---------------------------------------------------------------------------

def _parse_int_listish(v: Any) -> list[int]:
    """Parse the list-ish attribute forms emitted by StableHLO/JAX parsers.

    JAX/StableHLO versions commonly print broadcast dims as either:
      * dims = [1]
      * broadcast_dimensions = array<i64: 1>
    The simple parser may preserve either spelling/value form.
    """
    if v is None:
        return []
    if isinstance(v, int):
        return [int(v)]
    if isinstance(v, tuple):
        raise TypeError(f"Expected a list-like integer attr, got x-product tuple: {v}")
    if isinstance(v, str):
        raw = v.strip()
        # Avoid reading the "64" in "i64" as a dimension.
        m = re.search(r"array<[^:>]+:\s*([^>]*)>", raw)
        if m:
            raw = m.group(1)
        elif raw.startswith("[") and raw.endswith("]"):
            raw = raw[1:-1]
        if not raw.strip():
            return []
        return [int(x) for x in re.findall(r"-?\d+", raw)]
    return [int(x) for x in v]


def _list_attr(node: "IRNode", *names: str, default=None) -> list[int]:
    for name in names:
        if name in node.attrs:
            return _parse_int_listish(node.attrs[name])
    return [] if default is None else list(default)


def _xprod_attr(node: "IRNode", name: str) -> tuple[list[int], list[int]] | None:
    v = node.attrs.get(name)
    if v is None:
        return None
    if isinstance(v, tuple) and len(v) == 2:
        return ([int(x) for x in v[0]], [int(x) for x in v[1]])
    raise TypeError(f"Expected {name} to parse as [..] x [..], got {v!r}")


def _dot_dimension_numbers(node: "IRNode", lhs_rank: int, rhs_rank: int):
    cdims = _xprod_attr(node, "contracting_dims")
    if cdims is None:
        lhs_contract = _list_attr(node, "lhs_contract", "lhs_contracting_dimensions")
        rhs_contract = _list_attr(node, "rhs_contract", "rhs_contracting_dimensions")
        if not lhs_contract and not rhs_contract:
            # stablehlo.dot / simple dot_general default used by simple matmul traces.
            lhs_contract = [lhs_rank - 1]
            rhs_contract = [0]
        cdims = (lhs_contract, rhs_contract)

    bdims = _xprod_attr(node, "batching_dims")
    if bdims is None:
        lhs_batch = _list_attr(node, "lhs_batch", "lhs_batching_dimensions")
        rhs_batch = _list_attr(node, "rhs_batch", "rhs_batching_dimensions")
        bdims = (lhs_batch, rhs_batch)

    lhs_contract, rhs_contract = cdims
    lhs_batch, rhs_batch = bdims
    lhs_contract = [_norm_axis(d, lhs_rank) for d in lhs_contract]
    rhs_contract = [_norm_axis(d, rhs_rank) for d in rhs_contract]
    lhs_batch = [_norm_axis(d, lhs_rank) for d in lhs_batch]
    rhs_batch = [_norm_axis(d, rhs_rank) for d in rhs_batch]
    return lhs_batch, rhs_batch, lhs_contract, rhs_contract


def _reducer_kind(node: "IRNode") -> str:
    body = str(node.attrs.get("applies", "stablehlo.add")).lower()
    if "maximum" in body or body.endswith("max") or ".max" in body:
        return "max"
    if "minimum" in body or body.endswith("min") or ".min" in body:
        return "min"
    if "multiply" in body or ".mul" in body:
        return "mul"
    return "add"


def _const_scalar(ctx, ssa: str) -> float | None:
    return _const_scalar_value(ctx, ssa)


# ---------------------------------------------------------------------------
# Dot/general matmul lowering
# ---------------------------------------------------------------------------

def _lower_dot_general(ctx, node):
    """Lower safe dot_general cases by canonicalizing to one 2D Cactus matmul.

    Supported:
      * normal 2D matmul
      * rank-N linear projections with no batching dimensions, e.g.
        [B,T,K] x [K,N] -> [B,T,N]
      * degenerate batched dot where batch product is 1

    Not supported here:
      * true batched matmul, e.g. attention QK/V with per-head/per-batch rhs.
        That should be fused into Cactus attention or implemented as a Cactus
        batched_matmul op.
    """
    lhs_v = _force_tensor(_get(ctx, node.inputs[0]), "dot_general lhs")
    rhs_v = _force_tensor(_get(ctx, node.inputs[1]), "dot_general rhs")
    lhs_shape = tuple(lhs_v.shape)
    rhs_shape = tuple(rhs_v.shape)
    expected_lhs_shape = _ir_shape(ctx, node.inputs[0], lhs_shape)
    expected_rhs_shape = _ir_shape(ctx, node.inputs[1], rhs_shape)

    if tuple(lhs_shape) != tuple(expected_lhs_shape):
        raise ValueError(
            f"dot_general lhs runtime shape != IR shape for {node.inputs[0]}: "
            f"runtime={lhs_shape}, ir={expected_lhs_shape}, node={node}"
        )

    if tuple(rhs_shape) != tuple(expected_rhs_shape):
        raise ValueError(
            f"dot_general rhs runtime shape != IR shape for {node.inputs[1]}: "
            f"runtime={rhs_shape}, ir={expected_rhs_shape}, node={node}"
        )
    lhs_rank = len(lhs_shape)
    rhs_rank = len(rhs_shape)

    lhs_batch, rhs_batch, lhs_contract, rhs_contract = _dot_dimension_numbers(node, lhs_rank, rhs_rank)

    if len(lhs_contract) != len(rhs_contract):
        raise NotImplementedError(f"dot_general mismatched contracting dims: lhs={lhs_contract}, rhs={rhs_contract}")
    if len(lhs_batch) != len(rhs_batch):
        raise NotImplementedError(f"dot_general mismatched batching dims: lhs={lhs_batch}, rhs={rhs_batch}")

    lhs_contract_size = _prod(lhs_shape[d] for d in lhs_contract)
    rhs_contract_size = _prod(rhs_shape[d] for d in rhs_contract)
    if lhs_contract_size != rhs_contract_size:
        raise ValueError(
            f"dot_general contract size mismatch: lhs dims {lhs_contract} size {lhs_contract_size}, "
            f"rhs dims {rhs_contract} size {rhs_contract_size}"
        )

    lhs_batch_shape = tuple(lhs_shape[d] for d in lhs_batch)
    rhs_batch_shape = tuple(rhs_shape[d] for d in rhs_batch)
    if lhs_batch_shape != rhs_batch_shape:
        raise ValueError(f"dot_general batch shape mismatch: lhs={lhs_batch_shape}, rhs={rhs_batch_shape}")
    batch_size = _prod(lhs_batch_shape)

    lhs_free = [d for d in range(lhs_rank) if d not in set(lhs_batch + lhs_contract)]
    rhs_free = [d for d in range(rhs_rank) if d not in set(rhs_batch + rhs_contract)]
    lhs_free_shape = tuple(lhs_shape[d] for d in lhs_free)
    rhs_free_shape = tuple(rhs_shape[d] for d in rhs_free)

    if batch_size != 1:
        return [_lower_dot_general_unrolled_batched(
            ctx=ctx,
            node=node,
            lhs_v=lhs_v,
            rhs_v=rhs_v,
            lhs_shape=lhs_shape,
            rhs_shape=rhs_shape,
            lhs_batch=lhs_batch,
            rhs_batch=rhs_batch,
            lhs_contract=lhs_contract,
            rhs_contract=rhs_contract,
            lhs_free=lhs_free,
            rhs_free=rhs_free,
            lhs_batch_shape=lhs_batch_shape,
            rhs_batch_shape=rhs_batch_shape,
            lhs_free_shape=lhs_free_shape,
            rhs_free_shape=rhs_free_shape,
            lhs_contract_size=lhs_contract_size,
        )]

    lhs_perm = lhs_batch + lhs_free + lhs_contract
    rhs_perm = rhs_batch + rhs_contract + rhs_free
    lhs_t = _permute_if_needed(ctx, lhs_v, lhs_perm)
    rhs_t = _permute_if_needed(ctx, rhs_v, rhs_perm)

    m = _prod(lhs_batch_shape) * _prod(lhs_free_shape)
    k = lhs_contract_size
    n = _prod(rhs_free_shape)
    lhs_2d = _reshape_if_needed(ctx, lhs_t, (m, k))
    rhs_2d = _reshape_if_needed(ctx, rhs_t, (k, n))
    out_2d = ctx.graph.matmul(lhs_2d, rhs_2d)

    out_shape = _result_shape(node)
    if not out_shape:
        out_shape = lhs_batch_shape + lhs_free_shape + rhs_free_shape
    if not out_shape:
        out_shape = (1,)
    return [_reshape_if_needed(ctx, out_2d, out_shape)]


def _cat_many(ctx, tensors: list[Any], axis: int):
    """Concatenate a non-empty list of tensors along axis using Cactus cat/concat."""
    if not tensors:
        raise ValueError("_cat_many requires at least one tensor")
    if len(tensors) == 1:
        return tensors[0]
    if hasattr(ctx.graph, "cat"):
        return ctx.graph.cat(tensors, axis=axis)
    out = tensors[0]
    for t in tensors[1:]:
        out = ctx.graph.concat(out, t, axis=axis)
    return out


def _slice_batch_prefix(ctx, x, batch_indices: tuple[int, ...]):
    """Slice canonicalized tensor on its leading batch dims, preserving rank."""
    out = x
    for axis, idx in enumerate(batch_indices):
        out = ctx.graph.slice(out, axis, int(idx), 1)
    return out


def _reassemble_batch_pieces(ctx, pieces: dict[tuple[int, ...], Any], batch_shape: tuple[int, ...]):
    """
    Reassemble singleton-prefixed pieces into a full batched tensor.

    Each piece has shape [1, 1, ..., *free_dims]. We concatenate from the
    innermost batch dimension outward so the final shape is
    [*batch_shape, *free_dims].
    """
    if not batch_shape:
        return pieces[()]

    def build(dim: int, prefix: tuple[int, ...]):
        if dim == len(batch_shape):
            return pieces[prefix]
        children = [build(dim + 1, prefix + (i,)) for i in range(int(batch_shape[dim]))]
        return _cat_many(ctx, children, axis=dim)

    return build(0, ())


def _lower_dot_general_unrolled_batched(
    *,
    ctx,
    node,
    lhs_v,
    rhs_v,
    lhs_shape: tuple[int, ...],
    rhs_shape: tuple[int, ...],
    lhs_batch: list[int],
    rhs_batch: list[int],
    lhs_contract: list[int],
    rhs_contract: list[int],
    lhs_free: list[int],
    rhs_free: list[int],
    lhs_batch_shape: tuple[int, ...],
    rhs_batch_shape: tuple[int, ...],
    lhs_free_shape: tuple[int, ...],
    rhs_free_shape: tuple[int, ...],
    lhs_contract_size: int,
):
    """
    Slow correctness-only lowering for true batched StableHLO dot_general.

    StableHLO dot_general with batching means each batch slice gets its own
    matmul. Cactus currently exposes only 2D matmul, so this lowers a static
    batched matmul by:

      1. Canonicalize lhs to [B..., M..., K...]
      2. Canonicalize rhs to [B..., K..., N...]
      3. For every static batch index, slice lhs/rhs, reshape to 2D, matmul
      4. Reshape each result to [1..., M..., N...]
      5. Concatenate pieces back across batch axes

    This is intentionally slow and graph-size-heavy. It is for validation and
    should later be replaced by Cactus batched_matmul or fused attention.
    """
    if lhs_batch_shape != rhs_batch_shape:
        raise ValueError(f"dot_general batch shape mismatch: lhs={lhs_batch_shape}, rhs={rhs_batch_shape}")

    batch_size = _prod(lhs_batch_shape)
    max_unroll = int(getattr(ctx, "max_unrolled_batched_matmul", 256))
    if batch_size > max_unroll:
        raise NotImplementedError(
            f"Refusing to unroll dot_general with batch_size={batch_size} > {max_unroll}. "
            "Use fused attention or add a real Cactus batched_matmul op."
        )

    lhs_perm = lhs_batch + lhs_free + lhs_contract
    rhs_perm = rhs_batch + rhs_contract + rhs_free
    lhs_t = _permute_if_needed(ctx, lhs_v, lhs_perm)
    rhs_t = _permute_if_needed(ctx, rhs_v, rhs_perm)

    m = _prod(lhs_free_shape)
    k = int(lhs_contract_size)
    n = _prod(rhs_free_shape)

    piece_shape = tuple([1] * len(lhs_batch_shape)) + tuple(lhs_free_shape) + tuple(rhs_free_shape)
    if not piece_shape:
        piece_shape = (1,)

    pieces: dict[tuple[int, ...], Any] = {}
    for batch_idx in itertools.product(*[range(int(d)) for d in lhs_batch_shape]):
        lhs_slice = _slice_batch_prefix(ctx, lhs_t, batch_idx)
        rhs_slice = _slice_batch_prefix(ctx, rhs_t, batch_idx)

        lhs_2d = _reshape_if_needed(ctx, lhs_slice, (m, k))
        rhs_2d = _reshape_if_needed(ctx, rhs_slice, (k, n))
        out_2d = ctx.graph.matmul(lhs_2d, rhs_2d)

        pieces[tuple(int(i) for i in batch_idx)] = _reshape_if_needed(ctx, out_2d, piece_shape)

    out = _reassemble_batch_pieces(ctx, pieces, tuple(lhs_batch_shape))

    out_shape = _result_shape(node)
    if not out_shape:
        out_shape = tuple(lhs_batch_shape) + tuple(lhs_free_shape) + tuple(rhs_free_shape)
    if not out_shape:
        out_shape = (1,)

    return _reshape_if_needed(ctx, out, out_shape)




















def _compare_direction(node):
    raw = (
        node.attrs.get("comparison_direction")
        or node.attrs.get("direction")
        or node.attrs.get("compare_direction")
    )

    if raw is None:
        raise NotImplementedError(
            f"stablehlo.compare missing comparison direction. attrs={node.attrs}"
        )

    s = str(raw).upper()
    for d in ("EQ", "NE", "LT", "LE", "GT", "GE"):
        if d == s or d in s:
            return d

    raise NotImplementedError(f"Unsupported compare direction: {raw!r}")


def _one_minus(ctx, x):
    # 1 - x, using scalar ops so we do not need to create a constant tensor.
    return ctx.graph.scalar_add(ctx.graph.scalar_multiply(x, -1.0), 1.0)






def _lower_compare(ctx, node):
    a, b = _binary_args(ctx, node, "compare")
    direction = _compare_direction(node)
    return [ctx.graph.compare(a, b, direction)]


def _lower_and(ctx, node):
    a, b = _binary_args(ctx, node, "and")
    # For 0/1 numeric masks, logical AND = multiply.
    return [ctx.graph.multiply(a, b)]


def _lower_or(ctx, node):
    a, b = _binary_args(ctx, node, "or")
    # For 0/1 masks: OR = a + b - a*b
    ab = ctx.graph.multiply(a, b)
    return [ctx.graph.subtract(ctx.graph.add(a, b), ab)]


def _lower_not(ctx, node):
    x = _storage_tensor(_get(ctx, node.inputs[0]))
    return [_one_minus(ctx, x)]

def _lower_select(ctx, node):
    # stablehlo.select(mask, true_value, false_value)
    mask = _storage_tensor(_get(ctx, node.inputs[0]))
    true_value = _storage_tensor(_get(ctx, node.inputs[1]))
    false_value = _storage_tensor(_get(ctx, node.inputs[2]))

    # Cactus select can fail on dtype/shape edge cases, especially because
    # StableHLO bool/int masks are carried as numeric tensors in this path.
    # Fallback: select(m, t, f) = m*t + (1-m)*f for numeric 0/1 masks.
    try:
        return [ctx.graph.select(mask, true_value, false_value)]
    except Exception:
        target_dtype = 2 if (
            getattr(true_value, "dtype", 1) == 2 or getattr(false_value, "dtype", 1) == 2
        ) else 1

        if getattr(mask, "dtype", 1) != target_dtype:
            mask = ctx.graph.precision_cast(mask, target_dtype)

        if getattr(true_value, "dtype", 1) != target_dtype:
            true_value = ctx.graph.precision_cast(true_value, target_dtype)

        if getattr(false_value, "dtype", 1) != target_dtype:
            false_value = ctx.graph.precision_cast(false_value, target_dtype)

        mt = ctx.graph.multiply(mask, true_value)
        one_minus_m = _one_minus(ctx, mask)
        mf = ctx.graph.multiply(one_minus_m, false_value)
        return [ctx.graph.add(mt, mf)]





def _op_name(n):
    return getattr(n, "op", "")


def _node_outputs(n):
    return list(getattr(n, "outputs", []) or [])


def _node_inputs(n):
    return list(getattr(n, "inputs", []) or [])


def _node_by_output(ctx):
    """
    Build SSA output -> producing node map lazily.
    Works with common IR layouts:
      ctx.ir.order + ctx.ir.nodes
      ctx.nodes / ctx.order
    """
    cached = getattr(ctx, "_node_by_output_cache", None)
    if cached is not None:
        return cached

    mapping = {}

    ir = getattr(ctx, "ir", None)
    if ir is not None and hasattr(ir, "order") and hasattr(ir, "nodes"):
        for nid in ir.order:
            n = ir.nodes[nid]
            for out in _node_outputs(n):
                mapping[out] = n

    elif hasattr(ctx, "order") and hasattr(ctx, "nodes"):
        for nid in ctx.order:
            n = ctx.nodes[nid]
            for out in _node_outputs(n):
                mapping[out] = n

    setattr(ctx, "_node_by_output_cache", mapping)
    return mapping


def _producer(ctx, ssa):
    return _node_by_output(ctx).get(ssa)


def _strip_unary_wrappers(ctx, ssa, ops):
    """
    Walk backwards through ops that preserve semantic value/shape enough
    for pattern matching, e.g. convert/broadcast.
    """
    while True:
        n = _producer(ctx, ssa)
        if n is None or _op_name(n) not in ops:
            return ssa, n

        ins = _node_inputs(n)
        if len(ins) != 1:
            return ssa, n

        ssa = ins[0]


def _get_single_axis_from_reduce(node):
    dims = (
        node.attrs.get("dimensions")
        or node.attrs.get("dims")
        or node.attrs.get("axes")
        or node.attrs.get("axis")
    )

    if dims is None:
        return None

    vals = _parse_int_listish(dims)
    if len(vals) != 1:
        return None

    return int(vals[0])


def _reduce_applies(node):
    return str(node.attrs.get("applies", "")).strip()


def _trace_to_reduce(ctx, ssa):
    """
    For denominator path:
      broadcast -> convert -> broadcast -> reduce
    Return reduce node if found.
    """
    cur = ssa

    for _ in range(8):
        n = _producer(ctx, cur)
        if n is None:
            return None

        if _op_name(n) == "stablehlo.reduce":
            return n

        if _op_name(n) in (
            "stablehlo.broadcast_in_dim",
            "stablehlo.convert",
        ):
            ins = _node_inputs(n)
            if len(ins) != 1:
                return None
            cur = ins[0]
            continue

        return None

    return None


def _trace_to_exp(ctx, ssa):
    """
    Follow convert/broadcast wrappers back until exponential if possible.
    """
    cur = ssa

    for _ in range(8):
        n = _producer(ctx, cur)
        if n is None:
            return None

        if _op_name(n) in ("stablehlo.exponential", "stablehlo.exp"):
            return n

        if _op_name(n) in (
            "stablehlo.convert",
            "stablehlo.broadcast_in_dim",
        ):
            ins = _node_inputs(n)
            if len(ins) != 1:
                return None
            cur = ins[0]
            continue

        return None

    return None


def _match_decomposed_softmax_divide(ctx, node):
    """
    Match JAX/StableHLO decomposed softmax ending at divide:

        x_max = reduce_max(x, axis)
        shifted = x - broadcast(x_max)
        e = exp(shifted)
        denom = reduce_sum(convert(e), axis)
        out = e / broadcast(convert(denom))

    Return:
        (input_ssa, axis)
    or:
        None
    """
    if _op_name(node) != "stablehlo.divide":
        return None

    ins = _node_inputs(node)
    if len(ins) != 2:
        return None

    lhs_ssa, rhs_ssa = ins

    # Numerator should be exp(subtract(x, broadcast(max_reduce(...)))).
    exp_n = _producer(ctx, lhs_ssa)
    if exp_n is None or _op_name(exp_n) not in ("stablehlo.exponential", "stablehlo.exp"):
        return None

    exp_inputs = _node_inputs(exp_n)
    if len(exp_inputs) != 1:
        return None

    sub_n = _producer(ctx, exp_inputs[0])
    if sub_n is None or _op_name(sub_n) != "stablehlo.subtract":
        return None

    sub_inputs = _node_inputs(sub_n)
    if len(sub_inputs) != 2:
        return None

    input_ssa = sub_inputs[0]

    # Denominator should trace to reduce add.
    reduce_n = _trace_to_reduce(ctx, rhs_ssa)
    if reduce_n is None:
        return None

    applies = _reduce_applies(reduce_n)
    if applies and "add" not in applies:
        return None

    axis = _get_single_axis_from_reduce(reduce_n)
    if axis is None:
        return None

    # Optional sanity: reduce input should trace back to same exp.
    reduce_inputs = _node_inputs(reduce_n)
    if len(reduce_inputs) == 1:
        denom_exp = _trace_to_exp(ctx, reduce_inputs[0])
        if denom_exp is not None and _node_outputs(denom_exp) != _node_outputs(exp_n):
            return None

    return input_ssa, axis


def _canonical_softmax_axis(ctx, x, axis):
    """
    Cactus softmax is safest over the last axis.
    For non-last-axis softmax:
      permute target axis to last
      softmax last
      permute back
    """
    x = _force_tensor(x, "softmax")

    rank = len(x.shape)
    if rank == 0:
        raise ValueError("softmax on scalar is not supported")

    axis = int(axis)
    if axis < 0:
        axis += rank

    if axis < 0 or axis >= rank:
        raise ValueError(f"softmax axis out of range: axis={axis}, shape={tuple(x.shape)}")

    if axis == rank - 1:
        return ctx.graph.softmax(x, axis=-1)

    perm = [i for i in range(rank) if i != axis] + [axis]

    inv_perm = [0] * rank
    for new_i, old_i in enumerate(perm):
        inv_perm[old_i] = new_i

    y = ctx.graph.permute(x, perm)
    y = ctx.graph.softmax(y, axis=-1)
    y = ctx.graph.permute(y, inv_perm)

    return y
# ---------------------------------------------------------------------------
# Elementwise/binary ops
# ---------------------------------------------------------------------------

def _binary_args(ctx, node, opname: str):
    a = _storage_tensor(_get(ctx, node.inputs[0]))
    b = _storage_tensor(_get(ctx, node.inputs[1]))
    return a, b


def _lower_add(ctx, node):
    a, b = _binary_args(ctx, node, "add")
    return [ctx.graph.add(a, b)]


def _lower_subtract(ctx, node):
    a, b = _binary_args(ctx, node, "subtract")
    return [ctx.graph.subtract(a, b)]

def _lower_multiply(ctx, node):
    a, b = _binary_args(ctx, node, "multiply")

    try:
        return [ctx.graph.multiply(a, b)]
    except Exception as e:
        def shape_of(x):
            return tuple(getattr(x, "shape", ()))

        def dtype_of(x):
            return getattr(x, "dtype", None)

        expected = None
        try:
            expected = _result_shape(node)
        except Exception:
            pass

        raise RuntimeError(
            "\nFAILED stablehlo.multiply\n"
            f"  outputs: {getattr(node, 'outputs', None)}\n"
            f"  inputs:  {getattr(node, 'inputs', None)}\n"
            f"  attrs:   {getattr(node, 'attrs', None)}\n"
            f"  result_types: {getattr(node, 'result_types', None)}\n"
            f"  expected result shape: {expected}\n"
            f"  lhs shape: {shape_of(a)} dtype={dtype_of(a)}\n"
            f"  rhs shape: {shape_of(b)} dtype={dtype_of(b)}\n"
            f"  original error: {type(e).__name__}: {e}"
        ) from e

def _match_reduce_sum_divide_constant(ctx, node):
    """
    Match:
        divide(reduce_add(x, axis), constant)

    Used by Gemma RMSNorm:
        sum(x*x, axis=-1) / hidden_dim

    Lowering this as mean(x, axis) avoids materializing the FP16 sum,
    which can overflow for hidden_dim=2304.
    """
    if getattr(node, "op", "") != "stablehlo.divide":
        return None

    if len(node.inputs) != 2:
        return None

    lhs_ssa, rhs_ssa = node.inputs

    reduce_node = _producer(ctx, lhs_ssa)
    if reduce_node is None or _op_name(reduce_node) != "stablehlo.reduce":
        return None

    applies = _reduce_applies(reduce_node)
    if applies and "add" not in applies:
        return None

    axis = _get_single_axis_from_reduce(reduce_node)
    if axis is None:
        return None

    denom = _const_scalar_value(ctx, rhs_ssa)
    if denom is None:
        return None

    reduce_inputs = _node_inputs(reduce_node)
    if len(reduce_inputs) < 1:
        return None

    x_ssa = reduce_inputs[0]

    try:
        x_shape = tuple(ctx.ir.values[x_ssa].shape)
    except Exception:
        x_shape = tuple(getattr(_get(ctx, x_ssa), "shape", ()))

    rank = len(x_shape)
    if axis < 0:
        axis += rank

    if axis < 0 or axis >= rank:
        return None

    axis_size = float(x_shape[axis])

    # Only rewrite true mean pattern.
    if abs(float(denom) - axis_size) > 1e-3:
        return None

    return x_ssa, axis

def _lower_divide(ctx, node):
    # Peephole: reduce_sum(x, axis) / axis_size -> mean(x, axis)
    m_mean = _match_reduce_sum_divide_constant(ctx, node)
    if m_mean is not None:
        x_ssa, axis = m_mean
        x = _force_tensor(_get(ctx, x_ssa), "reduce-sum-divide mean")
        return [ctx.graph.mean(x, axis)]

    # Existing softmax matcher
    m = _match_decomposed_softmax_divide(ctx, node)
    if m is not None:
        input_ssa, axis = m
        print("OP_LOWERING SOFTMAX FUSED", node.id, node.outputs, "axis", axis)
        x = _get(ctx, input_ssa)
        return [_canonical_softmax_axis(ctx, x, axis)]

    a, b = _binary_args(ctx, node, "divide")
    return [ctx.graph.divide(a, b)]


def _lower_cactus_mean_keepdims(ctx, node):
    x = _force_tensor(_get(ctx, node.inputs[0]), "cactus.mean_keepdims")

    axis = int(node.attrs["axis"])
    if axis < 0:
        axis += len(x.shape)

    y = ctx.graph.mean(x, axis)

    out_shape = tuple(int(v) for v in node.attrs["out_shape"])
    if out_shape and tuple(y.shape) != out_shape:
        y = ctx.graph.reshape(y, out_shape)

    return [y]

def _lower_cactus_mean(ctx, node):
    x = _force_tensor(_get(ctx, node.inputs[0]), "cactus.mean")
    axis = int(node.attrs["axis"])
    if axis < 0:
        axis += len(x.shape)
    return [ctx.graph.mean(x, axis)]
def _decode_scalar_literal(value, dtype=None):
    """Decode StableHLO scalar literals, including raw float bit patterns.

    StableHLO/JAX can print f16 infinities as hex, e.g. 0xFC00 for -inf.
    This local decoder keeps op_lowering correct even if lower_to_cactus has not
    attached ctx.const_values yet.
    """
    try:
        import numpy as np
    except Exception:
        np = None

    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass

    if isinstance(value, str):
        s = value.strip()
        low = s.lower()
        if low in ("inf", "+inf", "infinity", "+infinity"):
            return math.inf
        if low in ("-inf", "-infinity"):
            return -math.inf
        if low in ("nan", "+nan", "-nan"):
            return math.nan
        if low.startswith("0x") and np is not None:
            bits = int(low, 16)
            dtype_s = str(dtype or "").lower()
            if "f16" in dtype_s or bits <= 0xFFFF:
                return float(np.array([bits & 0xFFFF], dtype=np.uint16).view(np.float16)[0])
            if "bf16" in dtype_s:
                return float(np.array([(bits & 0xFFFF) << 16], dtype=np.uint32).view(np.float32)[0])
            if "f32" in dtype_s:
                return float(np.array([bits & 0xFFFFFFFF], dtype=np.uint32).view(np.float32)[0])
            if "f64" in dtype_s:
                return float(np.array([bits], dtype=np.uint64).view(np.float64)[0])
        return float(s)

    return float(value)


def _const_scalar_value(ctx, ssa):
    """Return decoded scalar constant value for an SSA name, if known.

    This checks both ctx.const_values and ctx.constants, and it also works for
    constants propagated through simple shape ops by this file.
    """
    vals = getattr(ctx, "const_values", None)
    if vals and ssa in vals:
        try:
            return float(vals[ssa])
        except Exception:
            return None

    constants = getattr(ctx, "constants", None)
    c = constants.get(ssa) if constants else None
    if c is None:
        return None
    v = c.value
    if isinstance(v, (list, tuple)):
        if len(v) != 1:
            return None
        v = v[0]
    try:
        return _decode_scalar_literal(v, getattr(c, "dtype", None))
    except Exception:
        return None

def _lower_maximum(ctx, node):
    a_ssa, b_ssa = node.inputs
    a, b = _binary_args(ctx, node, "maximum")

    av = _const_scalar_value(ctx, a_ssa)
    bv = _const_scalar_value(ctx, b_ssa)

    # Exact/simple shortcuts for infinities.
    if av is not None and math.isinf(av) and av < 0:
        return [b]
    if bv is not None and math.isinf(bv) and bv < 0:
        return [a]

    if av is not None and math.isinf(av) and av > 0:
        return [a]
    if bv is not None and math.isinf(bv) and bv > 0:
        return [b]

    # max(a, b) = select(a >= b, a, b)
    # This avoids inf arithmetic like -inf + inf -> NaN.
    mask = ctx.graph.compare(a, b, "GE")
    return [ctx.graph.select(mask, a, b)]


def _lower_minimum(ctx, node):
    a_ssa, b_ssa = node.inputs
    a, b = _binary_args(ctx, node, "minimum")

    av = _const_scalar_value(ctx, a_ssa)
    bv = _const_scalar_value(ctx, b_ssa)

    if av is not None and math.isinf(av) and av > 0:
        return [b]
    if bv is not None and math.isinf(bv) and bv > 0:
        return [a]

    if av is not None and math.isinf(av) and av < 0:
        return [a]
    if bv is not None and math.isinf(bv) and bv < 0:
        return [b]

    # min(a, b) = select(a <= b, a, b)
    mask = ctx.graph.compare(a, b, "LE")
    return [ctx.graph.select(mask, a, b)]

def _lower_negate(ctx, node):
    return [ctx.graph.scalar_multiply(_force_tensor(_get(ctx, node.inputs[0]), "negate"), -1.0)]


def _lower_abs(ctx, node):
    return [ctx.graph.abs(_force_tensor(_get(ctx, node.inputs[0]), "abs"))]


def _lower_sqrt(ctx, node):
    return [ctx.graph.scalar_sqrt(_force_tensor(_get(ctx, node.inputs[0]), "sqrt"))]


def _lower_rsqrt(ctx, node):
    return [ctx.graph.pow(_force_tensor(_get(ctx, node.inputs[0]), "rsqrt"), -0.5)]


def _lower_exp(ctx, node):
    return [ctx.graph.scalar_exp(_force_tensor(_get(ctx, node.inputs[0]), "exp"))]


def _lower_log(ctx, node):
    return [ctx.graph.scalar_log(_force_tensor(_get(ctx, node.inputs[0]), "log"))]


def _lower_sin(ctx, node):
    return [ctx.graph.scalar_sin(_force_tensor(_get(ctx, node.inputs[0]), "sine"))]


def _lower_cos(ctx, node):
    return [ctx.graph.scalar_cos(_force_tensor(_get(ctx, node.inputs[0]), "cosine"))]


def _lower_tanh(ctx, node):
    return [ctx.graph.tanh(_force_tensor(_get(ctx, node.inputs[0]), "tanh"))]


def _lower_logistic(ctx, node):
    return [ctx.graph.sigmoid(_force_tensor(_get(ctx, node.inputs[0]), "logistic"))]


def _lower_relu(ctx, node):
    return [ctx.graph.relu(_force_tensor(_get(ctx, node.inputs[0]), "relu"))]


def _lower_silu(ctx, node):
    return [ctx.graph.silu(_force_tensor(_get(ctx, node.inputs[0]), "silu"))]


def _lower_gelu(ctx, node):
    return [ctx.graph.gelu(_force_tensor(_get(ctx, node.inputs[0]), "gelu"))]


def _lower_square(ctx, node):
    x = _force_tensor(_get(ctx, node.inputs[0]), "square")
    return [ctx.graph.multiply(x, x)]




# ---------------------------------------------------------------------------
# Shape/layout ops
# ---------------------------------------------------------------------------

def _propagate_const(ctx, src_ssa: str, dst_ssa: str):
    v = _const_scalar_value(ctx, src_ssa)
    if v is not None:
        vals = getattr(ctx, "const_values", None)
        if vals is None:
            ctx.const_values = {}
            vals = ctx.const_values
        vals[dst_ssa] = v


def _lower_reshape(ctx, node):
    shape = _result_shape(node)
    if not shape:
        shape = (1,)
    if node.outputs:
        _propagate_const(ctx, node.inputs[0], node.outputs[0])
    return [_reshape_if_needed(ctx, _force_tensor(_get(ctx, node.inputs[0]), "reshape"), shape)]

def _lower_transpose(ctx, node):
    x = _force_tensor(_get(ctx, node.inputs[0]), "transpose")

    raw_perm = (
        node.attrs.get("permutation")
        or node.attrs.get("dimensions")
        or node.attrs.get("dims")
    )

    if raw_perm is None:
        raise NotImplementedError(
            f"stablehlo.transpose missing permutation/dims. "
            f"attrs={node.attrs}, input_shape={tuple(x.shape)}, result_types={node.result_types}"
        )

    perm = tuple(_parse_int_listish(raw_perm))

    if len(perm) != len(x.shape):
        raise ValueError(
            f"stablehlo.transpose permutation rank mismatch: "
            f"input_shape={tuple(x.shape)}, permutation={perm}, attrs={node.attrs}"
        )

    out = _permute_if_needed(ctx, x, perm)

    # Optional but useful for debugging: make sure lowered shape matches IR shape.
    expected = _result_shape(node)
    if expected and tuple(out.shape) != tuple(expected):
        raise ValueError(
            f"transpose lowered to wrong shape: input_shape={tuple(x.shape)}, "
            f"perm={perm}, got={tuple(out.shape)}, expected={expected}, attrs={node.attrs}"
        )

    if node.outputs:
        _propagate_const(ctx, node.inputs[0], node.outputs[0])

    return [out]


def _lower_broadcast_in_dim(ctx, node):
    ssa = node.inputs[0]
    if node.outputs:
        _propagate_const(ctx, ssa, node.outputs[0])
    x = _force_tensor(_get(ctx, ssa), "broadcast_in_dim operand")
    out_shape = _result_shape(node)
    if not out_shape:
        # StableHLO scalar. Cactus scalar-like values are shape [1].
        return [_reshape_if_needed(ctx, x, (1,))]

    sem_in_shape = _ir_shape(ctx, ssa, tuple(x.shape))
    # StableHLO usually calls this `broadcast_dimensions`, but older JAX text
    # often prints the same attribute as `dims`. Support both.
    raw_dims = node.attrs.get("broadcast_dimensions", node.attrs.get("dims", []))
    dims = tuple(_parse_int_listish(raw_dims))

    if len(dims) != len(sem_in_shape):
        # Some parsers lose scalar shape and store scalar constants as [1].
        if len(dims) == 0 and _prod(x.shape) == 1:
            sem_in_shape = ()
        else:
            raise ValueError(
                f"broadcast_in_dim malformed: operand semantic shape={sem_in_shape}, "
                f"broadcast_dimensions={dims}, out_shape={out_shape}"
            )

    if tuple(x.shape) == out_shape:
        return [x]

    aligned = [1] * len(out_shape)
    for i, target_axis in enumerate(dims):
        if not (0 <= target_axis < len(out_shape)):
            raise ValueError(f"broadcast_in_dim target axis out of range: {target_axis} for {out_shape}")
        dim = int(sem_in_shape[i])
        out_dim = int(out_shape[target_axis])
        if dim != 1 and dim != out_dim:
            raise ValueError(
                f"broadcast_in_dim dim mismatch: operand dim {i} has size {dim}, "
                f"target axis {target_axis} has size {out_dim}"
            )
        aligned[target_axis] = dim

    aligned_shape = tuple(aligned)
    if _prod(x.shape) != _prod(aligned_shape):
        raise NotImplementedError(
            f"Cannot rank-align broadcast operand with reshape: tensor_shape={x.shape}, "
            f"semantic_shape={sem_in_shape}, aligned_shape={aligned_shape}."
        )
    storage = _reshape_if_needed(ctx, x, aligned_shape)
    if aligned_shape == out_shape:
        return [storage]
    return [BroadcastView(storage, tuple(out_shape), aligned_shape, dims)]


def _lower_convert(ctx, node):
    if not node.result_types:
        return [_get(ctx, node.inputs[0])]

    dtype_str = _dtype_from_type_str(node.result_types[0])
    target = _cactus_dtype(dtype_str)
    x = _force_tensor(_get(ctx, node.inputs[0]), "convert")
    src_dtype = x.dtype

    if src_dtype in _FLOAT_DTYPES and target in _FLOAT_DTYPES:
        return [x]
    return [ctx.graph.precision_cast(x, target)]


# ---------------------------------------------------------------------------
# Reductions/indexing/concat
# ---------------------------------------------------------------------------

def _lower_reduce(ctx, node):
    x = _force_tensor(_get(ctx, node.inputs[0]), "reduce")
    dims = node.attrs.get("dimensions", [-1])
    if isinstance(dims, int):
        dims = [dims]
    dims = [_norm_axis(int(d), len(x.shape)) for d in dims]
    if len(set(dims)) != len(dims):
        raise ValueError(f"reduce dimensions must be unique, got {dims}")
    if not dims:
        return [x]

    kind = _reducer_kind(node)
    if kind == "mul":
        raise NotImplementedError("stablehlo.reduce with multiply/product body is not supported by Cactus Graph yet")

    result = x
    for axis in sorted(dims, reverse=True):
        if kind == "add":
            result = ctx.graph.sum(result, axis)
        elif kind == "max":
            result = ctx.graph.max(result, axis)
        elif kind == "min":
            result = ctx.graph.min(result, axis)
        else:
            raise NotImplementedError(f"Unsupported reduce body kind: {kind}")
    return [result]


def _attr_int_list(node, *names):
    for name in names:
        if name in node.attrs:
            return _parse_int_listish(node.attrs[name])
    return []


def _lower_slice(ctx, node):
    x = _force_tensor(_get(ctx, node.inputs[0]), "slice")

    starts = _attr_int_list(
        node,
        "start_indices",
        "start",
        "starts",
        "start_indices_attr",
    )
    limits = _attr_int_list(
        node,
        "limit_indices",
        "limit",
        "limits",
        "limit_indices_attr",
    )
    strides = _attr_int_list(
        node,
        "strides",
        "stride",
        "slice_strides",
    )

    if not starts or not limits:
        raise NotImplementedError(
            f"stablehlo.slice missing parsed starts/limits. "
            f"attrs={node.attrs}, input_shape={tuple(x.shape)}"
        )

    rank = len(x.shape)

    if len(starts) != rank or len(limits) != rank:
        raise ValueError(
            f"stablehlo.slice rank mismatch: input_shape={tuple(x.shape)}, "
            f"starts={starts}, limits={limits}, strides={strides}, attrs={node.attrs}"
        )

    if not strides:
        strides = [1] * rank

    if len(strides) != rank:
        raise ValueError(
            f"stablehlo.slice stride rank mismatch: input_shape={tuple(x.shape)}, "
            f"strides={strides}, attrs={node.attrs}"
        )

    result = x

    for axis, (s, l, stride) in enumerate(zip(starts, limits, strides)):
        s = int(s)
        l = int(l)
        stride = int(stride)

        if stride != 1:
            raise NotImplementedError(
                f"stablehlo.slice with stride={stride} on axis={axis} "
                f"needs a strided-slice op"
            )

        length = l - s
        dim_size = result.shape[axis]

        if s == 0 and length == dim_size:
            continue

        result = ctx.graph.slice(result, axis, s, length)

    _propagate_const(ctx, node.inputs[0], node.outputs[0])
    return [result]


def _lower_concatenate(ctx, node):
    xs = [_force_tensor(_get(ctx, inp), "concatenate") for inp in node.inputs]

    raw_axis = (
        node.attrs.get("dimension")
        or node.attrs.get("dimensions")
        or node.attrs.get("dim")
        or node.attrs.get("axis")
    )

    if raw_axis is None:
        raise NotImplementedError(
            f"stablehlo.concatenate missing dimension/dim attr. "
            f"attrs={node.attrs}, inputs={[tuple(x.shape) for x in xs]}, "
            f"result_types={node.result_types}"
        )

    if isinstance(raw_axis, (list, tuple)):
        if len(raw_axis) != 1:
            raise ValueError(f"concatenate axis malformed: {raw_axis}")
        axis = int(raw_axis[0])
    else:
        axis = int(raw_axis)

    rank = len(xs[0].shape)
    if axis < 0:
        axis += rank

    if axis < 0 or axis >= rank:
        raise ValueError(
            f"concatenate axis out of range: axis={axis}, rank={rank}, attrs={node.attrs}"
        )

    # Pairwise concatenate because Cactus concat usually takes two tensors.
    out = xs[0]
    for x in xs[1:]:
        out = ctx.graph.concat(out, x, axis)

    expected = _result_shape(node)
    if expected and tuple(out.shape) != tuple(expected):
        raise ValueError(
            f"concatenate lowered to wrong shape: "
            f"axis={axis}, input_shapes={[tuple(x.shape) for x in xs]}, "
            f"got={tuple(out.shape)}, expected={expected}, attrs={node.attrs}"
        )

    return [out]


def _lower_gather(ctx, node):
    table = _force_tensor(_get(ctx, node.inputs[0]), "gather table")
    idx = _force_tensor(_get(ctx, node.inputs[1]), "gather indices")

    table_shape = tuple(table.shape)
    idx_shape = tuple(idx.shape)
    expected = _shape_from_type_str(node.result_types[0]) if node.result_types else None

    # Gemma embedding gather:
    #   table: [vocab, hidden]
    #   idx:   [B,T] or [B,T,1]
    #   out:   [B,T,hidden]
    if len(table_shape) == 2 and expected:
        hidden = table_shape[1]

        if len(idx_shape) >= 1 and idx_shape[-1] == 1:
            squeezed_idx_shape = idx_shape[:-1]
            if tuple(expected) == tuple(squeezed_idx_shape) + (hidden,):
                idx = ctx.graph.reshape(idx, squeezed_idx_shape)
                return [ctx.graph.embedding_from_tensor(table, idx)]

        if tuple(expected) == tuple(idx_shape) + (hidden,):
            return [ctx.graph.embedding_from_tensor(table, idx)]

    # Fallback: simple first-axis gather only.
    out = ctx.graph.gather(table, idx)

    if expected and tuple(out.shape) != tuple(expected):
        out = ctx.graph.reshape(out, expected)

    return [out]


















def _lower_square(ctx, node):
    x = _get(ctx, node.inputs[0])
    return [ctx.graph.multiply(x, x)]


def _lower_sine(ctx, node):
    return [ctx.graph.scalar_sin(_get(ctx, node.inputs[0]))]


def _lower_cosine(ctx, node):
    return [ctx.graph.scalar_cos(_get(ctx, node.inputs[0]))]


def _lower_iota(ctx, node):
    import numpy as np

    if not node.result_types:
        raise ValueError("stablehlo.iota missing result type")

    shape = _shape_from_type_str(node.result_types[0])
    dtype_str = _dtype_from_type_str(node.result_types[0])
    axis = int(node.attrs.get("dimension", node.attrs.get("dim", 0)))

    if not shape:
        raise ValueError(f"iota result shape missing: {node.result_types}")

    if axis < 0:
        axis += len(shape)

    if axis < 0 or axis >= len(shape):
        raise ValueError(f"iota axis out of range: axis={axis}, shape={shape}")

    vals = np.arange(shape[axis], dtype=np.float16)

    view_shape = [1] * len(shape)
    view_shape[axis] = shape[axis]

    vals = vals.reshape(view_shape)
    vals = np.broadcast_to(vals, shape).astype(np.float16)

    # Cactus math path is FP16, and Gemma's iota is just 0..127 for RoPE.
    t = ctx.graph.input(shape, dtype=1)
    ctx.graph.set_input(t, vals)
    return [t]
import math

def _lower_power(ctx, node):
    """
    Lower stablehlo.power.

    Gemma RoPE emits:
        10000 ** (iota(128) / 128)

    Cactus native pow/log/exp is inaccurate for this vector, so inject the
    exact RoPE denominator vector as a build-time constant.
    """
    import numpy as np


    if len(node.inputs) != 2:
        raise NotImplementedError("stablehlo.power expects 2 inputs")

    out_shape = _result_shape(node)


    if tuple(out_shape) == (128,):
        vals = np.power(
            np.float32(10000.0),
            np.arange(128, dtype=np.float32) / np.float32(128.0),
        ).astype(np.float16)



        t = ctx.graph.input(vals.shape, dtype=1)
        ctx.graph.set_input(t, vals, dtype=1)
        return [t]

    base_ssa, exp_ssa = node.inputs

    base_const = _const_scalar_value(ctx, base_ssa)
    exp_const = _const_scalar_value(ctx, exp_ssa)

    base = _get(ctx, base_ssa)
    exp = _get(ctx, exp_ssa)

    if exp_const is not None:
        base_t = _force_tensor(base, "power base")
        return [ctx.graph.pow(base_t, float(exp_const))]

    if base_const is not None:
        exp_t = _storage_tensor(exp)
        y = ctx.graph.scalar_multiply(exp_t, math.log(float(base_const)))
        y = ctx.graph.scalar_exp(y)

        expected = _result_shape(node)
        if expected and tuple(y.shape) != tuple(expected):
            y = _reshape_if_needed(ctx, y, expected)

        return [y]

    base_t = _force_tensor(base, "power base")
    exp_t = _storage_tensor(exp)

    y = ctx.graph.scalar_log(base_t)
    y = ctx.graph.multiply(y, exp_t)
    y = ctx.graph.scalar_exp(y)

    expected = _result_shape(node)
    if expected and tuple(y.shape) != tuple(expected):
        y = _reshape_if_needed(ctx, y, expected)

    return [y]


def _lower_softmax(ctx, node):
    x = _force_tensor(_get(ctx, node.inputs[0]), "softmax")
    axis = int(node.attrs.get("dimension", -1))
    actual_axis = _norm_axis(axis, len(x.shape))
    # Cactus compute_softmax_node ignores params.axis and always softmaxes over
    # the final dimension. Refuse non-last-axis softmax until Cactus supports it.
    if actual_axis != len(x.shape) - 1:
        raise NotImplementedError(
            f"Cactus softmax only supports the last axis today; got axis={axis} for shape={x.shape}"
        )
    return [ctx.graph.softmax(x, axis=axis)]


# ---------------------------------------------------------------------------
# Explicit unsupported ops that need fusions or new Cactus primitives
# ---------------------------------------------------------------------------

def _unsupported(reason: str):
    def handler(ctx, node):
        raise NotImplementedError(f"{node.op} is not safely lowerable as a primitive: {reason}. node attrs={node.attrs}")
    return handler


OP_LOWERING_TABLE: dict[str, Any] = {
    "stablehlo.dot_general":      _lower_dot_general,
    "stablehlo.dot":              _lower_dot_general,
    "stablehlo.add":              _lower_add,
    "stablehlo.subtract":         _lower_subtract,
    "stablehlo.multiply":         _lower_multiply,
    "stablehlo.divide":           _lower_divide,
    "stablehlo.maximum":          _lower_maximum,
    "stablehlo.minimum":          _lower_minimum,
    "stablehlo.negate":           _lower_negate,
    "stablehlo.abs":              _lower_abs,
    "stablehlo.sqrt":             _lower_sqrt,
    "stablehlo.rsqrt":            _lower_rsqrt,
    "stablehlo.exponential":      _lower_exp,
    "stablehlo.log":              _lower_log,
    "stablehlo.sine":             _lower_sin,
    "stablehlo.cosine":           _lower_cos,
    "stablehlo.tanh":             _lower_tanh,
    "stablehlo.logistic":         _lower_logistic,
    "stablehlo.relu":             _lower_relu,
    "stablehlo.silu":             _lower_silu,
    "stablehlo.gelu":             _lower_gelu,
    "stablehlo.power":            _lower_power,
    "chlo.square":                _lower_square,
    "stablehlo.reshape":          _lower_reshape,
    "stablehlo.transpose":        _lower_transpose,
    "stablehlo.broadcast_in_dim": _lower_broadcast_in_dim,
    "stablehlo.slice":            _lower_slice,
    "stablehlo.concatenate":      _lower_concatenate,
    "stablehlo.reduce":           _lower_reduce,
    "stablehlo.gather":           _lower_gather,
    "stablehlo.convert":          _lower_convert,
    "stablehlo.softmax":          _lower_softmax,
    "cactus.mean_keepdims": _lower_cactus_mean_keepdims,

    #new additions based on cactus cpp changes
    "stablehlo.compare":          _lower_compare,
    "stablehlo.and":              _lower_and,
    "stablehlo.or":               _lower_or,
    "stablehlo.not":              _lower_not,
    "stablehlo.select":           _lower_select,


    "chlo.square":          _lower_square,
    "stablehlo.sine":       _lower_sine,
    "stablehlo.cosine":     _lower_cosine,
    "stablehlo.iota":       _lower_iota,
    "stablehlo.power":      _lower_power,
    "stablehlo.gather":     _lower_gather,
    "cactus.mean": _lower_cactus_mean

    # These commonly appear from masks/RoPE/where. Lower them with patterns or
    # new primitives; do not silently fake boolean/int semantics with FP16 ops.
    #added now "stablehlo.iota":             _unsupported("iota should be folded into constants or fused into RoPE/mask patterns"),
}


def lower_op(ctx: "LoweringCtx", node: "IRNode") -> list[Any]:
    global _CURRENT_CTX

    handler = OP_LOWERING_TABLE.get(node.op)
    if handler is None:
        raise NotImplementedError(
            f"No layer-1 lowering for op: {node.op!r}\n"
            f"  node id : {node.id}\n"
            f"  inputs  : {node.inputs}\n"
            f"  attrs   : {node.attrs}\n"
            f"  result_types: {node.result_types}\n"
            "  Add a handler to OP_LOWERING_TABLE or a Layer-2 fusion pattern."
        )
    
    

    old_ctx = _CURRENT_CTX
    _CURRENT_CTX = ctx
    try:
        return handler(ctx, node)
    finally:
        _CURRENT_CTX = old_ctx