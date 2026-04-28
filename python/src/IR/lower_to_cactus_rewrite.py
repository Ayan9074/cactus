"""
Generic StableHLO IR -> Cactus graph lowerer.

This file is intentionally boring.
No model-specific rewrites, no attention fusion, no KV-cache rewriting,
no RMSNorm fusion, no residual clipping.

Goal:
    Parse StableHLO nodes and emit equivalent Cactus graph nodes.

Important current limitation:
    The existing Cactus kernels are mostly FP16 for generic math ops, so this
    lowerer defaults to a Cactus-FP16 execution contract. That means it is a
    semantic replacement for an FP16/mixed Cactus execution path, not exact FP32
    JAX unless the backend adds FP32 kernels for these ops.
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

from src.graph import Graph, Tensor
from src.IR.stablehlo_ir import IRNode, extract_constant


# -----------------------------------------------------------------------------
# Public compatibility stubs
# -----------------------------------------------------------------------------

def find_attention_blocks(nodes, verbose: bool = False):
    """Compatibility hook for older bridge code.

    This generic lowerer deliberately does not fuse attention. Returning [] keeps
    old imports working while making the correctness path completely op-by-op.
    """
    return []


# -----------------------------------------------------------------------------
# Small parsing helpers
# -----------------------------------------------------------------------------
def _extract_slice_params(raw: str):
    """
    Supports common StableHLO forms:

      stablehlo.slice %x [0:4, 128:256] : ...
      stablehlo.slice %x [0:4:1, 128:256:1] : ...

    Also supports attr-style forms if your exporter emits them:
      start_indices = [0, 128], limit_indices = [4, 256], strides = [1, 1]
    """

    # Attr-style.
    m_start = re.search(r"start_indices\s*=\s*\[([^\]]*)\]", raw)
    m_limit = re.search(r"limit_indices\s*=\s*\[([^\]]*)\]", raw)
    m_stride = re.search(r"strides\s*=\s*\[([^\]]*)\]", raw)

    if m_start and m_limit:
        starts = [int(x.strip()) for x in m_start.group(1).split(",") if x.strip()]
        limits = [int(x.strip()) for x in m_limit.group(1).split(",") if x.strip()]
        if m_stride:
            strides = [int(x.strip()) for x in m_stride.group(1).split(",") if x.strip()]
        else:
            strides = [1] * len(starts)
        return starts, limits, strides

    # Bracket-style: [0:4, 128:256] or [0:4:1, 128:256:1]
    m = re.search(r"stablehlo\.slice\s+%[\w\d_]+\s*\[([^\]]+)\]", raw)
    if not m:
        # Quoted op style fallback.
        m = re.search(r"\"stablehlo\.slice\".*?\[([^\]]+)\]", raw)

    if not m:
        raise RuntimeError(f"could not parse slice params from raw: {raw}")

    parts = [p.strip() for p in m.group(1).split(",") if p.strip()]

    starts = []
    limits = []
    strides = []

    for p in parts:
        nums = [int(x.strip()) for x in p.split(":") if x.strip()]
        if len(nums) == 2:
            s, l = nums
            st = 1
        elif len(nums) == 3:
            s, l, st = nums
        else:
            raise RuntimeError(f"bad slice component {p!r} in raw: {raw}")

        starts.append(s)
        limits.append(l)
        strides.append(st)

    return starts, limits, strides

def _prod(xs: Sequence[int]) -> int:
    out = 1
    for x in xs:
        out *= int(x)
    return int(out)

def ensure_fp16(g, x):
    if getattr(x, "dtype", None) == g.FP16:
        return x
    return g.precision_cast(x, g.FP16)


def _extract_concat_dimension(raw: str):
    # StableHLO forms seen:
    #   dimension = 1
    #   dim = 1
    #   dimensions = [1]  # less common, defensive
    patterns = [
        r"\bdimension\s*=\s*(-?\d+)",
        r"\bdim\s*=\s*(-?\d+)",
        r"\bdimensions\s*=\s*\[\s*(-?\d+)\s*\]",
    ]

    for p in patterns:
        m = re.search(p, raw)
        if m:
            return int(m.group(1))

    return None


def _infer_concat_axis(input_shapes, out_shape):
    input_shapes = [tuple(s) for s in input_shapes]
    out_shape = tuple(out_shape)

    rank = len(out_shape)
    candidates = []

    for axis in range(rank):
        ok = True
        expected = list(input_shapes[0])

        for s in input_shapes[1:]:
            if len(s) != rank:
                ok = False
                break

            for d in range(rank):
                if d == axis:
                    continue
                if s[d] != expected[d]:
                    ok = False
                    break

            if not ok:
                break

            expected[axis] += s[axis]

        if ok and tuple(expected) == out_shape:
            candidates.append(axis)

    if len(candidates) == 1:
        return candidates[0]

    raise RuntimeError(
        f"could not infer concatenate axis: inputs={input_shapes}, out={out_shape}, candidates={candidates}"
    )

def ensure_binary_fp16(g, a, b):
    return ensure_fp16(g, a), ensure_fp16(g, b)

def _as_shape(shape: Optional[Sequence[int]]) -> Tuple[int, ...]:
    if shape is None:
        return tuple()
    return tuple(int(x) for x in shape)


def _graph_shape(shape: Optional[Sequence[int]]) -> Tuple[int, ...]:
    """Cactus does not consistently like rank-0 tensors; use (1,) for scalars."""
    s = _as_shape(shape)
    return s if len(s) > 0 else (1,)


def _parse_int_list(text: str) -> list[int]:
    text = text.strip()
    if not text:
        return []
    return [int(x.strip()) for x in text.split(",") if x.strip()]

def make_constant_tensor(g, value, shape, dtype):
    arr = np.asarray(value)

    if shape is None:
        shape = arr.shape

    shape = tuple(int(s) for s in shape)

    if arr.shape == ():
        arr = np.full(shape, arr.item(), dtype=np.float32)
    else:
        arr = np.broadcast_to(arr, shape)

    if dtype == g.FP16:
        arr = np.ascontiguousarray(arr, dtype=np.float16)
    elif dtype == g.FP32:
        arr = np.ascontiguousarray(arr, dtype=np.float32)
    elif dtype == g.INT8:
        arr = np.ascontiguousarray(arr, dtype=np.int8)
    else:
        arr = np.ascontiguousarray(arr)

    t = g.input(shape, dtype)
    g.set_input(t, arr, dtype=dtype)
    return t

def _extract_list_after(raw: str, key: str) -> Optional[list[int]]:
    patterns = [
        rf"{re.escape(key)}\s*=\s*\[([^\]]*)\]",
        rf"{re.escape(key)}\s*=\s*dense<\[([^\]]*)\]>",
    ]
    for pat in patterns:
        m = re.search(pat, raw)
        if m:
            return _parse_int_list(m.group(1))
    return None


def _extract_first_list(raw: str, keys: Sequence[str]) -> Optional[list[int]]:
    for key in keys:
        v = _extract_list_after(raw, key)
        if v is not None:
            return v
    return None


def _extract_axis(raw: str, default: Optional[int] = None) -> Optional[int]:
    for key in ("dimension", "axis", "iota_dimension"):
        m = re.search(rf"{key}\s*=\s*([0-9]+)", raw)
        if m:
            return int(m.group(1))
    return default


def _extract_reduce_axes(raw: str) -> list[int]:
    v = _extract_first_list(raw, ("dimensions", "axes"))
    if v is not None:
        return v
    m = re.search(r"across\s+dimensions\s*=\s*\[([^\]]*)\]", raw)
    if m:
        return _parse_int_list(m.group(1))
    return []


def _extract_permutation(raw: str, rank: int) -> list[int]:
    perm = _extract_first_list(raw, ("permutation", "dims"))
    if perm is not None:
        return perm

    # StableHLO often prints: stablehlo.transpose %x, dims = [1, 0] : ...
    m = re.search(r"dims\s*=\s*\[([^\]]*)\]", raw)
    if m:
        return _parse_int_list(m.group(1))

    # Fallback to swapping the last two dims, matching Graph.transpose behavior.
    perm = list(range(rank))
    if rank >= 2:
        perm[-1], perm[-2] = perm[-2], perm[-1]
    return perm



def _parse_int_list(s: str):
    return [int(x) for x in re.findall(r"-?\d+", s)]


def _extract_attr_int_list(raw: str, attr: str):
    # start_indices = [0, 128]
    m = re.search(rf"\b{attr}\s*=\s*\[([^\]]*)\]", raw)
    if m:
        return _parse_int_list(m.group(1))

    # start_indices = array<i64: 0, 128>
    m = re.search(rf"\b{attr}\s*=\s*array<[^:>]+:\s*([^>]*)>", raw)
    if m:
        return _parse_int_list(m.group(1))

    # start_indices = dense<[0, 128]> : tensor<2xi64>
    m = re.search(rf"\b{attr}\s*=\s*dense<([^>]*)>", raw)
    if m:
        return _parse_int_list(m.group(1))

    return None


def _extract_slice_params(raw: str, input_shape):
    """
    Returns starts, limits, strides.

    Handles:
      stablehlo.slice %x [0:4, 128:256]
      stablehlo.slice %x [0:4:1, 128:256:1]
      start_indices / limit_indices / strides attr forms
    """
    input_shape = tuple(int(x) for x in input_shape)

    starts = _extract_attr_int_list(raw, "start_indices")
    limits = _extract_attr_int_list(raw, "limit_indices")
    strides = _extract_attr_int_list(raw, "strides")

    if starts is not None and limits is not None:
        if strides is None:
            strides = [1] * len(starts)
        return starts, limits, strides

    # Bracket style after stablehlo.slice.
    m = re.search(r"stablehlo\.slice\s+%[\w\d_]+\s*\[([^\]]+)\]", raw)
    if not m:
        m = re.search(r"\"stablehlo\.slice\".*?\[([^\]]+)\]", raw)

    if m:
        parts = [p.strip() for p in m.group(1).split(",") if p.strip()]

        starts = []
        limits = []
        strides = []

        for p in parts:
            nums = [int(x.strip()) for x in p.split(":") if x.strip()]
            if len(nums) == 2:
                s, l = nums
                st = 1
            elif len(nums) == 3:
                s, l, st = nums
            else:
                raise RuntimeError(f"bad slice component {p!r} in raw: {raw}")

            starts.append(s)
            limits.append(l)
            strides.append(st)

        return starts, limits, strides

    raise RuntimeError(f"could not parse slice params from raw: {raw}")


def _dtype_from_tensor_body(body: str) -> Optional[str]:
    """
    Examples:
      f32          -> f32
      2x6xf32      -> f32
      1x4x256xf16  -> f16
      i32          -> i32
      3xi1         -> i1
    """
    body = body.strip()

    # Remove StableHLO decorations if they appear.
    body = body.replace("?", "")

    # tensor<2x6xf32> -> ["2", "6", "f32"] -> f32
    parts = [p.strip() for p in body.split("x") if p.strip()]
    if not parts:
        return None

    return parts[-1].lower()


def _extract_tensor_bodies(raw: str) -> list[str]:
    return re.findall(r"tensor<([^>]+)>", raw)


def _extract_element_dtype(raw: str) -> Optional[str]:
    bodies = _extract_tensor_bodies(raw)
    if not bodies:
        return None
    return _dtype_from_tensor_body(bodies[-1])


def _extract_convert_target_dtype(raw: str) -> Optional[str]:
    m = re.search(r"->\s*tensor<([^>]+)>", raw)
    if m:
        return _dtype_from_tensor_body(m.group(1))
    return _extract_element_dtype(raw)





def _dtype_to_graph(dtype: Optional[str], default: int = Graph.FP16) -> int:
    if dtype is None:
        return default
    d = dtype.lower()
    if d in ("f16", "bf16"):
        return Graph.FP16
    if d in ("f32", "f64"):
        return Graph.FP32
    # Cactus Python wrapper only exposes INT8/INT4 for integer storage. For
    # generic arithmetic we avoid integer tensors except for tiny index constants.
    if d.startswith("i") or d.startswith("ui") or d.startswith("si"):
        return Graph.INT8
    return default


# -----------------------------------------------------------------------------
# Constant handling and constant folding
# -----------------------------------------------------------------------------

def _parse_dense_literal(raw: str, shape: Tuple[int, ...]) -> Optional[np.ndarray]:
    """Parse small StableHLO dense constants.

    Returns None for huge hex blobs / splats we cannot safely parse.
    """
    m = re.search(r"dense<([^>]+)>", raw)
    if not m:
        return None

    body = m.group(1).strip()
    if body.startswith('"0x') or len(body) > 5000:
        return None

    dtype = _extract_element_dtype(raw) or "f32"
    is_float = dtype.startswith("f") or dtype in ("bf16", "f64")

    # Defensive: infinities must always be treated as floating values.
    body_lower = body.lower()
    if "inf" in body_lower or "nan" in body_lower:
        is_float = True

    np_dtype = np.float32 if is_float else np.int64

    # Booleans in compare/select masks.
    if body in ("true", "false"):
        arr = np.array(1.0 if body == "true" else 0.0, dtype=np.float32)
        return np.broadcast_to(arr, shape).copy() if shape else arr

    # Scalar, including hex float bit patterns.
    if not body.startswith("["):
        val = extract_constant(raw)
        if val is None:
            return None
        arr = np.array(val, dtype=np.float32 if is_float else np.int64)
        return np.broadcast_to(arr, shape).copy() if shape else arr

    # Small list/tensor literal. This deliberately handles the common flat case.
    cleaned = body.replace("[", " ").replace("]", " ").replace("\n", " ")
    parts = [p.strip() for p in cleaned.split(",") if p.strip()]
    vals = []
    for p in parts:
        if p in ("true", "false"):
            vals.append(1.0 if p == "true" else 0.0)
        elif p.startswith("0x"):
            # Use extract_constant-style bit interpretation by rebuilding a fake line.
            fake = raw.replace(body, p)
            v = extract_constant(fake)
            if v is None:
                return None
            vals.append(v)
        else:
            vals.append(float(p) if is_float else int(p))

    arr = np.array(vals, dtype=np_dtype)
    if shape:
        try:
            arr = arr.reshape(shape)
        except ValueError:
            if arr.size == 1:
                arr = np.broadcast_to(arr, shape).copy()
            else:
                return None
    return arr


def _to_cactus_array(arr: Any, dtype: int) -> np.ndarray:
    if dtype == Graph.FP16:
        return np.ascontiguousarray(arr, dtype=np.float16)
    if dtype == Graph.FP32:
        return np.ascontiguousarray(arr, dtype=np.float32)
    if dtype == Graph.INT8:
        return np.ascontiguousarray(arr, dtype=np.int8)
    if dtype == Graph.INT4:
        return np.ascontiguousarray(arr, dtype=np.uint8)
    raise RuntimeError(f"unsupported graph dtype {dtype}")


def _emit_const(g: Graph, value: Any, shape: Optional[Sequence[int]] = None, *, dtype: int = Graph.FP16) -> Tensor:
    arr = np.asarray(value)
    logical_shape = _as_shape(shape) if shape is not None else tuple(arr.shape)
    physical_shape = _graph_shape(logical_shape)

    if logical_shape and arr.shape != logical_shape:
        if arr.size == 1:
            arr = np.broadcast_to(arr.reshape(()), logical_shape)
        else:
            arr = arr.reshape(logical_shape)
    elif not logical_shape:
        arr = arr.reshape(())

    # Cactus rank-0 workaround.
    arr = arr.reshape(physical_shape)
    t = g.input(physical_shape, dtype)
    g.set_input(t, _to_cactus_array(arr, dtype), dtype=dtype)
    return t


def _numpy_broadcast_in_dim(x: np.ndarray, target_shape: Tuple[int, ...], dims: Sequence[int]) -> np.ndarray:
    x = np.asarray(x)
    if x.shape == tuple():
        return np.broadcast_to(x, target_shape).copy()
    if len(dims) != x.ndim:
        # If Cactus scalar workaround left x as (1,), but StableHLO meant scalar.
        if x.size == 1 and len(dims) == 0:
            return np.broadcast_to(x.reshape(()), target_shape).copy()
        raise RuntimeError(f"broadcast dims {dims} do not match input rank {x.ndim}")
    reshaped = [1] * len(target_shape)
    for src_axis, dst_axis in enumerate(dims):
        reshaped[int(dst_axis)] = x.shape[src_axis]
    return np.broadcast_to(x.reshape(reshaped), target_shape).copy()


def _const_fold(op: str, vals: list[np.ndarray], node: IRNode) -> Optional[np.ndarray]:
    try:
        if op == "add":
            return vals[0] + vals[1]
        if op == "subtract":
            return vals[0] - vals[1]
        if op == "multiply":
            return vals[0] * vals[1]
        if op == "divide":
            return vals[0] / vals[1]
        if op == "maximum":
            return np.maximum(vals[0], vals[1])
        if op == "minimum":
            return np.minimum(vals[0], vals[1])
        if op == "negate":
            return -vals[0]
        if op in ("exponential", "exp"):
            return np.exp(vals[0])
        if op == "sqrt":
            return np.sqrt(vals[0])
        if op == "rsqrt":
            return 1.0 / np.sqrt(vals[0])
        if op == "tanh":
            return np.tanh(vals[0])
        if op == "logistic":
            return 1.0 / (1.0 + np.exp(-vals[0]))
        if op == "abs":
            return np.abs(vals[0])
        if op == "convert":
            return vals[0]
        if op == "reshape":
            return vals[0].reshape(_as_shape(node.shape))
        if op == "broadcast_in_dim":
            target = _as_shape(node.shape)
            dims = _extract_first_list(node.raw, ("broadcast_dimensions", "dimensions", "dims"))
            if dims is None:
                dims = list(range(vals[0].ndim)) if vals[0].ndim else []
            return _numpy_broadcast_in_dim(vals[0], target, dims)
        if op == "transpose":
            perm = _extract_permutation(node.raw, vals[0].ndim)
            return np.transpose(vals[0], axes=perm)
        if op.startswith("reduce_"):
            axes = tuple(_extract_reduce_axes(node.raw))
            kind = op[len("reduce_"):]
            if kind in ("add", "sum"):
                return np.sum(vals[0], axis=axes)
            if kind in ("maximum", "max"):
                return np.max(vals[0], axis=axes)
            if kind in ("minimum", "min"):
                return np.min(vals[0], axis=axes)
        if op == "compare":
            direction = _extract_compare_direction(node.raw)
            a, b = vals[0], vals[1]
            if direction in ("LT", "lt"):
                return (a < b).astype(np.float32)
            if direction in ("LE", "le"):
                return (a <= b).astype(np.float32)
            if direction in ("GT", "gt"):
                return (a > b).astype(np.float32)
            if direction in ("GE", "ge"):
                return (a >= b).astype(np.float32)
            if direction in ("EQ", "eq"):
                return (a == b).astype(np.float32)
            if direction in ("NE", "ne"):
                return (a != b).astype(np.float32)
        if op == "select":
            cond = vals[0].astype(bool)
            return np.where(cond, vals[1], vals[2])
        if op == "iota":
            return _numpy_iota(node)
        if op == "slice":
            starts, ends, strides = _extract_slice_params(node.raw, tuple(vals[0].shape))
            sl = tuple(slice(s, e, st) for s, e, st in zip(starts, ends, strides))
            return vals[0][sl]
        if op == "concatenate":
            axis = _extract_axis(node.raw, default=0)
            return np.concatenate(vals, axis=axis)
    except Exception:
        return None
    return None


def _numpy_iota(node: IRNode) -> np.ndarray:
    shape = _as_shape(node.shape)
    axis = _extract_axis(node.raw, default=0)
    if axis is None:
        axis = 0
    r = np.arange(shape[axis], dtype=np.float32)
    view_shape = [1] * len(shape)
    view_shape[axis] = shape[axis]
    return np.broadcast_to(r.reshape(view_shape), shape).copy()


def _extract_compare_direction(raw: str) -> str:
    # StableHLO forms vary; catch common ones.
    for pat in (
        r"comparison_direction\s*=\s*#stablehlo<comparison_direction\s+([A-Z]+)>",
        r"comparison_direction\s*=\s*([A-Z]+)",
        r"direction\s*=\s*#stablehlo<comparison_direction\s+([A-Z]+)>",
        r"direction\s*=\s*([A-Z]+)",
    ):
        m = re.search(pat, raw)
        if m:
            return m.group(1)
    return "EQ"


# -----------------------------------------------------------------------------
# Tensor lowering helpers
# -----------------------------------------------------------------------------

def _ensure_dtype(g: Graph, x: Tensor, dtype: int) -> Tensor:
    if getattr(x, "dtype", None) == dtype:
        return x
    return g.precision_cast(x, dtype)


def _ensure_math(g: Graph, x: Tensor, dtype_policy: str) -> Tensor:
    # Current generic Cactus arithmetic/matmul kernels are FP16-centric.
    # Keep this explicit so we can later add strict FP32 mode once kernels exist.
    if dtype_policy == "fp32":
        return _ensure_dtype(g, x, Graph.FP32)
    return _ensure_dtype(g, x, Graph.FP16)


def _ensure_binary_math(g: Graph, a: Tensor, b: Tensor, dtype_policy: str) -> tuple[Tensor, Tensor]:
    return _ensure_math(g, a, dtype_policy), _ensure_math(g, b, dtype_policy)
def ensure_fp16(g, x):
    if getattr(x, "dtype", None) == g.FP16:
        return x
    return g.precision_cast(x, g.FP16)


def repeat_axis(g, x, axis, times):
    if times == 1:
        return x
    return g.cat([x for _ in range(int(times))], axis=int(axis))


def _shape_of(x):
    return tuple(int(v) for v in getattr(x, "shape", ()))


def align_binary_operands(g, a, b, a_shape=None, b_shape=None):
    """
    Explicitly implements NumPy-style right-aligned broadcasting before calling
    Cactus binary ops. This avoids cactus_graph_multiply/add/etc failing on
    StableHLO broadcast shapes.
    """
    a_shape = tuple(int(v) for v in (a_shape if a_shape is not None else _shape_of(a)))
    b_shape = tuple(int(v) for v in (b_shape if b_shape is not None else _shape_of(b)))

    # Treat scalar/empty shape as [1] because Cactus tensors are happier rank >= 1.
    if len(a_shape) == 0:
        a_shape = (1,)
        a = g.reshape(a, a_shape)
    if len(b_shape) == 0:
        b_shape = (1,)
        b = g.reshape(b, b_shape)

    rank = max(len(a_shape), len(b_shape))

    a_padded = (1,) * (rank - len(a_shape)) + a_shape
    b_padded = (1,) * (rank - len(b_shape)) + b_shape

    if a_padded != a_shape:
        a = g.reshape(a, a_padded)
    if b_padded != b_shape:
        b = g.reshape(b, b_padded)

    out_shape = []
    for sa, sb in zip(a_padded, b_padded):
        if sa == sb:
            out_shape.append(sa)
        elif sa == 1:
            out_shape.append(sb)
        elif sb == 1:
            out_shape.append(sa)
        else:
            raise RuntimeError(f"cannot broadcast binary shapes {a_shape} and {b_shape}")

    out_shape = tuple(out_shape)

    for axis, (sa, so) in enumerate(zip(a_padded, out_shape)):
        if sa == 1 and so != 1:
            a = repeat_axis(g, a, axis, so)

    for axis, (sb, so) in enumerate(zip(b_padded, out_shape)):
        if sb == 1 and so != 1:
            b = repeat_axis(g, b, axis, so)

    a = ensure_fp16(g, a)
    b = ensure_fp16(g, b)

    return a, b, out_shape

def _repeat_axis(g: Graph, x: Tensor, axis: int, times: int) -> Tensor:
    if times == 1:
        return x
    return g.cat([x for _ in range(int(times))], axis=int(axis))


def _broadcast_tensor(
    g: Graph,
    x: Tensor,
    input_shape: Tuple[int, ...],
    target_shape: Tuple[int, ...],
    dims: Sequence[int],
) -> Tensor:
    """Lower stablehlo.broadcast_in_dim using reshape + cat repeats."""
    if input_shape == target_shape:
        return x

    if len(target_shape) == 0:
        return x

    # StableHLO scalar -> tensor.
    if len(input_shape) == 0 or (input_shape == (1,) and len(dims) == 0):
        reshaped_shape = tuple(1 for _ in target_shape)
    else:
        if len(dims) != len(input_shape):
            # If physical scalar is (1,), treat it as logical scalar.
            if _prod(input_shape) == 1 and len(dims) == 0:
                reshaped_shape = tuple(1 for _ in target_shape)
            else:
                raise RuntimeError(
                    f"broadcast_in_dim rank mismatch: input_shape={input_shape}, dims={dims}, target={target_shape}"
                )
        else:
            reshaped = [1] * len(target_shape)
            for src_axis, dst_axis in enumerate(dims):
                reshaped[int(dst_axis)] = int(input_shape[src_axis])
            reshaped_shape = tuple(reshaped)

    if tuple(x.shape) != reshaped_shape:
        x = g.reshape(x, reshaped_shape)

    cur_shape = list(reshaped_shape)
    for axis, (cur, target) in enumerate(zip(cur_shape, target_shape)):
        cur = int(cur)
        target = int(target)
        if cur == target:
            continue
        if cur != 1:
            raise RuntimeError(f"cannot broadcast axis {axis}: {cur_shape} -> {target_shape}")
        x = _repeat_axis(g, x, axis, target)
        cur_shape[axis] = target
    return x


def _ones_like(g: Graph, shape: Tuple[int, ...], dtype: int = Graph.FP16) -> Tensor:
    return _emit_const(g, np.ones(_graph_shape(shape), dtype=np.float16), shape, dtype=dtype)


def _zeros_like(g: Graph, shape: Tuple[int, ...], dtype: int = Graph.FP16) -> Tensor:
    return _emit_const(g, np.zeros(_graph_shape(shape), dtype=np.float16), shape, dtype=dtype)


def _elementwise_max(g: Graph, a: Tensor, b: Tensor, shape: Tuple[int, ...], dtype_policy: str) -> Tensor:
    a, b = _ensure_binary_math(g, a, b, dtype_policy)
    # max(a, b) = b + relu(a - b)
    return g.add(b, g.relu(g.subtract(a, b)))


def _elementwise_min(g: Graph, a: Tensor, b: Tensor, shape: Tuple[int, ...], dtype_policy: str) -> Tensor:
    a, b = _ensure_binary_math(g, a, b, dtype_policy)
    # min(a, b) = a - relu(a - b)
    return g.subtract(a, g.relu(g.subtract(a, b)))


def _lower_reduce(g: Graph, x: Tensor, axes: Sequence[int], kind: str) -> Tensor:
    if not axes:
        return x
    y = x
    # Reducing descending keeps remaining axis numbers valid.
    for axis in sorted([int(a) for a in axes], reverse=True):
        if kind in ("add", "sum"):
            y = g.sum(y, axis)
        elif kind in ("maximum", "max"):
            y = g.max(y, axis)
        elif kind in ("minimum", "min"):
            y = g.min(y, axis)
        else:
            raise NotImplementedError(f"unsupported reduce kind: {kind}")
    return y


# -----------------------------------------------------------------------------
# dot_general lowering
# -----------------------------------------------------------------------------

def _parse_dot_dims(raw: str, lhs_shape: Tuple[int, ...], rhs_shape: Tuple[int, ...]):
    lhs_contract = _extract_first_list(raw, ("lhs_contracting_dimensions", "lhs_contracting_dims"))
    rhs_contract = _extract_first_list(raw, ("rhs_contracting_dimensions", "rhs_contracting_dims"))
    lhs_batch = _extract_first_list(raw, ("lhs_batching_dimensions", "lhs_batch_dimensions", "lhs_batching_dims"))
    rhs_batch = _extract_first_list(raw, ("rhs_batching_dimensions", "rhs_batch_dimensions", "rhs_batching_dims"))

    if lhs_contract is None or rhs_contract is None:
        # Infer common matmul cases.
        lhs_contract = [len(lhs_shape) - 1]
        if len(rhs_shape) >= 2 and lhs_shape[-1] == rhs_shape[-2]:
            rhs_contract = [len(rhs_shape) - 2]
        elif len(rhs_shape) >= 1 and lhs_shape[-1] == rhs_shape[-1]:
            rhs_contract = [len(rhs_shape) - 1]
        else:
            rhs_contract = [0]

    if lhs_batch is None:
        lhs_batch = []
    if rhs_batch is None:
        rhs_batch = []

    # If no batch dims are printed but both tensors look like batched matmul,
    # infer equal leading dims as batch dims.
    if not lhs_batch and not rhs_batch and len(lhs_shape) >= 3 and len(rhs_shape) >= 3:
        l_non = set(lhs_contract)
        r_non = set(rhs_contract)
        lhs_lead = [i for i in range(len(lhs_shape) - 2) if i not in l_non]
        rhs_lead = [i for i in range(len(rhs_shape) - 2) if i not in r_non]
        if len(lhs_lead) == len(rhs_lead):
            ok = all(lhs_shape[i] == rhs_shape[j] for i, j in zip(lhs_lead, rhs_lead))
            if ok:
                lhs_batch = lhs_lead
                rhs_batch = rhs_lead

    return lhs_batch, rhs_batch, lhs_contract, rhs_contract


def _permute_if_needed(g: Graph, x: Tensor, perm: list[int]) -> Tensor:
    if perm == list(range(len(perm))):
        return x
    return g.permute(x, perm)


def _lower_dot_general(
    g: Graph,
    lhs: Tensor,
    rhs: Tensor,
    lhs_shape: Tuple[int, ...],
    rhs_shape: Tuple[int, ...],
    rhs_storage_shape: Optional[Tuple[int, ...]],
    out_shape: Tuple[int, ...],
    raw: str,
    dtype_policy: str,
    pretransposed_rhs: bool = False,
    rhs_logical_to_storage: Optional[Sequence[int]] = None,
) -> Tensor:
    lhs = _ensure_math(g, lhs, dtype_policy)
    rhs = _ensure_math(g, rhs, dtype_policy)

    if rhs_logical_to_storage is None:
        rhs_logical_to_storage = list(range(len(rhs_shape)))
    rhs_logical_to_storage = [int(x) for x in rhs_logical_to_storage]
    if len(rhs_logical_to_storage) != len(rhs_shape):
        raise RuntimeError(
            f"rhs logical/storage rank mismatch: logical={rhs_shape}, map={rhs_logical_to_storage}"
        )

    rhs_storage_shape = tuple(rhs_storage_shape or rhs.shape)
    if len(rhs_storage_shape) != len(rhs_shape):
        raise RuntimeError(
            f"rhs storage rank mismatch: storage={rhs_storage_shape}, logical={rhs_shape}"
        )

    lhs_batch, rhs_batch, lhs_contract, rhs_contract = _parse_dot_dims(raw, lhs_shape, rhs_shape)
    lhs_batch = [int(x) for x in lhs_batch]
    rhs_batch = [int(x) for x in rhs_batch]
    lhs_contract = [int(x) for x in lhs_contract]
    rhs_contract = [int(x) for x in rhs_contract]

    if len(lhs_contract) != 1 or len(rhs_contract) != 1:
        raise NotImplementedError(
            f"dot_general only supports one contracting dim for now, got {lhs_contract=} {rhs_contract=}"
        )

    lhs_free = [i for i in range(len(lhs_shape)) if i not in set(lhs_batch + lhs_contract)]
    rhs_free = [i for i in range(len(rhs_shape)) if i not in set(rhs_batch + rhs_contract)]

    lhs_perm = lhs_batch + lhs_free + lhs_contract
    rhs_perm_logical = rhs_batch + (rhs_free + rhs_contract if pretransposed_rhs else rhs_contract + rhs_free)
    rhs_perm_storage = [rhs_logical_to_storage[i] for i in rhs_perm_logical]
    lhs_p = _permute_if_needed(g, lhs, lhs_perm)
    rhs_p = _permute_if_needed(g, rhs, rhs_perm_storage)

    batch_shape = tuple(lhs_shape[i] for i in lhs_batch)
    rhs_batch_shape = tuple(rhs_shape[i] for i in rhs_batch)
    if batch_shape != rhs_batch_shape:
        raise NotImplementedError(f"dot_general batch shape mismatch: {batch_shape} vs {rhs_batch_shape}")

    lhs_free_shape = tuple(lhs_shape[i] for i in lhs_free)
    rhs_free_shape = tuple(rhs_shape[i] for i in rhs_free)
    contract_shape_l = tuple(lhs_shape[i] for i in lhs_contract)
    contract_shape_r = tuple(rhs_shape[i] for i in rhs_contract)
    if _prod(contract_shape_l) != _prod(contract_shape_r):
        raise RuntimeError(f"dot_general K mismatch: {contract_shape_l} vs {contract_shape_r}")

    B = _prod(batch_shape) if batch_shape else 1
    M = _prod(lhs_free_shape) if lhs_free_shape else 1
    K = _prod(contract_shape_l)
    N = _prod(rhs_free_shape) if rhs_free_shape else 1

    if B == 1:
        lhs_2d = g.reshape(lhs_p, (M, K))
        rhs_2d = g.reshape(rhs_p, (N, K) if pretransposed_rhs else (K, N))
        y = g.matmul(lhs_2d, rhs_2d, pretransposed_rhs=pretransposed_rhs)
        return g.reshape(y, _graph_shape(out_shape)) if tuple(y.shape) != _graph_shape(out_shape) else y

    lhs_3d = g.reshape(lhs_p, (B, M, K))
    rhs_3d = g.reshape(rhs_p, (B, N, K) if pretransposed_rhs else (B, K, N))

    pieces = []
    for bi in range(B):
        l_i = lhs_3d.index(bi, axis=0)  # [M, K]
        r_i = rhs_3d.index(bi, axis=0)  # [K, N] or [N, K] (pretransposed)
        y_i = g.matmul(l_i, r_i, pretransposed_rhs=pretransposed_rhs)
        pieces.append(g.reshape(y_i, (1, M, N)))

    y = pieces[0] if len(pieces) == 1 else g.cat(pieces, axis=0)
    return g.reshape(y, _graph_shape(out_shape))


# -----------------------------------------------------------------------------
# Main lowerer
# -----------------------------------------------------------------------------

def lower_to_cactus(
    nodes: Sequence[IRNode],
    g: Graph,
    input_map: Dict[str, Tensor],
    input_shapes: Optional[Dict[str, Sequence[int]]] = None,
    raw_inputs: Optional[Sequence[Any]] = None,
    *,
    dtype_policy: str = "fp16",
    debug_taps: Optional[Dict[str, Tensor]] = None,
    unsupported: str = "raise",
    **ignored_options: Any,
) -> Dict[str, Tensor]:
    """Lower parsed StableHLO IR nodes to a Cactus graph.

    Parameters intentionally accept older optimization kwargs, but this function
    ignores them. This keeps bridge code stable while correctness work stays
    generic and op-by-op.

    dtype_policy:
        "fp16"  -> cast math inputs to FP16 before Cactus ops. Current default.
        "fp32"  -> attempt FP32, but many current Cactus generic ops may fail.
    """
    if dtype_policy not in ("fp16", "fp32"):
        raise ValueError("dtype_policy must be 'fp16' or 'fp32'")

    env: Dict[str, Tensor] = dict(input_map)
    shapes: Dict[str, Tuple[int, ...]] = {
        k: _as_shape(v) for k, v in (input_shapes or {}).items()
    }
    nodes_by_name: Dict[str, IRNode] = {n.name: n for n in nodes}
    arg_specs: Dict[str, Dict[str, Any]] = dict(ignored_options.get("arg_specs") or {})
    for name, t in input_map.items():
        shapes.setdefault(name, tuple(t.shape))

    # Best-effort constant values for folding masks/iotas/static select trees.
    consts: Dict[str, np.ndarray] = {}

    def fail(msg: str):
        if unsupported == "ignore":
            return None
        raise NotImplementedError(msg)

    for idx, node in enumerate(nodes):
        op = node.op
        out_shape = _as_shape(node.shape)

        # ------------------------------------------------------------------
        # Constants / static op folding
        # ------------------------------------------------------------------
        if op == "constant":
            arr = _parse_dense_literal(node.raw, out_shape)
            if arr is None:
                fail(f"could not parse constant at node {idx}: {node.raw[:200]}")
                continue
            consts[node.name] = arr
            dtype = _dtype_to_graph(_extract_element_dtype(node.raw), default=Graph.FP16)
            # Prefer FP16 for floating constants under the current Cactus math contract.
            if dtype == Graph.FP32 and dtype_policy == "fp16":
                dtype = Graph.FP16
            env[node.name] = _emit_const(g, arr, out_shape, dtype=dtype)
            shapes[node.name] = out_shape
            continue

        # iota has no inputs but is effectively a constant for fixed-shape graphs.
        if op == "iota":
            arr = _numpy_iota(node)
            consts[node.name] = arr
            env[node.name] = _emit_const(g, arr, out_shape, dtype=Graph.FP16)
            shapes[node.name] = out_shape
            continue

        # If every input is constant, do a numpy fold and emit one Cactus input.
        if node.inputs and all(inp in consts for inp in node.inputs):
            folded = _const_fold(op, [consts[inp] for inp in node.inputs], node)
            if folded is not None:
                consts[node.name] = folded
                dtype = Graph.FP16
                if np.issubdtype(np.asarray(folded).dtype, np.integer):
                    # Keep small gather indices usable.
                    dtype = Graph.INT8
                env[node.name] = _emit_const(g, folded, out_shape or folded.shape, dtype=dtype)
                shapes[node.name] = out_shape or tuple(folded.shape)
                continue

        # ------------------------------------------------------------------
        # Dynamic op lowering
        # ------------------------------------------------------------------
        try:
            if op == "reshape":
                x = env[node.inputs[0]]
                env[node.name] = g.reshape(x, _graph_shape(out_shape))

            elif op == "broadcast_in_dim":
                x_name = node.inputs[0]
                x = env[x_name]
                in_shape = shapes.get(x_name, tuple(x.shape))
                dims = _extract_first_list(node.raw, ("broadcast_dimensions", "dimensions", "dims"))
                if dims is None:
                    dims = list(range(len(in_shape))) if len(in_shape) else []
                env[node.name] = _broadcast_tensor(g, x, in_shape, out_shape, dims)


            
            elif op == "tril":
                x = env[node.inputs[0]]
                x_shape = tuple(shapes[node.inputs[0]])

                if len(x_shape) < 2:
                    raise RuntimeError(f"tril requires rank >= 2, got shape {x_shape}")

                # StableHLO/JAX tril applies over the last two dimensions.
                rows = x_shape[-2]
                cols = x_shape[-1]

                base_mask = np.tril(np.ones((rows, cols), dtype=np.float32))

                # Broadcast mask over leading dimensions if needed.
                mask_shape = (1,) * (len(x_shape) - 2) + (rows, cols)
                base_mask = base_mask.reshape(mask_shape)
                base_mask = np.broadcast_to(base_mask, x_shape)

                mask = make_constant_tensor(g, base_mask, x_shape, getattr(x, "dtype", g.FP16))

                x, mask = ensure_binary_fp16(g, x, mask)
                env[node.name] = g.multiply(x, mask)
                shapes[node.name] = tuple(node.shape or x_shape)
                
            elif op == "transpose":
                x_name = node.inputs[0]
                x = env[x_name]
                in_shape = tuple(shapes.get(x_name, tuple(x.shape)))
                perm = _extract_permutation(node.raw, len(in_shape))

                # no-op transpose
                if perm == list(range(len(in_shape))):
                    env[node.name] = x
                    shapes[node.name] = in_shape
                    continue

                try:
                    env[node.name] = g.permute(x, perm)
                except Exception as e:
                    raise RuntimeError(
                        f"permute failed for {node.name}: input={x_name}, "
                        f"in_shape={in_shape}, tensor_shape={tuple(x.shape)}, "
                        f"perm={perm}, out_shape={node.shape}, raw={node.raw}"
                    ) from e

            elif op == "convert":
                x = env[node.inputs[0]]
                target_dtype = _dtype_to_graph(_extract_convert_target_dtype(node.raw), default=Graph.FP16)
                if dtype_policy == "fp16" and target_dtype == Graph.FP32:
                    target_dtype = Graph.FP16
                env[node.name] = _ensure_dtype(g, x, target_dtype)

            elif op in ("add", "subtract", "multiply", "divide"):
                a_name = node.inputs[0]
                b_name = node.inputs[1]

                a = env[a_name]
                b = env[b_name]

                a_shape = shapes.get(a_name, getattr(a, "shape", None))
                b_shape = shapes.get(b_name, getattr(b, "shape", None))

                a, b, out_shape = align_binary_operands(g, a, b, a_shape, b_shape)

                try:
                    if op == "add":
                        y = g.add(a, b)
                    elif op == "subtract":
                        y = g.subtract(a, b)
                    elif op == "multiply":
                        y = g.multiply(a, b)
                    elif op == "divide":
                        y = g.divide(a, b)
                except Exception as e:
                    raise RuntimeError(
                        f"binary {op} failed for {node.name}: "
                        f"{a_name} shape={getattr(a, 'shape', None)} dtype={getattr(a, 'dtype', None)}, "
                        f"{b_name} shape={getattr(b, 'shape', None)} dtype={getattr(b, 'dtype', None)}, "
                        f"ir_shapes={a_shape},{b_shape}, raw={node.raw}"
                    ) from e

                env[node.name] = y
                shapes[node.name] = tuple(node.shape or out_shape)

            elif op == "maximum":
                env[node.name] = _elementwise_max(
                    g, env[node.inputs[0]], env[node.inputs[1]], out_shape, dtype_policy
                )

            elif op == "minimum":
                env[node.name] = _elementwise_min(
                    g, env[node.inputs[0]], env[node.inputs[1]], out_shape, dtype_policy
                )

            elif op == "negate":
                x = env[node.inputs[0]]
                x_shape = tuple(shapes[node.inputs[0]])

                x = ensure_fp16(g, x)
                env[node.name] = g.scalar_multiply(x, -1.0)
                shapes[node.name] = tuple(node.shape or x_shape)

            elif op in ("exponential", "exp"):
                x = _ensure_math(g, env[node.inputs[0]], dtype_policy)
                env[node.name] = g.scalar_exp(x)

            elif op == "sqrt":
                x = _ensure_math(g, env[node.inputs[0]], dtype_policy)
                env[node.name] = g.scalar_sqrt(x)

            elif op == "rsqrt":
                x = _ensure_math(g, env[node.inputs[0]], dtype_policy)
                sx = g.scalar_sqrt(x)
                one = _ones_like(g, shapes.get(node.inputs[0], tuple(x.shape)), Graph.FP16)
                env[node.name] = g.divide(one, sx)

            elif op == "tanh":
                x = _ensure_math(g, env[node.inputs[0]], dtype_policy)
                env[node.name] = g.tanh(x)

            elif op == "logistic":
                x = _ensure_math(g, env[node.inputs[0]], dtype_policy)
                env[node.name] = g.sigmoid(x)

            elif op == "abs":
                x = _ensure_math(g, env[node.inputs[0]], dtype_policy)
                env[node.name] = g.abs(x)

            elif op == "pow":
                x = _ensure_math(g, env[node.inputs[0]], dtype_policy)
                exponent = extract_constant(node.raw)
                if exponent is None:
                    fail(f"dynamic pow exponent not supported at {node.name}")
                    continue
                env[node.name] = g.pow(x, float(exponent))

            elif op.startswith("reduce_"):
                x = _ensure_math(g, env[node.inputs[0]], dtype_policy)
                kind = op[len("reduce_"):]
                axes = _extract_reduce_axes(node.raw)
                env[node.name] = _lower_reduce(g, x, axes, kind)

            elif op == "dot_general":
                lhs_name, rhs_name = node.inputs[0], node.inputs[1]
                rhs_tensor = env[rhs_name]
                rhs_storage_shape = shapes.get(rhs_name, tuple(rhs_tensor.shape))
                rhs_logical_to_storage = list(range(len(rhs_storage_shape)))
                use_pretransposed_rhs = False

                # Cactus-aware fast path:
                # dot_general(lhs, transpose(static_weight)) -> matmul(pretransposed_rhs=True)
                rhs_node = nodes_by_name.get(rhs_name)
                if rhs_node is not None and rhs_node.op == "transpose" and rhs_node.inputs:
                    src_name = rhs_node.inputs[0]
                    src_spec = arg_specs.get(src_name, {})
                    src_shape = shapes.get(src_name, tuple(env[src_name].shape))
                    perm = _extract_permutation(rhs_node.raw, len(src_shape))
                    expected_swap = list(range(len(src_shape)))
                    if len(expected_swap) >= 2:
                        expected_swap[-1], expected_swap[-2] = expected_swap[-2], expected_swap[-1]
                    if (
                        src_spec.get("kind") == "weight"
                        and bool(src_spec.get("pretransposed_rhs"))
                        and perm == expected_swap
                    ):
                        rhs_tensor = env[src_name]
                        rhs_storage_shape = shapes.get(src_name, tuple(rhs_tensor.shape))
                        # logical rhs dim i maps to storage dim perm[i]
                        rhs_logical_to_storage = list(perm)
                        use_pretransposed_rhs = True

                env[node.name] = _lower_dot_general(
                    g,
                    env[lhs_name],
                    rhs_tensor,
                    shapes.get(lhs_name, tuple(env[lhs_name].shape)),
                    shapes.get(rhs_name, tuple(env[rhs_name].shape)),
                    rhs_storage_shape,
                    out_shape,
                    node.raw,
                    dtype_policy,
                    pretransposed_rhs=use_pretransposed_rhs,
                    rhs_logical_to_storage=rhs_logical_to_storage,
                )

            elif op in ("concatenate", "concat"):
                xs = [env[name] for name in node.inputs]
                in_shapes = [tuple(shapes[name]) for name in node.inputs]
                out_shape = tuple(node.shape)

                axis = _extract_concat_dimension(node.raw)
                if axis is None:
                    axis = _infer_concat_axis(in_shapes, out_shape)

                if axis < 0:
                    axis += len(out_shape)

                xs = [ensure_fp16(g, x) for x in xs]

                if len(xs) == 0:
                    raise RuntimeError(f"concatenate requires at least one input: {node.raw}")

                y = xs[0]
                for x_next in xs[1:]:
                    y = g.concat(y, x_next, axis=axis)

                if tuple(y.shape) != out_shape:
                    raise RuntimeError(
                        f"concatenate produced wrong shape for {node.name}: "
                        f"got {tuple(y.shape)}, expected {out_shape}, axis={axis}, "
                        f"input_shapes={in_shapes}, raw={node.raw}"
                    )

                env[node.name] = y
                shapes[node.name] = out_shape

            elif op == "slice":
                x_name = node.inputs[0]
                x = env[x_name]
                in_shape = tuple(shapes[x_name])
                out_shape = tuple(node.shape)

                starts, limits, strides = _extract_slice_params(node.raw, in_shape)

                if len(starts) != len(in_shape):
                    raise RuntimeError(
                        f"slice rank mismatch for {node.name}: "
                        f"starts={starts}, input_shape={in_shape}, raw={node.raw}"
                    )

                y = x
                cur_shape = list(in_shape)

                for axis, (start, limit, stride) in enumerate(zip(starts, limits, strides)):
                    if stride != 1:
                        raise RuntimeError(
                            f"slice stride != 1 not supported yet for {node.name}: "
                            f"axis={axis}, stride={stride}, raw={node.raw}"
                        )

                    length = int(limit) - int(start)
                    if length < 0:
                        raise RuntimeError(
                            f"bad slice length for {node.name}: start={start}, limit={limit}, raw={node.raw}"
                        )

                    # Only emit an actual Cactus slice if this dimension changes.
                    if start != 0 or length != cur_shape[axis]:
                        y = g.slice(y, axis, int(start), int(length))
                        cur_shape[axis] = length

                # Cactus slice should already have correct shape, but reshape defensively.
                actual_shape = tuple(getattr(y, "shape", ()))

                if actual_shape != out_shape:
                    raise RuntimeError(
                        f"slice produced wrong shape for {node.name}: "
                        f"got {actual_shape}, expected {out_shape}, "
                        f"starts={starts}, limits={limits}, strides={strides}, "
                        f"input_shape={in_shape}, raw={node.raw}"
                    )

                env[node.name] = y
                shapes[node.name] = out_shape

            elif op == "gather":
                data = env[node.inputs[0]]
                indices = env[node.inputs[1]]
                if indices.dtype not in (Graph.INT8, Graph.FP32):
                    indices = _ensure_dtype(g, indices, Graph.FP32)
                env[node.name] = g.gather(data, indices)

            elif op == "select":
                cond_name, a_name, b_name = node.inputs[:3]
                if cond_name in consts:
                    cond_t = _emit_const(g, consts[cond_name].astype(np.float32), shapes.get(cond_name, consts[cond_name].shape), dtype=Graph.FP16)
                    a = _ensure_math(g, env[a_name], dtype_policy)
                    b = _ensure_math(g, env[b_name], dtype_policy)
                    cond_shape = shapes.get(cond_name, tuple(cond_t.shape))
                    a_shape = shapes.get(a_name, tuple(a.shape))
                    b_shape = shapes.get(b_name, tuple(b.shape))
                    target = out_shape or tuple(np.broadcast_shapes(cond_shape, a_shape, b_shape))
                    cond_t = _broadcast_tensor(g, cond_t, cond_shape, target, list(range(len(cond_shape))) if cond_shape else [])
                    a_part = g.multiply(cond_t, a)
                    one_minus = g.subtract(_ones_like(g, target, Graph.FP16), cond_t)
                    b_part = g.multiply(one_minus, b)
                    env[node.name] = g.add(a_part, b_part)
                else:
                    fail(f"dynamic select unsupported at {node.name}")
                    continue

            elif op == "compare":
                # Dynamic compare is not in the current Graph API. If this was not
                # folded above, fail clearly.
                fail(f"dynamic compare unsupported at {node.name}")
                continue

            elif op in ("and", "or"):
                fail(f"dynamic logical op unsupported at {node.name}: {op}")
                continue

            elif op in ("relu",):
                x = _ensure_math(g, env[node.inputs[0]], dtype_policy)
                env[node.name] = g.relu(x)

            elif op in ("silu",):
                x = _ensure_math(g, env[node.inputs[0]], dtype_policy)
                env[node.name] = g.silu(x)

            elif op in ("gelu",):
                x = _ensure_math(g, env[node.inputs[0]], dtype_policy)
                env[node.name] = g.gelu(x)

            elif op in ("softmax",):
                x = _ensure_math(g, env[node.inputs[0]], dtype_policy)
                axis = _extract_axis(node.raw, default=-1)
                env[node.name] = g.softmax(x, axis=axis)

            else:
                fail(f"unsupported op at node {idx}: {node.name} op={op} raw={node.raw[:200]}")
                continue

        except KeyError as e:
            raise KeyError(f"missing input {e} while lowering node {idx}: {node.name} op={op}") from e
        except Exception as e:
            raise RuntimeError(f"failed lowering node {idx}: {node.name} op={op}: {e}") from e

        shapes[node.name] = out_shape if out_shape is not None else tuple(env[node.name].shape)

        # Optional coarse debugging: save every dynamic node if caller wants taps.
        # The bridge already scans env directly, so this is mostly for external tests.
        if debug_taps is not None and node.name in debug_taps:
            debug_taps[node.name] = env[node.name]

    return env
