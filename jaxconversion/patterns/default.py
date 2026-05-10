"""
patterns_default.py / default.py
================================

Built-in model-agnostic fusions.

Each handler:
    (ctx, nodes, idx) -> ([out_tensors], nodes_consumed) | None

Current fusions:
  - RMSNorm for Qwen/Llama-style post mean_keepdims canonicalization
  - Approx GELU for Gemma-style tanh GELU
  - SiLU / Swish
  - ReLU
  - Softmax, including clamped/casted StableHLO variants

Important:
  These are conservative pattern checks. If the exact pattern is not present,
  they return None and generic lowering handles the graph.
"""

from __future__ import annotations

from pattern_registry import Pattern, LoweringCtx, is_const, const_value


# ---------------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------------

def _op(nodes, idx, offset=0):
    i = idx + offset
    return nodes[i] if i < len(nodes) else None


def _is(nodes, idx, opcode, offset=0):
    n = _op(nodes, idx, offset)
    return n is not None and n.op == opcode


def _get(ctx, ssa):
    return ctx.env[ssa]


def _out(node, i=0):
    if node is None or not node.outputs or len(node.outputs) <= i:
        return None
    return node.outputs[i]


def _inp(node, i=0):
    if node is None or not node.inputs or len(node.inputs) <= i:
        return None
    return node.inputs[i]


def _has_input(node, ssa):
    return node is not None and ssa in node.inputs


def _other_input(node, known):
    if node is None:
        return None
    xs = [x for x in node.inputs if x != known]
    return xs[0] if len(xs) == 1 else None


def _const_float(ctx, ssa):
    if not is_const(ctx, ssa):
        return None
    try:
        return float(const_value(ctx, ssa))
    except Exception:
        return None


def _is_const_value(ctx, ssa, target, tol=1e-6):
    v = _const_float(ctx, ssa)
    return v is not None and abs(v - target) <= tol


def _bcast_const_value(ctx, node, target, tol=1e-6):
    if node is None or node.op != "stablehlo.broadcast_in_dim":
        return False
    src = _inp(node, 0)
    return src is not None and _is_const_value(ctx, src, target, tol=tol)


def _axis_from_reduce(node):
    dims = node.attrs.get("dimensions", node.attrs.get("dim", node.attrs.get("axis", [-1])))

    if isinstance(dims, (list, tuple)):
        if len(dims) == 0:
            return -1
        return int(dims[0])

    return int(dims)


def _reduce_applies(node):
    if node is None:
        return ""
    return str(
        node.attrs.get(
            "applies",
            node.attrs.get("body", node.attrs.get("reduce_op", "")),
        )
    ).lower()


def _reduce_is_add(node):
    applies = _reduce_applies(node)
    return applies == "" or "add" in applies


def _reduce_is_max(node):
    applies = _reduce_applies(node)
    return applies == "" or "max" in applies or "maximum" in applies


def _bind_consumed_outputs(ctx, nodes, idx, consumed, out):
    for off in range(consumed):
        n = _op(nodes, idx, off)
        if n is None:
            continue
        for ssa in n.outputs:
            ctx.env[ssa] = out


def _producer_idx(nodes, ssa):
    for i, n in enumerate(nodes):
        if ssa in n.outputs:
            return i
    return None


def _call_rms_norm(ctx, x, weight, eps):
    # Cactus RMSNorm kernel is currently FP16-only.
    x_in = x
    w_in = weight
    x_work = x if getattr(x, "dtype", None) == 1 else ctx.graph.precision_cast(x, 1)
    w_work = weight if getattr(weight, "dtype", None) == 1 else ctx.graph.precision_cast(weight, 1)
    try:
        out = ctx.graph.rms_norm(x_work, w_work, eps=eps)
    except TypeError:
        out = ctx.graph.rms_norm(x_work, w_work, eps)
    if getattr(x_in, "dtype", None) != 1:
        out = ctx.graph.precision_cast(out, getattr(x_in, "dtype", 1))
    return out


def _call_gelu(ctx, x):
    if hasattr(ctx.graph, "gelu"):
        return ctx.graph.gelu(x)
    if hasattr(ctx.graph, "gelu_erf"):
        return ctx.graph.gelu_erf(x)
    return None

def _consumers_map(nodes):
    m = {}
    for i, n in enumerate(nodes):
        for inp in getattr(n, "inputs", []) or []:
            m.setdefault(inp, []).append((i, n))
    return m


def _single_consumer(nodes, ssa, allowed_ops=None, after_idx=None):
    consumers = _consumers_map(nodes).get(ssa, [])
    if after_idx is not None:
        consumers = [(i, n) for i, n in consumers if i > after_idx]
    if allowed_ops is not None:
        consumers = [(i, n) for i, n in consumers if n.op in allowed_ops]
    if len(consumers) != 1:
        return None
    return consumers[0]


def _first_consumer(nodes, ssa, allowed_ops=None, after_idx=None):
    consumers = _consumers_map(nodes).get(ssa, [])
    if after_idx is not None:
        consumers = [(i, n) for i, n in consumers if i > after_idx]
    if allowed_ops is not None:
        consumers = [(i, n) for i, n in consumers if n.op in allowed_ops]
    if not consumers:
        return None
    return sorted(consumers, key=lambda x: x[0])[0]


def _trace_back_through_unary(nodes, ssa, allowed=("stablehlo.convert", "stablehlo.broadcast_in_dim")):
    changed = True
    while changed:
        changed = False
        for n in nodes:
            if ssa in getattr(n, "outputs", []):
                if n.op in allowed and len(n.inputs) == 1:
                    ssa = n.inputs[0]
                    changed = True
                break
    return ssa


def _producer_node(nodes, ssa):
    for i, n in enumerate(nodes):
        if ssa in getattr(n, "outputs", []):
            return i, n
    return None

# ---------------------------------------------------------------------------
# RMSNorm, post mean_keepdims canonicalizer
# ---------------------------------------------------------------------------
def _try_rmsnorm_from_mean_keepdims_dataflow(ctx: LoweringCtx, nodes: list, idx: int):
    """
    Looser JAXgarden/Qwen-style RMSNorm matcher starting at cactus.mean_keepdims.

    Matches dataflow:
      square(x) -> mean_keepdims
      mean + eps -> sqrt or rsqrt path
      inv = 1/sqrt(...) or rsqrt(...)
      norm = x * inv
      out = norm * weight

    This is intentionally looser than fixed-offset matching so it survives
    extra convert/broadcast nodes.
    """
    mean = _op(nodes, idx)
    if mean is None or mean.op != "cactus.mean_keepdims":
        return None

    mean_in = mean.inputs[0]
    prod = _producer_node(nodes, mean_in)
    if prod is None:
        return None

    square_idx, square = prod

    if square.op == "chlo.square":
        x_ssa = square.inputs[0]
    elif square.op == "stablehlo.multiply" and len(square.inputs) == 2 and square.inputs[0] == square.inputs[1]:
        x_ssa = square.inputs[0]
    else:
        return None

    if x_ssa not in ctx.env:
        return None

    # mean -> add eps
    add_hit = _first_consumer(nodes, mean.outputs[0], {"stablehlo.add"}, after_idx=idx)
    if add_hit is None:
        return None
    add_idx, add_eps = add_hit

    # add -> sqrt or rsqrt
    sqrt_hit = _first_consumer(nodes, add_eps.outputs[0], {"stablehlo.sqrt", "stablehlo.rsqrt"}, after_idx=add_idx)
    if sqrt_hit is None:
        return None
    sqrt_idx, sqrt_node = sqrt_hit

    if sqrt_node.op == "stablehlo.rsqrt":
        inv_ssa = sqrt_node.outputs[0]
    else:
        # sqrt -> divide one/sqrt
        div_hit = _first_consumer(nodes, sqrt_node.outputs[0], {"stablehlo.divide"}, after_idx=sqrt_idx)
        if div_hit is None:
            return None
        div_idx, div_inv = div_hit
        if sqrt_node.outputs[0] != div_inv.inputs[1]:
            return None
        inv_ssa = div_inv.outputs[0]

    # inv may be broadcast/convert before multiply
    # Find multiply that uses x_ssa and something tracing back to inv_ssa.
    norm_mul = None
    norm_mul_idx = None
    search_start = sqrt_idx

    for j in range(search_start + 1, min(len(nodes), search_start + 20)):
        n = nodes[j]
        if n.op != "stablehlo.multiply" or len(n.inputs) != 2:
            continue

        a, b = n.inputs
        a_base = _trace_back_through_unary(nodes, a)
        b_base = _trace_back_through_unary(nodes, b)

        if (a_base == x_ssa and b_base == inv_ssa) or (b_base == x_ssa and a_base == inv_ssa):
            norm_mul = n
            norm_mul_idx = j
            break

    if norm_mul is None:
        return None

    # Find final multiply with weight after norm_mul.
    final_mul = None
    final_mul_idx = None
    weight_ssa = None

    norm_out = norm_mul.outputs[0]

    for j in range(norm_mul_idx + 1, min(len(nodes), norm_mul_idx + 25)):
        n = nodes[j]
        if n.op != "stablehlo.multiply" or len(n.inputs) != 2:
            continue

        a, b = n.inputs
        a_base = _trace_back_through_unary(nodes, a)
        b_base = _trace_back_through_unary(nodes, b)

        if a_base == norm_out:
            maybe_weight = b_base
        elif b_base == norm_out:
            maybe_weight = a_base
        else:
            continue

        if maybe_weight in ctx.env:
            final_mul = n
            final_mul_idx = j
            weight_ssa = maybe_weight
            break

    if final_mul is None or weight_ssa is None:
        return None

    eps = 1e-6
    try:
        # eps is the non-mean input to add.
        for s in add_eps.inputs:
            if s != mean.outputs[0]:
                parsed = _const_float(ctx, _trace_back_through_unary(nodes, s))
                if parsed is not None and 0.0 < float(parsed) < 1e-2:
                    eps = float(parsed)
                    break
    except Exception:
        pass

    x_tensor = ctx.env[x_ssa]
    w_tensor = ctx.env[weight_ssa]
    x_shape = tuple(int(v) for v in x_tensor.shape)

    if len(x_shape) == 3:
        b, t, d = x_shape
        x_2d = ctx.graph.reshape(x_tensor, (b * t, d))
        out_2d = _call_rms_norm(ctx, x_2d, w_tensor, eps)
        out = ctx.graph.reshape(out_2d, x_shape)
    elif len(x_shape) == 2:
        out = _call_rms_norm(ctx, x_tensor, w_tensor, eps)
    else:
        return None

    # Bind all nodes from mean through final_mul to the fused output.
    for j in range(idx, final_mul_idx + 1 if "final_mul_idx" in locals() else idx + 1):
        pass

    # Simpler binding: bind final output and common path outputs.
    ctx.env[mean.outputs[0]] = out
    ctx.env[norm_mul.outputs[0]] = out
    ctx.env[final_mul.outputs[0]] = out

    return [out], 1




def _try_jaxgarden_sqrt_rmsnorm(ctx: LoweringCtx, nodes: list, idx: int):
    """
    Match JAXgarden / Flax NNX Llama RMSNorm:

      0  chlo.square              x^2
      1  stablehlo.reduce         sum(x^2, hidden)
      2  stablehlo.broadcast      sum keepdims-ish
      3  stablehlo.broadcast      hidden_size const
      4  stablehlo.divide         mean = sum / hidden_size
      5  stablehlo.broadcast      eps
      6  stablehlo.add            mean + eps
      7  stablehlo.sqrt           sqrt(mean + eps)
      8  stablehlo.broadcast      one
      9  stablehlo.divide         inv = 1 / sqrt(...)
      10 stablehlo.broadcast      inv expanded
      11 stablehlo.multiply       x * inv
      12 stablehlo.convert
      13 stablehlo.convert        weight
      14 stablehlo.convert        norm
      15 stablehlo.broadcast      weight
      16 stablehlo.broadcast      weight expanded
      17 stablehlo.multiply       norm * weight

    This differs from Qwen pattern because it uses chlo.square + sqrt/divide,
    not multiply square + rsqrt.
    """
    expected = [
        "chlo.square",
        "stablehlo.reduce",
        "stablehlo.broadcast_in_dim",
        "stablehlo.broadcast_in_dim",
        "stablehlo.divide",
        "stablehlo.broadcast_in_dim",
        "stablehlo.add",
        "stablehlo.sqrt",
        "stablehlo.broadcast_in_dim",
        "stablehlo.divide",
        "stablehlo.broadcast_in_dim",
        "stablehlo.multiply",
        "stablehlo.convert",
        "stablehlo.convert",
        "stablehlo.convert",
        "stablehlo.broadcast_in_dim",
        "stablehlo.broadcast_in_dim",
        "stablehlo.multiply",
    ]

    for off, opcode in enumerate(expected):
        if not _is(nodes, idx, opcode, off):
            return None

    square = _op(nodes, idx, 0)
    red = _op(nodes, idx, 1)
    bsum = _op(nodes, idx, 2)
    bhidden = _op(nodes, idx, 3)
    div_mean = _op(nodes, idx, 4)
    beps = _op(nodes, idx, 5)
    add_eps = _op(nodes, idx, 6)
    sqrt = _op(nodes, idx, 7)
    bone = _op(nodes, idx, 8)
    div_inv = _op(nodes, idx, 9)
    binv = _op(nodes, idx, 10)
    mul_norm = _op(nodes, idx, 11)
    conv_norm1 = _op(nodes, idx, 12)
    conv_weight = _op(nodes, idx, 13)
    conv_norm2 = _op(nodes, idx, 14)
    bweight1 = _op(nodes, idx, 15)
    bweight2 = _op(nodes, idx, 16)
    final_mul = _op(nodes, idx, 17)

    x_ssa = square.inputs[0]
    if x_ssa not in ctx.env:
        return None

    # square -> reduce
    if red.inputs[0] != square.outputs[0]:
        return None

    if not _reduce_is_add(red):
        return None

    # reduce -> broadcast -> divide lhs
    if bsum.inputs[0] != red.outputs[0]:
        return None

    if bsum.outputs[0] != div_mean.inputs[0]:
        return None

    if bhidden.outputs[0] != div_mean.inputs[1]:
        return None

    # mean + eps
    if div_mean.outputs[0] not in add_eps.inputs:
        return None

    if beps.outputs[0] not in add_eps.inputs:
        return None

    # sqrt(mean + eps)
    if sqrt.inputs[0] != add_eps.outputs[0]:
        return None

    # 1 / sqrt
    if bone.outputs[0] != div_inv.inputs[0]:
        return None

    if sqrt.outputs[0] != div_inv.inputs[1]:
        return None

    # broadcast inv, x * inv
    if binv.inputs[0] != div_inv.outputs[0]:
        return None

    if x_ssa not in mul_norm.inputs:
        return None

    if binv.outputs[0] not in mul_norm.inputs:
        return None

    # norm convert chain
    if conv_norm1.inputs[0] != mul_norm.outputs[0]:
        return None

    if conv_norm2.inputs[0] != conv_norm1.outputs[0]:
        return None

    # weight path
    weight_ssa = conv_weight.inputs[0]
    if weight_ssa not in ctx.env:
        return None

    if bweight1.inputs[0] != conv_weight.outputs[0]:
        return None

    if bweight2.inputs[0] != bweight1.outputs[0]:
        return None

    if conv_norm2.outputs[0] not in final_mul.inputs:
        return None

    if bweight2.outputs[0] not in final_mul.inputs:
        return None

    # Parse eps if possible.
    eps = 1e-6
    try:
        eps_src = beps.inputs[0]
        parsed = _const_float(ctx, eps_src)
        if parsed is not None and 0.0 < float(parsed) < 1e-2:
            eps = float(parsed)
    except Exception:
        pass

    x_tensor = ctx.env[x_ssa]
    w_tensor = ctx.env[weight_ssa]
    x_shape = tuple(int(v) for v in x_tensor.shape)

    # Cactus rms_norm expects 2D, so flatten [B,T,D] -> [B*T,D].
    if len(x_shape) == 3:
        b, t, d = x_shape
        x_2d = ctx.graph.reshape(x_tensor, (b * t, d))
        out_2d = _call_rms_norm(ctx, x_2d, w_tensor, eps)
        out = ctx.graph.reshape(out_2d, x_shape)
    elif len(x_shape) == 2:
        out = _call_rms_norm(ctx, x_tensor, w_tensor, eps)
    else:
        return None

    _bind_consumed_outputs(ctx, nodes, idx, 18, out)
    ctx.env[final_mul.outputs[0]] = out

    return [out], 18
    
def _try_rmsnorm_after_mean_keepdims(ctx: LoweringCtx, nodes: list, idx: int):
    """
    Match Qwen/Llama RMSNorm after lower_to_cactus rewrites:

        reduce sum -> broadcast -> divide by hidden size

    into:

        cactus.mean_keepdims

    Post-canonicalized pattern:

      0   multiply              x_f32 * x_f32
      1   reduce                old reduce, mostly dead
      2   broadcast_in_dim      old bsum, mostly dead
      3   broadcast_in_dim      old hidden const, mostly dead
      4   cactus.mean_keepdims  mean(x_f32 * x_f32), keepdims=True
      5   broadcast_in_dim      eps
      6   add                   mean + eps
      7   rsqrt
      8   broadcast_in_dim      rsqrt expanded
      9   multiply              x_f32 * rsqrt
      10  convert               norm back to fp16/bf16
      11  broadcast_in_dim      weight
      12  broadcast_in_dim      weight expanded
      13  multiply              norm * weight
    """
    expected = [
        "stablehlo.multiply",
        "stablehlo.reduce",
        "stablehlo.broadcast_in_dim",
        "stablehlo.broadcast_in_dim",
        "cactus.mean_keepdims",
        "stablehlo.broadcast_in_dim",
        "stablehlo.add",
        "stablehlo.rsqrt",
        "stablehlo.broadcast_in_dim",
        "stablehlo.multiply",
        "stablehlo.convert",
        "stablehlo.broadcast_in_dim",
        "stablehlo.broadcast_in_dim",
        "stablehlo.multiply",
    ]

    for off, opcode in enumerate(expected):
        if not _is(nodes, idx, opcode, off):
            return None

    sq = _op(nodes, idx, 0)
    red = _op(nodes, idx, 1)
    bsum = _op(nodes, idx, 2)
    bhidden = _op(nodes, idx, 3)
    mean = _op(nodes, idx, 4)
    beps = _op(nodes, idx, 5)
    add = _op(nodes, idx, 6)
    rsqrt = _op(nodes, idx, 7)
    brsqrt = _op(nodes, idx, 8)
    norm_mul = _op(nodes, idx, 9)
    norm_convert = _op(nodes, idx, 10)
    weight_b1 = _op(nodes, idx, 11)
    weight_b2 = _op(nodes, idx, 12)
    final_mul = _op(nodes, idx, 13)

    if len(sq.inputs) != 2 or sq.inputs[0] != sq.inputs[1]:
        return None

    x_f32 = sq.inputs[0]

    conv_idx = _producer_idx(nodes, x_f32)
    if conv_idx is None:
        return None

    conv_x = nodes[conv_idx]
    if conv_x.op != "stablehlo.convert":
        return None

    x_ssa = conv_x.inputs[0]
    if x_ssa not in ctx.env:
        return None

    if red.inputs[0] != sq.outputs[0]:
        return None

    if bsum.inputs[0] != red.outputs[0]:
        return None

    if bhidden.op != "stablehlo.broadcast_in_dim":
        return None

    if mean.inputs[0] != sq.outputs[0]:
        return None

    try:
        axis = int(mean.attrs.get("axis", -1))
        x_shape = getattr(ctx.env[x_ssa], "shape", None)
        if x_shape is not None:
            rank = len(x_shape)
            axis_norm = axis + rank if axis < 0 else axis
            if axis_norm != rank - 1:
                return None
    except Exception:
        pass

    if mean.outputs[0] not in add.inputs:
        return None

    if beps.outputs[0] not in add.inputs:
        return None

    if rsqrt.inputs[0] != add.outputs[0]:
        return None

    if brsqrt.inputs[0] != rsqrt.outputs[0]:
        return None

    if x_f32 not in norm_mul.inputs:
        return None

    if brsqrt.outputs[0] not in norm_mul.inputs:
        return None

    if norm_convert.inputs[0] != norm_mul.outputs[0]:
        return None

    weight_ssa = weight_b1.inputs[0]

    if weight_ssa not in ctx.env:
        return None

    if weight_b2.inputs[0] != weight_b1.outputs[0]:
        return None

    if norm_convert.outputs[0] not in final_mul.inputs:
        return None

    if weight_b2.outputs[0] not in final_mul.inputs:
        return None

    eps = 1e-6
    try:
        eps_src = beps.inputs[0]
        parsed = _const_float(ctx, eps_src)
        if parsed is not None and 0.0 < float(parsed) < 1e-2:
            eps = float(parsed)
    except Exception:
        pass

    x_tensor = ctx.env[x_ssa]
    w_tensor = ctx.env[weight_ssa]
    x_shape = tuple(int(v) for v in x_tensor.shape)

    # Cactus RMSNorm currently expects 2D [batch, dims].
    if len(x_shape) == 3:
        b, t, d = x_shape
        x_2d = ctx.graph.reshape(x_tensor, (b * t, d))
        out_2d = _call_rms_norm(ctx, x_2d, w_tensor, eps)
        out = ctx.graph.reshape(out_2d, x_shape)
    elif len(x_shape) == 2:
        out = _call_rms_norm(ctx, x_tensor, w_tensor, eps)
    else:
        return None

    _bind_consumed_outputs(ctx, nodes, idx, 14, out)
    ctx.env[final_mul.outputs[0]] = out

    return [out], 14

def _try_jaxgarden_sqrt_rmsnorm_after_mean_keepdims(ctx: LoweringCtx, nodes: list, idx: int):
    """
    JAXgarden RMSNorm after lower_to_cactus canonicalizes:

      reduce_sum(square) / hidden_size -> cactus.mean_keepdims

    Expected:
      0  chlo.square
      1  stablehlo.reduce
      2  stablehlo.broadcast_in_dim
      3  stablehlo.broadcast_in_dim
      4  cactus.mean_keepdims
      5  stablehlo.broadcast_in_dim      eps
      6  stablehlo.add
      7  stablehlo.sqrt
      8  stablehlo.broadcast_in_dim      one
      9  stablehlo.divide               1 / sqrt(...)
      10 stablehlo.broadcast_in_dim
      11 stablehlo.multiply             x * inv
      12 stablehlo.convert
      13 stablehlo.convert              weight
      14 stablehlo.convert
      15 stablehlo.broadcast_in_dim
      16 stablehlo.broadcast_in_dim
      17 stablehlo.multiply             norm * weight
    """
    expected = [
        "chlo.square",
        "stablehlo.reduce",
        "stablehlo.broadcast_in_dim",
        "stablehlo.broadcast_in_dim",
        "cactus.mean_keepdims",
        "stablehlo.broadcast_in_dim",
        "stablehlo.add",
        "stablehlo.sqrt",
        "stablehlo.broadcast_in_dim",
        "stablehlo.divide",
        "stablehlo.broadcast_in_dim",
        "stablehlo.multiply",
        "stablehlo.convert",
        "stablehlo.convert",
        "stablehlo.convert",
        "stablehlo.broadcast_in_dim",
        "stablehlo.broadcast_in_dim",
        "stablehlo.multiply",
    ]

    for off, opcode in enumerate(expected):
        if not _is(nodes, idx, opcode, off):
            return None

    square = _op(nodes, idx, 0)
    mean = _op(nodes, idx, 4)
    beps = _op(nodes, idx, 5)
    add_eps = _op(nodes, idx, 6)
    sqrt = _op(nodes, idx, 7)
    bone = _op(nodes, idx, 8)
    div_inv = _op(nodes, idx, 9)
    binv = _op(nodes, idx, 10)
    mul_norm = _op(nodes, idx, 11)
    conv_norm1 = _op(nodes, idx, 12)
    conv_weight = _op(nodes, idx, 13)
    conv_norm2 = _op(nodes, idx, 14)
    bweight1 = _op(nodes, idx, 15)
    bweight2 = _op(nodes, idx, 16)
    final_mul = _op(nodes, idx, 17)

    x_ssa = square.inputs[0]
    if x_ssa not in ctx.env:
        return None

    if mean.inputs[0] != square.outputs[0]:
        return None

    if mean.outputs[0] not in add_eps.inputs:
        return None

    if beps.outputs[0] not in add_eps.inputs:
        return None

    if sqrt.inputs[0] != add_eps.outputs[0]:
        return None

    if bone.outputs[0] != div_inv.inputs[0]:
        return None

    if sqrt.outputs[0] != div_inv.inputs[1]:
        return None

    if binv.inputs[0] != div_inv.outputs[0]:
        return None

    if x_ssa not in mul_norm.inputs:
        return None

    if binv.outputs[0] not in mul_norm.inputs:
        return None

    if conv_norm1.inputs[0] != mul_norm.outputs[0]:
        return None

    if conv_norm2.inputs[0] != conv_norm1.outputs[0]:
        return None

    weight_ssa = conv_weight.inputs[0]
    if weight_ssa not in ctx.env:
        return None

    if bweight1.inputs[0] != conv_weight.outputs[0]:
        return None

    if bweight2.inputs[0] != bweight1.outputs[0]:
        return None

    if conv_norm2.outputs[0] not in final_mul.inputs:
        return None

    if bweight2.outputs[0] not in final_mul.inputs:
        return None

    eps = 1e-6
    parsed = _const_float(ctx, beps.inputs[0])
    if parsed is not None and 0.0 < float(parsed) < 1e-2:
        eps = float(parsed)

    x_tensor = ctx.env[x_ssa]
    w_tensor = ctx.env[weight_ssa]
    x_shape = tuple(int(v) for v in x_tensor.shape)

    if len(x_shape) == 3:
        b, t, d = x_shape
        x_2d = ctx.graph.reshape(x_tensor, (b * t, d))
        out_2d = _call_rms_norm(ctx, x_2d, w_tensor, eps)
        out = ctx.graph.reshape(out_2d, x_shape)
    elif len(x_shape) == 2:
        out = _call_rms_norm(ctx, x_tensor, w_tensor, eps)
    else:
        return None

    _bind_consumed_outputs(ctx, nodes, idx, 18, out)
    ctx.env[final_mul.outputs[0]] = out

    return [out], 18



def _rmsnorm_pattern(ctx: LoweringCtx, nodes: list, idx: int):
    result = _try_rmsnorm_after_mean_keepdims(ctx, nodes, idx)
    if result is not None:
        return result

    result = _try_jaxgarden_sqrt_rmsnorm_after_mean_keepdims(ctx, nodes, idx)
    if result is not None:
        return result

    result = _try_rmsnorm_from_mean_keepdims_dataflow(ctx, nodes, idx)
    if result is not None:
        return result

    result = _try_jaxgarden_sqrt_rmsnorm(ctx, nodes, idx)
    if result is not None:
        return result

    return None

RMSNORM = Pattern(
    name="rmsnorm",
    handler=_rmsnorm_pattern,
    trigger_ops={"stablehlo.multiply", "chlo.square", "stablehlo.reduce", "cactus.mean_keepdims"}
)


# ---------------------------------------------------------------------------
# Approx GELU, Gemma tanh form
# ---------------------------------------------------------------------------

def _try_gemma_approx_gelu(ctx: LoweringCtx, nodes: list, idx: int):
    """
    Match Gemma approximate GELU:

      x2 = x * x
      x3 = x2 * x
      a  = c0 * x3
      b  = x + a
      c  = c1 * b
      t  = tanh(c)
      d  = 1 + t
      e  = 0.5 * d
      y  = x * e

    Seen in Gemma windows like nodes 189..201:
      square, cube, const multiply, add, const multiply, tanh,
      add one, multiply half, multiply x.
    """
   # if _call_gelu(ctx, None) is None:
        # This is not actually reachable because passing None would error,
        # but keep the availability check below instead.
   #     pass

    if not hasattr(ctx.graph, "gelu") and not hasattr(ctx.graph, "gelu_erf"):
        return None

    expected = [
        "stablehlo.multiply",          # 0: x*x
        "stablehlo.multiply",          # 1: x2*x
        "stablehlo.broadcast_in_dim",  # 2: 0.044715
        "stablehlo.multiply",          # 3: c*x3
        "stablehlo.add",               # 4: x + c*x3
        "stablehlo.broadcast_in_dim",  # 5: sqrt(2/pi)
        "stablehlo.multiply",          # 6: c*(...)
        "stablehlo.tanh",              # 7
        "stablehlo.broadcast_in_dim",  # 8: 1
        "stablehlo.add",               # 9: 1 + tanh
        "stablehlo.broadcast_in_dim",  # 10: 0.5
        "stablehlo.multiply",          # 11: 0.5*(...)
        "stablehlo.multiply",          # 12: x * ...
    ]

    for off, opcode in enumerate(expected):
        if not _is(nodes, idx, opcode, off):
            return None

    sq = _op(nodes, idx, 0)
    cube = _op(nodes, idx, 1)
    c044_b = _op(nodes, idx, 2)
    c044_mul = _op(nodes, idx, 3)
    add_x = _op(nodes, idx, 4)
    csqrt_b = _op(nodes, idx, 5)
    csqrt_mul = _op(nodes, idx, 6)
    tanh = _op(nodes, idx, 7)
    one_b = _op(nodes, idx, 8)
    add_one = _op(nodes, idx, 9)
    half_b = _op(nodes, idx, 10)
    half_mul = _op(nodes, idx, 11)
    final_mul = _op(nodes, idx, 12)

    if len(sq.inputs) != 2 or sq.inputs[0] != sq.inputs[1]:
        return None

    x_ssa = sq.inputs[0]
    if x_ssa not in ctx.env:
        return None

    if sq.outputs[0] not in cube.inputs or x_ssa not in cube.inputs:
        return None

    if c044_b.outputs[0] not in c044_mul.inputs or cube.outputs[0] not in c044_mul.inputs:
        return None

    if x_ssa not in add_x.inputs or c044_mul.outputs[0] not in add_x.inputs:
        return None

    if csqrt_b.outputs[0] not in csqrt_mul.inputs or add_x.outputs[0] not in csqrt_mul.inputs:
        return None

    if tanh.inputs[0] != csqrt_mul.outputs[0]:
        return None

    if one_b.outputs[0] not in add_one.inputs or tanh.outputs[0] not in add_one.inputs:
        return None

    if half_b.outputs[0] not in half_mul.inputs or add_one.outputs[0] not in half_mul.inputs:
        return None

    if x_ssa not in final_mul.inputs or half_mul.outputs[0] not in final_mul.inputs:
        return None

    out = _call_gelu(ctx, ctx.env[x_ssa])
    if out is None:
        return None

    _bind_consumed_outputs(ctx, nodes, idx, 13, out)
    ctx.env[final_mul.outputs[0]] = out

    return [out], 13


def _gelu_pattern(ctx: LoweringCtx, nodes: list, idx: int):
    return _try_gemma_approx_gelu(ctx, nodes, idx)


GELU = Pattern(
    name="gelu",
    handler=_gelu_pattern,
    trigger_ops={"stablehlo.multiply"},
)


# ---------------------------------------------------------------------------
# ReLU
# ---------------------------------------------------------------------------

def _relu_pattern(ctx: LoweringCtx, nodes: list, idx: int):
    n0 = _op(nodes, idx)
    n1 = _op(nodes, idx, 1)

    if n0 is not None and n1 is not None:
        if n0.op == "stablehlo.broadcast_in_dim" and n1.op == "stablehlo.maximum":
            if _bcast_const_value(ctx, n0, 0.0):
                bcast_out = _out(n0)
                if bcast_out is not None and _has_input(n1, bcast_out):
                    x_ssa = _other_input(n1, bcast_out)
                    if x_ssa in ctx.env:
                        out = ctx.graph.relu(_get(ctx, x_ssa))
                        _bind_consumed_outputs(ctx, nodes, idx, 2, out)
                        return [out], 2

    if n0 is not None and n0.op == "stablehlo.maximum":
        a, b = n0.inputs

        if _is_const_value(ctx, a, 0.0) and b in ctx.env:
            out = ctx.graph.relu(_get(ctx, b))
            _bind_consumed_outputs(ctx, nodes, idx, 1, out)
            return [out], 1

        if _is_const_value(ctx, b, 0.0) and a in ctx.env:
            out = ctx.graph.relu(_get(ctx, a))
            _bind_consumed_outputs(ctx, nodes, idx, 1, out)
            return [out], 1

    return None


RELU = Pattern(
    name="relu",
    handler=_relu_pattern,
    trigger_ops={"stablehlo.broadcast_in_dim", "stablehlo.maximum"},
)


# ---------------------------------------------------------------------------
# Softmax
# ---------------------------------------------------------------------------

def _try_softmax_13_clamped_casted(ctx: LoweringCtx, nodes: list, idx: int):
    if not (
        _is(nodes, idx, "stablehlo.reduce", 0)
        and _is(nodes, idx, "stablehlo.broadcast_in_dim", 1)
        and _is(nodes, idx, "stablehlo.maximum", 2)
        and _is(nodes, idx, "stablehlo.broadcast_in_dim", 3)
        and _is(nodes, idx, "stablehlo.broadcast_in_dim", 4)
        and _is(nodes, idx, "stablehlo.subtract", 5)
        and _is(nodes, idx, "stablehlo.exponential", 6)
        and _is(nodes, idx, "stablehlo.convert", 7)
        and _is(nodes, idx, "stablehlo.reduce", 8)
        and _is(nodes, idx, "stablehlo.broadcast_in_dim", 9)
        and _is(nodes, idx, "stablehlo.convert", 10)
        and _is(nodes, idx, "stablehlo.broadcast_in_dim", 11)
        and _is(nodes, idx, "stablehlo.divide", 12)
    ):
        return None

    red_max = _op(nodes, idx, 0)
    clamp_bcast = _op(nodes, idx, 1)
    max2 = _op(nodes, idx, 2)
    bmax1 = _op(nodes, idx, 3)
    bmax2 = _op(nodes, idx, 4)
    sub = _op(nodes, idx, 5)
    exp = _op(nodes, idx, 6)
    exp_convert = _op(nodes, idx, 7)
    red_sum = _op(nodes, idx, 8)
    bsum1 = _op(nodes, idx, 9)
    sum_convert = _op(nodes, idx, 10)
    bsum2 = _op(nodes, idx, 11)
    div = _op(nodes, idx, 12)

    if not _reduce_is_max(red_max):
        return None

    if not _reduce_is_add(red_sum):
        return None

    x_ssa = _inp(red_max, 0)
    if x_ssa not in ctx.env:
        return None

    if _out(red_max) not in max2.inputs:
        return None

    if _out(clamp_bcast) not in max2.inputs:
        return None

    if _inp(bmax1, 0) != _out(max2):
        return None

    if _inp(bmax2, 0) != _out(bmax1):
        return None

    if _inp(sub, 0) != x_ssa:
        return None

    if _inp(sub, 1) != _out(bmax2):
        return None

    if _inp(exp, 0) != _out(sub):
        return None

    if _inp(exp_convert, 0) != _out(exp):
        return None

    if _inp(red_sum, 0) != _out(exp_convert):
        return None

    if _inp(bsum1, 0) != _out(red_sum):
        return None

    if _inp(sum_convert, 0) != _out(bsum1):
        return None

    if _inp(bsum2, 0) != _out(sum_convert):
        return None

    if _inp(div, 0) != _out(exp):
        return None

    if _inp(div, 1) != _out(bsum2):
        return None

    axis = _axis_from_reduce(red_max)
    x = _get(ctx, x_ssa)
    if getattr(x, "dtype", None) != 1:
        x = ctx.graph.precision_cast(x, 1)
    out = ctx.graph.softmax(x, axis=axis)

    _bind_consumed_outputs(ctx, nodes, idx, 13, out)
    return [out], 13


def _try_softmax_8_with_maximum(ctx: LoweringCtx, nodes: list, idx: int):
    if not (
        _is(nodes, idx, "stablehlo.reduce", 0)
        and _is(nodes, idx, "stablehlo.maximum", 1)
        and _is(nodes, idx, "stablehlo.broadcast_in_dim", 2)
        and _is(nodes, idx, "stablehlo.subtract", 3)
        and _is(nodes, idx, "stablehlo.exponential", 4)
        and _is(nodes, idx, "stablehlo.reduce", 5)
        and _is(nodes, idx, "stablehlo.broadcast_in_dim", 6)
        and _is(nodes, idx, "stablehlo.divide", 7)
    ):
        return None

    red_max = _op(nodes, idx, 0)
    max2 = _op(nodes, idx, 1)
    bmax = _op(nodes, idx, 2)
    sub = _op(nodes, idx, 3)
    exp = _op(nodes, idx, 4)
    red_sum = _op(nodes, idx, 5)
    bsum = _op(nodes, idx, 6)
    div = _op(nodes, idx, 7)

    if not _reduce_is_max(red_max):
        return None
    if not _reduce_is_add(red_sum):
        return None

    x_ssa = _inp(red_max, 0)
    if x_ssa not in ctx.env:
        return None

    if _out(red_max) not in max2.inputs:
        return None
    if _inp(bmax, 0) != _out(max2):
        return None
    if x_ssa not in sub.inputs or _out(bmax) not in sub.inputs:
        return None
    if _inp(exp, 0) != _out(sub):
        return None
    if _inp(red_sum, 0) != _out(exp):
        return None
    if _inp(bsum, 0) != _out(red_sum):
        return None
    if _out(exp) not in div.inputs or _out(bsum) not in div.inputs:
        return None

    axis = _axis_from_reduce(red_max)
    x = _get(ctx, x_ssa)
    if getattr(x, "dtype", None) != 1:
        x = ctx.graph.precision_cast(x, 1)
    out = ctx.graph.softmax(x, axis=axis)

    _bind_consumed_outputs(ctx, nodes, idx, 8, out)
    return [out], 8


def _try_softmax_7(ctx: LoweringCtx, nodes: list, idx: int):
    if not (
        _is(nodes, idx, "stablehlo.reduce", 0)
        and _is(nodes, idx, "stablehlo.broadcast_in_dim", 1)
        and _is(nodes, idx, "stablehlo.subtract", 2)
        and _is(nodes, idx, "stablehlo.exponential", 3)
        and _is(nodes, idx, "stablehlo.reduce", 4)
        and _is(nodes, idx, "stablehlo.broadcast_in_dim", 5)
        and _is(nodes, idx, "stablehlo.divide", 6)
    ):
        return None

    red_max = _op(nodes, idx, 0)
    bmax = _op(nodes, idx, 1)
    sub = _op(nodes, idx, 2)
    exp = _op(nodes, idx, 3)
    red_sum = _op(nodes, idx, 4)
    bsum = _op(nodes, idx, 5)
    div = _op(nodes, idx, 6)

    if not _reduce_is_max(red_max):
        return None
    if not _reduce_is_add(red_sum):
        return None

    x_ssa = _inp(red_max, 0)
    if x_ssa not in ctx.env:
        return None

    if _inp(bmax, 0) != _out(red_max):
        return None
    if x_ssa not in sub.inputs or _out(bmax) not in sub.inputs:
        return None
    if _inp(exp, 0) != _out(sub):
        return None
    if _inp(red_sum, 0) != _out(exp):
        return None
    if _inp(bsum, 0) != _out(red_sum):
        return None
    if _out(exp) not in div.inputs or _out(bsum) not in div.inputs:
        return None

    axis = _axis_from_reduce(red_max)
    x = _get(ctx, x_ssa)
    if getattr(x, "dtype", None) != 1:
        x = ctx.graph.precision_cast(x, 1)
    out = ctx.graph.softmax(x, axis=axis)

    _bind_consumed_outputs(ctx, nodes, idx, 7, out)
    return [out], 7


def _try_softmax_4_unstable(ctx: LoweringCtx, nodes: list, idx: int):
    if not (
        _is(nodes, idx, "stablehlo.exponential", 0)
        and _is(nodes, idx, "stablehlo.reduce", 1)
        and _is(nodes, idx, "stablehlo.broadcast_in_dim", 2)
        and _is(nodes, idx, "stablehlo.divide", 3)
    ):
        return None

    exp = _op(nodes, idx, 0)
    red_sum = _op(nodes, idx, 1)
    bsum = _op(nodes, idx, 2)
    div = _op(nodes, idx, 3)

    if not _reduce_is_add(red_sum):
        return None

    x_ssa = _inp(exp, 0)
    if x_ssa not in ctx.env:
        return None

    if _inp(red_sum, 0) != _out(exp):
        return None
    if _inp(bsum, 0) != _out(red_sum):
        return None
    if _out(exp) not in div.inputs or _out(bsum) not in div.inputs:
        return None

    axis = _axis_from_reduce(red_sum)
    x = _get(ctx, x_ssa)
    if getattr(x, "dtype", None) != 1:
        x = ctx.graph.precision_cast(x, 1)
    out = ctx.graph.softmax(x, axis=axis)

    _bind_consumed_outputs(ctx, nodes, idx, 4, out)
    return [out], 4


def _softmax_pattern(ctx: LoweringCtx, nodes: list, idx: int):
    result = _try_softmax_13_clamped_casted(ctx, nodes, idx)
    if result is not None:
        return result

    result = _try_softmax_8_with_maximum(ctx, nodes, idx)
    if result is not None:
        return result

    result = _try_softmax_7(ctx, nodes, idx)
    if result is not None:
        return result

    result = _try_softmax_4_unstable(ctx, nodes, idx)
    if result is not None:
        return result

    return None


SOFTMAX = Pattern(
    name="softmax",
    handler=_softmax_pattern,
    trigger_ops={"stablehlo.reduce", "stablehlo.exponential"},
)


# ---------------------------------------------------------------------------
# SiLU / Swish
# ---------------------------------------------------------------------------

def _try_silu_neg_exp_add_div_mul_7(ctx: LoweringCtx, nodes: list, idx: int):
    if not hasattr(ctx.graph, "silu"):
        return None

    if not (
        _is(nodes, idx, "stablehlo.negate", 0)
        and _is(nodes, idx, "stablehlo.exponential", 1)
        and _is(nodes, idx, "stablehlo.broadcast_in_dim", 2)
        and _is(nodes, idx, "stablehlo.add", 3)
        and _is(nodes, idx, "stablehlo.broadcast_in_dim", 4)
        and _is(nodes, idx, "stablehlo.divide", 5)
        and _is(nodes, idx, "stablehlo.multiply", 6)
    ):
        return None

    neg = _op(nodes, idx, 0)
    exp = _op(nodes, idx, 1)
    one_a = _op(nodes, idx, 2)
    add = _op(nodes, idx, 3)
    one_b = _op(nodes, idx, 4)
    div = _op(nodes, idx, 5)
    mul = _op(nodes, idx, 6)

    x_ssa = _inp(neg, 0)
    if x_ssa not in ctx.env:
        return None

    if _inp(exp, 0) != _out(neg):
        return None
    if not _bcast_const_value(ctx, one_a, 1.0):
        return None
    if _out(exp) not in add.inputs or _out(one_a) not in add.inputs:
        return None
    if not _bcast_const_value(ctx, one_b, 1.0):
        return None
    if _out(one_b) != _inp(div, 0):
        return None
    if _out(add) != _inp(div, 1):
        return None
    if x_ssa not in mul.inputs or _out(div) not in mul.inputs:
        return None

    out = ctx.graph.silu(_get(ctx, x_ssa))

    _bind_consumed_outputs(ctx, nodes, idx, 7, out)
    return [out], 7


def _try_silu_neg_exp_add_div_mul_6(ctx: LoweringCtx, nodes: list, idx: int):
    if not hasattr(ctx.graph, "silu"):
        return None

    if not (
        _is(nodes, idx, "stablehlo.negate", 0)
        and _is(nodes, idx, "stablehlo.exponential", 1)
        and _is(nodes, idx, "stablehlo.broadcast_in_dim", 2)
        and _is(nodes, idx, "stablehlo.add", 3)
        and _is(nodes, idx, "stablehlo.divide", 4)
        and _is(nodes, idx, "stablehlo.multiply", 5)
    ):
        return None

    neg = _op(nodes, idx, 0)
    exp = _op(nodes, idx, 1)
    one = _op(nodes, idx, 2)
    add = _op(nodes, idx, 3)
    div = _op(nodes, idx, 4)
    mul = _op(nodes, idx, 5)

    x_ssa = _inp(neg, 0)
    if x_ssa not in ctx.env:
        return None

    if _inp(exp, 0) != _out(neg):
        return None
    if not _bcast_const_value(ctx, one, 1.0):
        return None
    if _out(exp) not in add.inputs or _out(one) not in add.inputs:
        return None
    if _out(one) != _inp(div, 0):
        return None
    if _out(add) != _inp(div, 1):
        return None
    if x_ssa not in mul.inputs or _out(div) not in mul.inputs:
        return None

    out = ctx.graph.silu(_get(ctx, x_ssa))

    _bind_consumed_outputs(ctx, nodes, idx, 6, out)
    return [out], 6


def _try_silu_logistic_mul_2(ctx: LoweringCtx, nodes: list, idx: int):
    if not hasattr(ctx.graph, "silu"):
        return None

    if not (
        _is(nodes, idx, "stablehlo.logistic", 0)
        and _is(nodes, idx, "stablehlo.multiply", 1)
    ):
        return None

    logistic = _op(nodes, idx, 0)
    mul = _op(nodes, idx, 1)

    x_ssa = _inp(logistic, 0)
    if x_ssa not in ctx.env:
        return None

    if x_ssa not in mul.inputs or _out(logistic) not in mul.inputs:
        return None

    out = ctx.graph.silu(_get(ctx, x_ssa))

    _bind_consumed_outputs(ctx, nodes, idx, 2, out)
    return [out], 2


def _silu_pattern(ctx: LoweringCtx, nodes: list, idx: int):
    result = _try_silu_neg_exp_add_div_mul_7(ctx, nodes, idx)
    if result is not None:
        return result

    result = _try_silu_neg_exp_add_div_mul_6(ctx, nodes, idx)
    if result is not None:
        return result

    result = _try_silu_logistic_mul_2(ctx, nodes, idx)
    if result is not None:
        return result

    return None


SILU = Pattern(
    name="silu",
    handler=_silu_pattern,
    trigger_ops={"stablehlo.negate", "stablehlo.logistic"},
)


# ---------------------------------------------------------------------------
# Attention softmax fusion (qk -> softmax -> @v)
# ---------------------------------------------------------------------------

def _try_attention_from_softmax_window(ctx: LoweringCtx, nodes: list, idx: int):
    """
    Fuse common StableHLO attention window:

      scores = (q @ k^T) * scale
      masked = select(mask, scores, -inf/large-neg)
      probs = softmax(masked)
      out = probs @ v

    This matcher anchors on `stablehlo.exponential` in the softmax chain and
    only fires when the downstream consumer is a `stablehlo.dot_general`.
    """
    exp = _op(nodes, idx, 0)
    if exp is None or exp.op != "stablehlo.exponential":
        return None

    c0 = _op(nodes, idx, 1)
    red = _op(nodes, idx, 2)
    b0 = _op(nodes, idx, 3)
    c1 = _op(nodes, idx, 4)
    b1 = _op(nodes, idx, 5)
    div = _op(nodes, idx, 6)
    attn = _op(nodes, idx, 7)
    if not (c0 and red and b0 and c1 and b1 and div and attn):
        return None
    if c0.op != "stablehlo.convert":
        return None
    if red.op != "stablehlo.reduce":
        return None
    if b0.op != "stablehlo.broadcast_in_dim":
        return None
    if c1.op != "stablehlo.convert":
        return None
    if b1.op != "stablehlo.broadcast_in_dim":
        return None
    if div.op != "stablehlo.divide":
        return None
    if attn.op != "stablehlo.dot_general":
        return None
    if _inp(div, 0) != _out(exp):
        return None
    if _inp(attn, 0) != _out(div):
        return None

    # Walk back to masked score select.
    sub_prod = _producer_node(nodes, _inp(exp, 0))
    if sub_prod is None:
        return None
    _, sub = sub_prod
    if sub.op != "stablehlo.subtract":
        return None

    masked_ssa = _inp(sub, 0)
    masked_prod = _producer_node(nodes, masked_ssa)
    if masked_prod is None:
        return None
    _, sel = masked_prod
    if sel.op != "stablehlo.select":
        return None

    # One select input should be score tensor from multiply(scale, qk_dot).
    score_mul = None
    for ssa in sel.inputs:
        p = _producer_node(nodes, ssa)
        if p is None:
            continue
        _, n = p
        if n.op == "stablehlo.multiply":
            # Prefer multiply that consumes a dot_general.
            a0 = _producer_node(nodes, _inp(n, 0))
            a1 = _producer_node(nodes, _inp(n, 1))
            if (a0 and a0[1].op == "stablehlo.dot_general") or (a1 and a1[1].op == "stablehlo.dot_general"):
                score_mul = n
                break
    if score_mul is None:
        return None

    qk_dot = None
    scale = 1.0
    for ssa in score_mul.inputs:
        p = _producer_node(nodes, ssa)
        if p and p[1].op == "stablehlo.dot_general":
            qk_dot = p[1]
        else:
            # Try scalar const through bcast/convert.
            base = _trace_back_through_unary(nodes, ssa)
            cval = _const_float(ctx, base)
            if cval is not None and abs(float(cval)) > 0:
                scale = float(cval)
    if qk_dot is None:
        return None

    q_ssa = _inp(qk_dot, 0)
    k_ssa = _inp(qk_dot, 1)
    v_ssa = _inp(attn, 1)
    if q_ssa not in ctx.env or k_ssa not in ctx.env or v_ssa not in ctx.env:
        return None

    q = _get(ctx, q_ssa)
    k = _get(ctx, k_ssa)
    v = _get(ctx, v_ssa)

    out = ctx.graph.attention(q, k, v, float(scale))

    # Bind the whole softmax window and attention output.
    _bind_consumed_outputs(ctx, nodes, idx, 8, out)
    ctx.env[_out(attn)] = out
    return [out], 8


ATTENTION = Pattern(
    name="attention_softmax",
    handler=_try_attention_from_softmax_window,
    trigger_ops={"stablehlo.exponential"},
)


# ---------------------------------------------------------------------------
# Exported pattern list
# ---------------------------------------------------------------------------

DEFAULT_PATTERNS: list[Pattern] = [
    # Larger/more specific patterns first.
    RMSNORM,
    GELU,
    SILU,
    RELU,
    ATTENTION,
    SOFTMAX,
]
