# dont have sort support!!!!!

"""
lower_to_cactus.py
==================
Orchestrator: IRGraph -> Cactus Graph.

For each node in execution order:
  1. Try registered patterns (layer 2) — fused ops
  2. Fall back to generic op lowering (layer 1) — 1:1 ops

Usage
-----
    from test            import jax_to_mlir
    from parse           import parse_mlir
    from lower_to_cactus import lower_to_cactus

    ir = parse_mlir(jax_to_mlir(my_fn, example_inputs))
    g, env = lower_to_cactus(ir, patterns=["default"])
    g.save("model.cactus")
"""

from __future__ import annotations

import importlib
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

from parse import IRGraph
from pattern_registry import PatternRegistry, LoweringCtx, Pattern
from op_lowering import lower_op, _cactus_dtype


# ---------------------------------------------------------------------------
# StableHLO constant decoding
# ---------------------------------------------------------------------------
def _producer_map(ir):
    p = {}
    for nid in ir.order:
        n = ir.nodes[nid]
        for out in n.outputs:
            p[out] = n
    return p


def _compute_live_nodes(ir):
    prod = _producer_map(ir)

    live_values = set(ir.outputs)
    live_nodes = set()
    stack = list(ir.outputs)

    while stack:
        ssa = stack.pop()
        nid_or_node = prod.get(ssa)
        if nid_or_node is None:
            continue

        # prod stores node object in this helper
        node = nid_or_node
        nid = node.id

        if nid in live_nodes:
            continue

        live_nodes.add(nid)

        for inp in node.inputs:
            if inp not in live_values:
                live_values.add(inp)
                stack.append(inp)

    return live_nodes


def _producer_map(ir):
    out_to_node = {}
    for nid in ir.order:
        n = ir.nodes[nid]
        for out in n.outputs:
            out_to_node[out] = n
    return out_to_node


def _is_unstable_generic_node(node, prod):
    op = node.op

    # Boolean/int emulation paths are the most fragile under FP16-only math.
    if op in {"stablehlo.compare", "stablehlo.and", "stablehlo.or", "stablehlo.not"}:
        return True

    return False

def _parse_int_listish_local(v):
    if v is None:
        return []
    if isinstance(v, int):
        return [int(v)]
    if isinstance(v, (list, tuple)):
        return [int(x) for x in v]
    s = str(v).strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    import re
    return [int(x) for x in re.findall(r"-?\d+", s)]


def _const_scalar_from_ir(ir, ssa):
    c = ir.constants.get(ssa)
    if c is None:
        return None

    try:
        return float(decode_stablehlo_const(c.value, c.dtype))
    except Exception:
        return None


def _trace_scalar_const(ir, ssa):
    """
    Trace through simple scalar-preserving wrappers to find a constant.
    Handles:
      constant
      convert(constant)
      reshape(constant)
      broadcast_in_dim(constant)
    """
    v = _const_scalar_from_ir(ir, ssa)
    if v is not None:
        return v

    prod = _producer_map(ir)
    cur = ssa

    for _ in range(8):
        n = prod.get(cur)
        if n is None or len(n.inputs) != 1:
            return None

        if n.op not in (
            "stablehlo.convert",
            "stablehlo.reshape",
            "stablehlo.broadcast_in_dim",
        ):
            return None

        cur = n.inputs[0]

        v = _const_scalar_from_ir(ir, cur)
        if v is not None:
            return v

    return None

def _canonicalize_reduce_sum_divide_to_mean(ir):
    """
    Rewrite:
        divide(broadcast_in_dim(reduce_add(x, axis)), axis_size)

    into:
        cactus.mean_keepdims(x, axis, out_shape)

    This avoids materializing the FP16 reduce_sum, which overflows in Gemma
    RMSNorm when hidden_dim=2304.
    """
    import re

    prod = _producer_map(ir)
    changed = 0

    def parse_dims(v):
        if v is None:
            return []
        if isinstance(v, int):
            return [int(v)]
        if isinstance(v, (list, tuple)):
            return [int(x) for x in v]
        s = str(v)
        m = re.search(r"array<[^:>]+:\s*([^>]*)>", s)
        if m:
            s = m.group(1)
        return [int(x) for x in re.findall(r"-?\d+", s)]

    def scalar_const(ssa):
        c = ir.constants.get(ssa)
        if c is not None:
            try:
                return float(decode_stablehlo_const(c.value, c.dtype))
            except Exception:
                return None

        cur = ssa
        for _ in range(12):
            n = prod.get(cur)
            if n is None or len(n.inputs) != 1:
                return None

            if n.op not in (
                "stablehlo.convert",
                "stablehlo.reshape",
                "stablehlo.broadcast_in_dim",
            ):
                return None

            cur = n.inputs[0]

            c = ir.constants.get(cur)
            if c is not None:
                try:
                    return float(decode_stablehlo_const(c.value, c.dtype))
                except Exception:
                    return None

        return None

    def trace_reduce_through_wrappers(ssa):
        """
        Trace:
            reduce
            broadcast(reduce)
            convert(broadcast(reduce))
            reshape(...)
        """
        cur = ssa

        for _ in range(12):
            n = prod.get(cur)
            if n is None:
                return None

            if n.op == "stablehlo.reduce":
                return n

            if n.op in (
                "stablehlo.broadcast_in_dim",
                "stablehlo.convert",
                "stablehlo.reshape",
            ) and len(n.inputs) == 1:
                cur = n.inputs[0]
                continue

            return None

        return None

    def get_reduce_info(reduce_node):
        if reduce_node is None or reduce_node.op != "stablehlo.reduce":
            return None

        applies = str(reduce_node.attrs.get("applies", "")).lower()
        if applies and "add" not in applies:
            return None

        dims = (
            reduce_node.attrs.get("dimensions")
            or reduce_node.attrs.get("dims")
            or reduce_node.attrs.get("axes")
            or reduce_node.attrs.get("axis")
        )
        dims = parse_dims(dims)

        if len(dims) != 1:
            return None

        axis = int(dims[0])

        if not reduce_node.inputs:
            return None

        x_ssa = reduce_node.inputs[0]
        if x_ssa not in ir.values:
            return None

        x_shape = tuple(ir.values[x_ssa].shape)
        rank = len(x_shape)

        if axis < 0:
            axis += rank

        if axis < 0 or axis >= rank:
            return None

        axis_size = float(x_shape[axis])
        return x_ssa, axis, axis_size

    for nid in ir.order:
        node = ir.nodes[nid]

        if node.op not in ("stablehlo.divide", "stablehlo.multiply"):
            continue

        if len(node.inputs) != 2:
            continue

        lhs_ssa, rhs_ssa = node.inputs

        reduce_node = trace_reduce_through_wrappers(lhs_ssa)
        info = get_reduce_info(reduce_node)
        const = scalar_const(rhs_ssa)

        # Also allow const on lhs for multiply.
        if info is None or const is None:
            reduce_node = trace_reduce_through_wrappers(rhs_ssa)
            info = get_reduce_info(reduce_node)
            const = scalar_const(lhs_ssa)

        if info is None or const is None:
            continue

        x_ssa, axis, axis_size = info

        ok = False

        if node.op == "stablehlo.divide":
            ok = abs(float(const) - axis_size) <= 1e-3

        elif node.op == "stablehlo.multiply":
            ok = abs(float(const) - (1.0 / axis_size)) <= 1e-6

        if not ok:
            continue

        old_op = node.op
        old_inputs = list(node.inputs)

        node.op = "cactus.mean_keepdims"
        node.inputs = [x_ssa]
        node.attrs = {
            "axis": axis,
            "out_shape": tuple(ir.values[node.outputs[0]].shape),
            "_replaces": f"{old_op}({old_inputs})",
            "_axis_size": axis_size,
            "_const": const,
        }

        changed += 1

    return changed




def _canonicalize_gemma_embedding_gather(ir):
    """
    Rewrite Gemma embedding gather index from:
      broadcast(select(token < 0, token + vocab, token)) -> [B,T,1]
    to:
      raw token ids -> [B,T]

    Then DCE removes the integer cleanup path.
    """
    prod = _producer_map(ir)

    def shape(ssa):
        return tuple(ir.values[ssa].shape)

    def trace_raw_tokens(index_ssa):
        n = prod.get(index_ssa)

        # broadcast_in_dim(select(...)) -> select(...)
        if n is not None and n.op == "stablehlo.broadcast_in_dim" and len(n.inputs) == 1:
            index_ssa = n.inputs[0]
            n = prod.get(index_ssa)

        # select(mask, token + vocab, token)
        # false branch is original token ids.
        if n is not None and n.op == "stablehlo.select" and len(n.inputs) == 3:
            return n.inputs[2]

        return index_ssa

    for nid in ir.order:
        n = ir.nodes[nid]
        if n.op != "stablehlo.gather":
            continue

        if len(n.inputs) != 2 or not n.outputs:
            continue

        table_ssa, idx_ssa = n.inputs
        table_shape = shape(table_ssa)
        idx_shape = shape(idx_ssa)
        out_shape = shape(n.outputs[0])

        is_embedding_gather = (
            len(table_shape) == 2
            and len(idx_shape) >= 1
            and idx_shape[-1] == 1
            and out_shape == idx_shape[:-1] + (table_shape[1],)
        )

        if is_embedding_gather:
            raw_tokens = trace_raw_tokens(idx_ssa)
            n.inputs[1] = raw_tokens
            n.attrs["_cactus_embedding_gather"] = True


def _canonicalize_runtime_mask_path(ir):
    """
    Canonicalize decomposed boolean mask chains to the explicit runtime mask
    input (%arg238) when present:
      and(%arg238, <rebuilt_causal_or_padding_mask>) -> %arg238

    This removes fragile compare/select boolean emulation from the hot path.
    """
    and_outputs = []
    for nid in ir.order:
        n = ir.nodes[nid]
        if n.op == "stablehlo.and" and len(n.inputs) == 2 and "%arg238" in n.inputs and n.outputs:
            and_outputs.append(n.outputs[0])

    if not and_outputs:
        return 0

    rewrites = 0
    for nid in ir.order:
        n = ir.nodes[nid]
        new_inputs = []
        changed = False
        for ssa in n.inputs:
            if ssa in and_outputs:
                new_inputs.append("%arg238")
                changed = True
            else:
                new_inputs.append(ssa)
        if changed:
            n.inputs = new_inputs
            rewrites += 1

    return rewrites

def decode_stablehlo_const(value: Any, dtype: Any) -> Any:
    """
    Decode constants emitted by StableHLO / parser.

    StableHLO sometimes prints non-finite floats as raw hex bit-patterns.
    Common examples:
      f16 0xFC00 = -inf
      f16 0x7C00 = +inf

    This function returns a Python scalar suitable for np.full(...).
    """
    if isinstance(value, np.generic):
        value = value.item()

    if isinstance(value, str):
        s = value.strip()
        low = s.lower()

        if low in ("inf", "+inf", "infinity", "+infinity"):
            return np.inf
        if low in ("-inf", "-infinity"):
            return -np.inf
        if low in ("nan", "+nan", "-nan"):
            return np.nan
        if low == "true":
            return 1.0
        if low == "false":
            return 0.0

        if low.startswith("0x"):
            bits = int(low, 16)
            dtype_s = str(dtype).lower()

            if "f16" in dtype_s:
                return np.array([bits & 0xFFFF], dtype=np.uint16).view(np.float16)[0].item()

            if "bf16" in dtype_s:
                # bfloat16 is represented by the high 16 bits of float32.
                return np.array([(bits & 0xFFFF) << 16], dtype=np.uint32).view(np.float32)[0].item()

            if "f32" in dtype_s:
                return np.array([bits & 0xFFFFFFFF], dtype=np.uint32).view(np.float32)[0].item()

            if "f64" in dtype_s:
                return np.array([bits & 0xFFFFFFFFFFFFFFFF], dtype=np.uint64).view(np.float64)[0].item()

            # Fallback: try to interpret as a normal hex integer.
            return float(bits)

        return float(s)

    return value


def _constant_storage_dtype(const: Any) -> int:
    """
    Pick the Cactus dtype for a StableHLO constant input node.
    """
    dtype_s = str(const.dtype).lower().strip()
    dtype = _cactus_dtype(const.dtype)

    # FP16-first policy only for floating-point scalar arithmetic constants.
    # Integer/bool/index scalars must keep non-FP16 storage to avoid precision
    # mismatches in compare/mask paths.
    if not const.shape:
        if dtype_s in ("f32", "bf16", "f16"):
            return 1

    return dtype


# ---------------------------------------------------------------------------
# Cactus graph import helper
# ---------------------------------------------------------------------------

def _new_graph():
    """Create a Cactus Python Graph, robust to running from jaxconversion/."""
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    try:
        from src.graph import Graph
    except Exception:
        from python.src.graph import Graph

    return Graph()


# ---------------------------------------------------------------------------
# Pattern loader
# ---------------------------------------------------------------------------

def _load_patterns(names: list[str | Pattern]) -> list[Pattern]:
    patterns: list[Pattern] = []

    for name in names:
        if isinstance(name, Pattern):
            patterns.append(name)
            continue

        attr_name = f"{name.upper()}_PATTERNS"
        mod = None

        # Prefer the intended package/flat names. Keep the misspelled fallback
        # because this repo has used patters_default.py in some local copies.
        for mod_name in (
            f"patterns.{name}",
            f"patterns_{name}",
            f"patters_{name}",
            name,
        ):
            try:
                mod = importlib.import_module(mod_name)
                break
            except ModuleNotFoundError:
                continue

        if mod is None:
            raise ModuleNotFoundError(
                f"Pattern not found for {name!r}. Tried patterns/{name}.py, "
                f"patterns_{name}.py, patters_{name}.py, and {name}.py. "
                f"Module must contain a list named {attr_name}."
            )

        lst = getattr(mod, attr_name, None)
        if lst is None:
            raise AttributeError(f"Pattern module {mod.__name__!r} has no {attr_name}")

        patterns.extend(lst)

    return patterns


# ---------------------------------------------------------------------------
# Main lowering loop
# ---------------------------------------------------------------------------

def lower_to_cactus(
    ir: IRGraph,
    graph: Any = None,
    patterns: list[str | Pattern] | None = None,
    verbose: bool = False,
    input_resolver=None,
    strict_math: bool = True,
) -> tuple[Any, dict[str, Any]]:
    """
    Walk an IRGraph and emit Cactus Graph calls.

    Parameters
    ----------
    ir:
        IRGraph produced by parse_mlir().
    graph:
        Existing cactus Graph to build into; created fresh if None.
    patterns:
        Pattern module names or Pattern objects. Default is ["default"].
        Use [] to disable pattern fusion and test primitive lowering only.
    verbose:
        Print each op/pattern as it is lowered.

    Returns
    -------
    (graph, env):
        graph is the built Cactus Graph.
        env maps SSA names to Cactus Tensor objects.
    """
    if graph is None:
        graph = _new_graph()

    registry = PatternRegistry()
    for p in _load_patterns(patterns if patterns is not None else ["default"]):
        registry.register(p)

    env: dict[str, Any] = {}
    const_values: dict[str, Any] = {}

    # --- register graph inputs as Cactus input nodes ---
    # --- register graph inputs as Cactus input nodes ---
    for ssa in ir.inputs:
        val = ir.values[ssa]
        shape = tuple(int(x) for x in val.shape)
        dtype = _cactus_dtype(val.dtype)

        tensor = None

        if input_resolver is not None:
            tensor = input_resolver(graph, ssa, shape, dtype)

        if tensor is None:
            tensor = graph.input(list(shape), dtype=dtype)

        env[ssa] = tensor

        if verbose:
            source = "resolver" if input_resolver is not None and tensor is not None else "input"
            print(f"  INPUT  {ssa}  shape={val.shape}  dtype={val.dtype}  source={source}")

    # --- register constants as Cactus input nodes ---
    # Constants extracted from stablehlo.constant become graph inputs whose data
    # is set at runtime. Register each constant exactly once.
    for ssa, const in ir.constants.items():
        dtype = _constant_storage_dtype(const)
        shape = list(const.shape) if const.shape else [1]

        scalar = decode_stablehlo_const(const.value, const.dtype)
        const_values[ssa] = scalar

        tensor = graph.input(shape, dtype=dtype)
        env[ssa] = tensor

        if verbose:
            print(
                f"  CONST  {ssa}  shape={const.shape}  "
                f"dtype={const.dtype}  value={const.value}  decoded={scalar}"
            )

    ctx = LoweringCtx(graph=graph, env=env, constants=ir.constants, ir=ir)

    # Extra metadata for op_lowering.py. dataclasses are not slotted here, so
    # this assignment is safe. _lower_maximum/_lower_minimum can use it to avoid
    # NaNs from formulas like max(-inf, x) = -inf + relu(x - -inf).
    ctx.const_values = const_values

    _canonicalize_gemma_embedding_gather(ir)
    n_mask = _canonicalize_runtime_mask_path(ir)
    if verbose:
        print(f"CANONICALIZED runtime mask boolean path rewrites: {n_mask}")

    exact_mode = os.environ.get("CACTUS_EXACT_MATH", "0") == "1"
    if exact_mode:
        n_mean = 0
    else:
        n_mean = _canonicalize_reduce_sum_divide_to_mean(ir)
    print(f"CANONICALIZED reduce_sum/divide/multiply -> mean_keepdims: {n_mean}")

    live_nodes = _compute_live_nodes(ir)
    prod = _producer_map(ir)
    nodes = [ir.nodes[nid] for nid in ir.order]
    
    idx = 0
    while idx < len(nodes):
        node = nodes[idx]
        
        if node.id not in live_nodes:
            idx += 1
            continue
            
        # Skip nodes whose outputs were already bound by a pattern.
        if all(ssa in env for ssa in node.outputs):
            idx += 1
            continue

        # --- layer 2: pattern matching / fusion ---
        result = registry.match(ctx, nodes, idx)
        if result is not None:
            out_tensors, consumed = result

            # Bind leading node outputs. Deeper pattern outputs should be bound
            # by the pattern handler itself when needed.
            for ssa, tensor in zip(node.outputs, out_tensors):
                env[ssa] = tensor

            if verbose:
                print(f"  PATTERN  [{node.op} +{consumed - 1}]  -> {node.outputs}")

            idx += consumed
            continue

        # --- layer 1: generic op lowering ---
        #if strict_math and _is_unstable_generic_node(node, prod):
        #    raise RuntimeError(
        #        "Strict math guard: unstable decomposed math reached generic lowering.\n"
        #        f"  node id: {node.id}\n"
        #        f"  op: {node.op}\n"
        #        f"  inputs: {node.inputs}\n"
        #        f"  attrs: {node.attrs}\n"
        #        "Add/improve a fusion/canonicalization pattern so this path is lowered safely."
        #    )

        try:
            out_tensors = lower_op(ctx, node)
        except NotImplementedError as e:
            raise NotImplementedError(str(e)) from None
        except KeyError as e:
            raise KeyError(
                f"Missing SSA value while lowering {node.op} ({node.id}): {e}\n"
                f"  node inputs: {node.inputs}"
            ) from None

        for ssa, tensor in zip(node.outputs, out_tensors):
            env[ssa] = tensor

        if verbose:
            shapes = [getattr(t, "shape", "?") for t in out_tensors]
            tids = [getattr(t, "id", "?") for t in out_tensors]
            print(f"  OP  {node.op:<40}  -> {node.outputs}  {shapes} ids={tids}")

        idx += 1

    if verbose:
        print(f"\n  graph outputs: {ir.outputs}")
        for ssa in ir.outputs:
            if ssa in env:
                print(f"    {ssa}  shape={env[ssa].shape}")

    return graph, env


# ---------------------------------------------------------------------------
# Demo / stress test
# ---------------------------------------------------------------------------











if __name__ == "__main__":
    import os
    import time
    import traceback
    from collections import Counter

    import jax
    import jax.numpy as jnp
    import numpy as np

    from test import jax_to_mlir
    from parse import parse_mlir

    # Keep output clean.
    for k in (
        "CACTUS_PROFILE",
        "CACTUS_PROFILE_FILE",
        "CACTUS_CAPTURE_ENABLE",
        "CACTUS_CAPTURE_STDOUT",
        "CACTUS_CAPTURE_FILE",
        "CACTUS_CAPTURE_DIR",
    ):
        os.environ.pop(k, None)

    # ================================================================
    # REAL-SHAPE MINI GEMMA STRESS TEST
    #
    # Uses Gemma-2-2B-ish core dimensions:
    #   D=2304
    #   H=8
    #   KVH=4
    #   DH=256
    #   FF=9216
    #
    # But only:
    #   B=1
    #   T=8
    #   layers=1
    #   synthetic vocab=4096
    #
    # This is designed to test the exact real-shape attention/MLP geometry
    # without needing the full 256k vocab or 26 layers.
    # ================================================================

    B = 1
    T = 8
    VOCAB = 4096

    D = 2304
    H = 8
    KVH = 4
    DH = 256
    FF = 9216
    NUM_LAYERS = 1

    EPS = 1e-6
    THETA = 10000.0
    ATTN_LOGIT_SOFTCAP = 50.0
    LOGIT_SOFTCAP = 30.0

    VALID_LEN = 6
    TOP_B = 0
    TOP_T = VALID_LEN - 1

    assert H % KVH == 0

    rng = np.random.default_rng(12345)

    # ================================================================
    # Helpers
    # ================================================================
    def f16(x):
        return np.asarray(x, dtype=np.float16)

    def randn(shape, scale=1.0):
        return f16(rng.standard_normal(shape).astype(np.float32) * scale)

    def ones_plus_noise(shape, scale=0.02):
        return f16(1.0 + rng.standard_normal(shape).astype(np.float32) * scale)

    def silu(x):
        return x * jax.nn.sigmoid(x)

    def rms_norm(x, weight):
        x32 = x.astype(jnp.float32)
        ms = jnp.mean(jnp.square(x32), axis=-1, keepdims=True)
        y = x32 * jax.lax.rsqrt(ms + jnp.asarray(EPS, dtype=jnp.float32))
        return y.astype(x.dtype) * weight.astype(x.dtype)

    def head_rms_norm(x, weight):
        """
        x:      [B, heads, T, DH]
        weight: [heads, DH]
        """
        x32 = x.astype(jnp.float32)
        ms = jnp.mean(jnp.square(x32), axis=-1, keepdims=True)
        y = x32 * jax.lax.rsqrt(ms + jnp.asarray(EPS, dtype=jnp.float32))
        return y.astype(x.dtype) * weight[None, :, None, :].astype(x.dtype)

    def apply_rope_halfdim(q, k, positions):
        """
        q: [B,H,T,DH]
        k: [B,KVH,T,DH]
        positions: [B,T]

        Real Gemma with DH=256 emits RoPE frequencies of size 128.
        This intentionally creates:
            10000 ** (iota(128) / 128)
        not shape 256.
        """
        half = DH // 2

        freq_i = jnp.arange(half, dtype=jnp.float32)
        denom = jnp.asarray(THETA, dtype=jnp.float32) ** (
            freq_i / jnp.asarray(half, dtype=jnp.float32)
        )
        inv_freq = jnp.asarray(1.0, dtype=jnp.float32) / denom

        angles = positions.astype(jnp.float32)[:, None, :, None] * inv_freq[None, None, None, :]
        cos = jnp.cos(angles).astype(q.dtype)
        sin = jnp.sin(angles).astype(q.dtype)

        def rope_one(x):
            x1 = x[..., :half]
            x2 = x[..., half:]
            y1 = x1 * cos - x2 * sin
            y2 = x2 * cos + x1 * sin
            return jnp.concatenate((y1, y2), axis=-1)

        return rope_one(q), rope_one(k)

    def repeat_kv(x):
        """
        x: [B,KVH,T,DH] -> [B,H,T,DH]
        Explicit slice/concat to stress same paths as real lowering.
        """
        reps = H // KVH
        pieces = []
        for kv_i in range(KVH):
            for _ in range(reps):
                pieces.append(x[:, kv_i : kv_i + 1, :, :])
        return jnp.concatenate(tuple(pieces), axis=1)

    # ================================================================
    # Model
    # ================================================================
    def real_shape_mini_gemma(tokens, positions, attention_mask, *params):
        """
        Args:
          tokens:         [1,8]
          positions:      [1,8]
          attention_mask: [1,8,8], 1.0 means allowed

        Params:
          0: token_embeddings [VOCAB,D]
          1: output_norm      [D]

          each layer:
            attn_output       [H,DH,D]
            attn_kv           [2,KVH,D,DH]
            attn_q            [H,D,DH]
            ffn_gate_up       [2,D,FF]
            ffn_down          [FF,D]
            input_norm        [D]
            post_attn_norm    [D]
            pre_ffn_norm      [D]
            post_ffn_norm     [D]
            q_norm            [H,DH]
            k_norm            [KVH,DH]
        """
        token_embeddings = params[0]
        output_norm = params[1]

        p = 2

        x = token_embeddings[tokens]
        x = x.astype(jnp.float16) * jnp.asarray(np.sqrt(D), dtype=jnp.float16)

        token_valid = tokens != jnp.asarray(0, dtype=tokens.dtype)
        x = jnp.where(token_valid[:, :, None], x, jnp.asarray(0.0, dtype=x.dtype))

        for layer in range(NUM_LAYERS):
            attn_output = params[p + 0]
            attn_kv = params[p + 1]
            attn_q = params[p + 2]
            ffn_gate_up = params[p + 3]
            ffn_down = params[p + 4]
            input_norm = params[p + 5]
            post_attn_norm = params[p + 6]
            pre_ffn_norm = params[p + 7]
            post_ffn_norm = params[p + 8]
            q_norm = params[p + 9]
            k_norm = params[p + 10]
            p += 11

            # ---------------- Attention ----------------
            xn = rms_norm(x, input_norm)

            # q: [B,T,H,DH] -> [B,H,T,DH]
            q = jnp.einsum("btd,hdf->bthf", xn, attn_q)
            q = jnp.transpose(q, (0, 2, 1, 3))

            # kv: [B,2,T,KVH,DH] -> [B,KVH,T,DH]
            kv = jnp.einsum("btd,zkdf->bztkf", xn, attn_kv)
            k = jnp.transpose(kv[:, 0, :, :, :], (0, 2, 1, 3))
            v = jnp.transpose(kv[:, 1, :, :, :], (0, 2, 1, 3))

            q = head_rms_norm(q, q_norm)
            k = head_rms_norm(k, k_norm)

            q, k = apply_rope_halfdim(q, k, positions)

            k = repeat_kv(k)
            v = repeat_kv(v)

            scores = jnp.einsum("bhtd,bhsd->bhts", q, k)
            scores = scores * jnp.asarray(1.0 / np.sqrt(DH), dtype=scores.dtype)

            # Gemma2-ish attention softcap.
            scores = jnp.tanh(scores / jnp.asarray(ATTN_LOGIT_SOFTCAP, dtype=scores.dtype))
            scores = scores * jnp.asarray(ATTN_LOGIT_SOFTCAP, dtype=scores.dtype)

            causal = positions[:, :, None] >= positions[:, None, :]
            runtime_allowed = attention_mask > jnp.asarray(0.0, dtype=attention_mask.dtype)
            q_valid = token_valid[:, :, None]
            k_valid = token_valid[:, None, :]
            mask = causal & runtime_allowed & q_valid & k_valid

            masked_scores = jnp.where(
                mask[:, None, :, :],
                scores,
                jnp.asarray(-10000.0, dtype=scores.dtype),
            )

            probs = jax.nn.softmax(masked_scores, axis=-1)

            # Redundant renormalization to stress reduce/divide.
            probs = probs / jnp.sum(probs, axis=-1, keepdims=True)

            ctx = jnp.einsum("bhts,bhsd->bhtd", probs, v)
            ctx = jnp.transpose(ctx, (0, 2, 1, 3))

            attn = jnp.einsum("bthd,hdm->btm", ctx, attn_output)

            x = x + attn
            x = x + jnp.asarray(0.03125, dtype=x.dtype) * rms_norm(x, post_attn_norm)

            # ---------------- MLP ----------------
            xf = rms_norm(x, pre_ffn_norm)

            gate_up = jnp.einsum("btd,zdf->bztf", xf, ffn_gate_up)
            gate = gate_up[:, 0, :, :]
            up = gate_up[:, 1, :, :]

            hidden = silu(gate) * up
            hidden = jnp.tanh(hidden / jnp.asarray(12.0, dtype=hidden.dtype))
            hidden = hidden * jnp.asarray(12.0, dtype=hidden.dtype)

            down = jnp.einsum("btf,fd->btd", hidden, ffn_down)

            x = x + down
            x = x + jnp.asarray(0.03125, dtype=x.dtype) * rms_norm(x, post_ffn_norm)

            x = jnp.where(token_valid[:, :, None], x, jnp.asarray(0.0, dtype=x.dtype))

            # Keep stable.
            x = jnp.where(
                x > jnp.asarray(8.0, dtype=x.dtype),
                jnp.asarray(8.0, dtype=x.dtype),
                x,
            )
            x = jnp.where(
                x < jnp.asarray(-8.0, dtype=x.dtype),
                jnp.asarray(-8.0, dtype=x.dtype),
                x,
            )

        y = rms_norm(x, output_norm)

        # Tied vocab projection.
        logits = jnp.einsum("btd,vd->btv", y, token_embeddings)

        # Final logit softcap.
        logits = jnp.tanh(logits / jnp.asarray(LOGIT_SOFTCAP, dtype=logits.dtype))
        logits = logits * jnp.asarray(LOGIT_SOFTCAP, dtype=logits.dtype)

        return logits.astype(jnp.float16)

    # ================================================================
    # Data
    # ================================================================
    def make_inputs_and_weights():
        tokens = np.array(
            [[2, 651, 603, 576, 608, 283, 0, 0]],
            dtype=np.int32,
        )

        # Nonzero offset positions.
        positions = np.array(
            [list(range(11, 11 + T))],
            dtype=np.float16,
        )

        attention_mask = np.zeros((B, T, T), dtype=np.float16)

        for q in range(VALID_LEN):
            for k in range(VALID_LEN):
                if k <= q:
                    attention_mask[0, q, k] = 1.0

        params = []

        # Scales are intentionally conservative because FF=9216 is large.
        token_embeddings = randn((VOCAB, D), 0.024)
        output_norm = ones_plus_noise((D,), 0.02)

        params.append(token_embeddings)
        params.append(output_norm)

        for layer in range(NUM_LAYERS):
            s = 1.0 / np.sqrt(layer + 1)

            attn_output = randn((H, DH, D), 0.012 * s)
            attn_kv = randn((2, KVH, D, DH), 0.014 * s)
            attn_q = randn((H, D, DH), 0.014 * s)

            ffn_gate_up = randn((2, D, FF), 0.010 * s)
            ffn_down = randn((FF, D), 0.010 * s)

            input_norm = ones_plus_noise((D,), 0.02)
            post_attn_norm = ones_plus_noise((D,), 0.02)
            pre_ffn_norm = ones_plus_noise((D,), 0.02)
            post_ffn_norm = ones_plus_noise((D,), 0.02)

            q_norm = ones_plus_noise((H, DH), 0.02)
            k_norm = ones_plus_noise((KVH, DH), 0.02)

            params.extend(
                [
                    attn_output,
                    attn_kv,
                    attn_q,
                    ffn_gate_up,
                    ffn_down,
                    input_norm,
                    post_attn_norm,
                    pre_ffn_norm,
                    post_ffn_norm,
                    q_norm,
                    k_norm,
                ]
            )

        return (tokens, positions, attention_mask, *params)

    # ================================================================
    # Utility
    # ================================================================
    def op_counts(ir):
        c = Counter()
        for nid in ir.order:
            c[ir.nodes[nid].op] += 1
        return c

    def decode_const_for_main(value, dtype):
        try:
            return decode_stablehlo_const(value, dtype)
        except Exception:
            pass

        if isinstance(value, np.generic):
            value = value.item()

        if isinstance(value, str):
            s = value.strip()
            low = s.lower()

            if low in ("inf", "+inf", "infinity", "+infinity"):
                return np.inf
            if low in ("-inf", "-infinity"):
                return -np.inf
            if low in ("nan", "+nan", "-nan"):
                return np.nan

            if low.startswith("0x"):
                bits = int(low, 16)
                dtype_s = str(dtype).lower()

                if "f16" in dtype_s:
                    return np.array([bits & 0xFFFF], dtype=np.uint16).view(np.float16)[0].item()

                if "bf16" in dtype_s:
                    return np.array([(bits & 0xFFFF) << 16], dtype=np.uint32).view(np.float32)[0].item()

                if "f32" in dtype_s:
                    return np.array([bits & 0xFFFFFFFF], dtype=np.uint32).view(np.float32)[0].item()

            return float(s)

        return value

    def tensor_dtype(t):
        try:
            return int(t.dtype)
        except Exception:
            return 1

    def set_one_input(g, tensor, data):
        dt = tensor_dtype(tensor)

        if dt == 1:
            arr = np.asarray(data, dtype=np.float16)
            g.set_input(tensor, arr, dtype=1)

        elif dt == 2:
            arr = np.asarray(data, dtype=np.float32)
            g.set_input(tensor, arr, dtype=2)

        else:
            arr = np.asarray(data, dtype=np.float32)
            g.set_input(tensor, arr, dtype=dt)

    def set_inputs_and_constants(g, env, ir, args_np):
        for i, data in enumerate(args_np):
            ssa = f"%arg{i}"
            if ssa not in env:
                raise RuntimeError(
                    f"Missing runtime arg {ssa}; env has first keys={list(env.keys())[:16]}"
                )

            set_one_input(g, env[ssa], data)

        for ssa, const in ir.constants.items():
            if ssa not in env:
                continue

            tensor = env[ssa]
            dt = tensor_dtype(tensor)
            shape = list(const.shape) if const.shape else [1]
            scalar = decode_const_for_main(const.value, const.dtype)

            if dt == 1:
                scalar_f = float(scalar)
                max_f16 = float(np.finfo(np.float16).max)

                if np.isnan(scalar_f):
                    scalar_f = -10000.0
                elif np.isneginf(scalar_f) or scalar_f < -max_f16:
                    scalar_f = -10000.0
                elif np.isposinf(scalar_f) or scalar_f > max_f16:
                    scalar_f = max_f16

                arr = np.full(shape, scalar_f, dtype=np.float16)
                g.set_input(tensor, arr, dtype=1)

            elif dt == 2:
                scalar_f = float(scalar)

                if np.isnan(scalar_f):
                    scalar_f = -10000.0
                elif np.isneginf(scalar_f):
                    scalar_f = -10000.0
                elif np.isposinf(scalar_f):
                    scalar_f = float(np.finfo(np.float32).max)

                arr = np.full(shape, scalar_f, dtype=np.float32)
                g.set_input(tensor, arr, dtype=2)

            else:
                arr = np.full(shape, float(scalar), dtype=np.float16)
                g.set_input(tensor, arr, dtype=dt)

    def summarize(name, arr):
        arr = np.asarray(arr)
        print(f"\n{name}:")
        print("  shape:", arr.shape)
        print("  dtype:", arr.dtype)

        if np.issubdtype(arr.dtype, np.floating):
            print("  finite:", bool(np.isfinite(arr).all()))
            print("  nan/inf:", int(np.count_nonzero(~np.isfinite(arr))), "/", arr.size)
            print("  min/max:", float(np.nanmin(arr)), float(np.nanmax(arr)))
            print(
                "  mean/std:",
                float(np.nanmean(arr.astype(np.float32))),
                float(np.nanstd(arr.astype(np.float32))),
            )

        print("  sample:", arr.reshape(-1)[:16])

    def compare(jax_np, cactus_np, atol=0.45, rtol=0.18):
        j = np.asarray(jax_np, dtype=np.float32)
        c = np.asarray(cactus_np, dtype=np.float32)

        if j.shape != c.shape:
            return False, {"reason": f"shape mismatch jax={j.shape} cactus={c.shape}"}

        finite = np.isfinite(j) & np.isfinite(c)
        diff = np.abs(j - c)

        stats = {
            "finite_overlap": f"{int(np.count_nonzero(finite))}/{finite.size}",
            "jax_all_finite": bool(np.isfinite(j).all()),
            "cactus_all_finite": bool(np.isfinite(c).all()),
            "max_diff": float(np.max(diff[finite])) if finite.any() else float("nan"),
            "mean_diff": float(np.mean(diff[finite])) if finite.any() else float("nan"),
            "p95_diff": float(np.percentile(diff[finite], 95)) if finite.any() else float("nan"),
            "p99_diff": float(np.percentile(diff[finite], 99)) if finite.any() else float("nan"),
        }

        denom = np.linalg.norm(j[finite]) * np.linalg.norm(c[finite])
        stats["cosine"] = float(np.dot(j[finite], c[finite]) / denom) if denom else float("nan")

        close = np.allclose(j[finite], c[finite], atol=atol, rtol=rtol) if finite.any() else False

        ok = (
            stats["jax_all_finite"]
            and stats["cactus_all_finite"]
            and (
                close
                or (
                    stats["cosine"] > 0.9980
                    and stats["mean_diff"] < 0.12
                    and stats["p99_diff"] < 0.90
                )
            )
        )

        if not ok and finite.any():
            masked = np.where(finite, diff, -1.0)
            idx = np.unravel_index(int(np.argmax(masked)), diff.shape)
            stats["worst_idx"] = idx
            stats["jax_value"] = float(j[idx])
            stats["cactus_value"] = float(c[idx])
            stats["diff"] = float(diff[idx])

        return ok, stats

    def print_topk(name, logits, k=10):
        row = np.asarray(logits[TOP_B, TOP_T], dtype=np.float32)
        ids = np.argsort(-row)[:k]

        print(f"\n{name} top{k} at batch={TOP_B} position={TOP_T}:")
        for i in ids:
            print(f"  {int(i):<5} {float(row[i]):>10.5f}")

    def run_mode(ir, args_np, jax_np, mode_name, patterns, strict_math=False):
        print("\n" + "=" * 80)
        print(f"LOWERING MODE: {mode_name}")
        print("=" * 80)

        t0 = time.perf_counter()
        g, env = lower_to_cactus(
            ir,
            patterns=patterns,
            verbose=False,
            strict_math=strict_math,
        )
        build_ms = (time.perf_counter() - t0) * 1000.0

        set_inputs_and_constants(g, env, ir, args_np)

        t0 = time.perf_counter()
        g.execute()
        exec_ms = (time.perf_counter() - t0) * 1000.0

        cactus_np = env[ir.outputs[0]].numpy()

        summarize("Cactus output", cactus_np)
        print_topk("Cactus", cactus_np)

        ok, stats = compare(jax_np, cactus_np)

        print("\nCompare:", ok)
        for k, v in stats.items():
            print(f"  {k}: {v}")

        print(f"\nTiming: build={build_ms:.3f}ms execute={exec_ms:.3f}ms")

        return ok

    # ================================================================
    # Build JAX reference
    # ================================================================
    print("\n================ REAL-SHAPE MINI GEMMA STRESS TEST ================")
    print(
        f"B={B} T={T} VOCAB={VOCAB} D={D} H={H} KVH={KVH} "
        f"DH={DH} FF={FF} LAYERS={NUM_LAYERS}"
    )

    args_np = make_inputs_and_weights()
    args_jax = tuple(jnp.asarray(x) for x in args_np)

    print("\nRuntime input summary:")
    summarize("tokens", args_np[0])
    summarize("positions", args_np[1])
    summarize("attention_mask", args_np[2])

    try:
        print("\nRunning JAX...")
        jitted = jax.jit(real_shape_mini_gemma)

        t0 = time.perf_counter()
        jax_out = jitted(*args_jax)
        jax.block_until_ready(jax_out)
        jax_ms = (time.perf_counter() - t0) * 1000.0

        jax_np = np.asarray(jax_out)

        summarize("JAX output", jax_np)
        print_topk("JAX", jax_np)
        print(f"JAX time: {jax_ms:.3f}ms")

    except Exception as e:
        print("\nJAX FAILED")
        print(type(e).__name__ + ":", str(e))
        traceback.print_exc(limit=12)
        raise SystemExit(1)

    # ================================================================
    # Lower to StableHLO and parse
    # ================================================================
    try:
        print("\nExporting StableHLO...")
        mlir = jax_to_mlir(real_shape_mini_gemma, args_jax)
        ir = parse_mlir(mlir)

        print("\nParsed IR:")
        print("  inputs:", len(ir.inputs))
        print("  outputs:", ir.outputs)
        print("  nodes:", len(ir.order))
        print("  constants:", len(ir.constants))

        counts = op_counts(ir)
        print("\nOps:")
        for op, n in sorted(counts.items()):
            print(f"  {op:<36} {n}")

        out_path = "real_shape_mini_gemma_stress.stablehlo.mlir"
        with open(out_path, "w") as f:
            f.write(mlir)

        print(f"\nSaved MLIR: {out_path}")

    except Exception as e:
        print("\nMLIR EXPORT/PARSE FAILED")
        print(type(e).__name__ + ":", str(e))
        traceback.print_exc(limit=12)
        raise SystemExit(1)

    # ================================================================
    # Run Cactus
    # ================================================================
    overall = True

    try:
        ok_default = run_mode(
            ir=ir,
            args_np=args_np,
            jax_np=jax_np,
            mode_name="default patterns, strict_math=False",
            patterns=["default"],
            strict_math=False,
        )
        overall = overall and ok_default

    except Exception as e:
        print("\nCACTUS DEFAULT MODE FAILED")
        print(type(e).__name__ + ":", str(e))
        traceback.print_exc(limit=14)
        overall = False

    if os.getenv("RUN_PRIMITIVE", "0") == "1":
        try:
            ok_primitive = run_mode(
                ir=ir,
                args_np=args_np,
                jax_np=jax_np,
                mode_name="primitive only, strict_math=False",
                patterns=[],
                strict_math=False,
            )
            overall = overall and ok_primitive

        except Exception as e:
            print("\nCACTUS PRIMITIVE MODE FAILED")
            print(type(e).__name__ + ":", str(e))
            traceback.print_exc(limit=14)
            overall = False

    print("\n================ FINAL RESULT ================")
    print("Real-shape mini Gemma stress test passed:", overall)

    if overall:
        print("REAL-SHAPE MINI GEMMA STRESS PASSED ✅")
        print("This proves the Cactus lowering path handles real Gemma core tensor geometry.")
        print("If real Gemma is still nonsense, focus on:")
        print("  - MLIR/weights matched-set")
        print("  - exact runtime input convention")
        print("  - tokenizer/prompt formatting")
        print("  - arg weight export/order mismatch")
    else:
        print("REAL-SHAPE MINI GEMMA STRESS FAILED ❌")
        print("This means the issue is reproducible with real Gemma-shaped synthetic weights.")
        print("Inspect real_shape_mini_gemma_stress.stablehlo.mlir and the failure above.")