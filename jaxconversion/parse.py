"""
parse_mlir.py
=============
Parse a StableHLO MLIR text module into a clean IRGraph.

What comes out:
  IRGraph
    .inputs    : [ssa_name, ...]          - graph-level inputs in order
    .outputs   : [ssa_name, ...]          - graph-level outputs in order
    .constants : {ssa_name -> IRConstant} - extracted constants (weights/biases)
    .values    : {ssa_name -> IRValue}    - every SSA value with shape + dtype
    .nodes     : {node_id  -> IRNode}     - every op with structured attrs
    .order     : [node_id, ...]           - topological execution order

All call sites are inlined. No `call` ops remain in the output.
Constants are extracted and never become nodes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

SSA_RE = r"%[\w.]+(?:#\d+)?"
# ---------------------------------------------------------------------------
# IR dataclasses
# ---------------------------------------------------------------------------

@dataclass
class IRValue:
    name: str                    # SSA name e.g. "%0"
    shape: tuple[int, ...]       # e.g. (4, 16)
    dtype: str                   # e.g. "f32", "f16", "i8"
    producer: Optional[str]      # node_id that produces this, None for inputs/constants


@dataclass
class IRConstant:
    name: str                    # SSA name e.g. "%cst"
    shape: tuple[int, ...]
    dtype: str
    value: Any                   # parsed scalar/list, or raw string if unparseable


@dataclass
class IRNode:
    id: str                      # unique e.g. "node_0"
    op: str                      # e.g. "stablehlo.dot_general"
    inputs: list[str]            # SSA names consumed
    outputs: list[str]           # SSA names produced
    attrs: dict[str, Any]        # structured attributes
    input_types: list[str]       # MLIR type strings for inputs
    result_types: list[str]      # MLIR type strings for outputs


@dataclass
class IRGraph:
    inputs:    list[str]                   # SSA names of graph inputs
    outputs:   list[str]                   # SSA names of graph outputs
    constants: dict[str, IRConstant]       # extracted constants
    values:    dict[str, IRValue]          # all SSA values
    nodes:     dict[str, IRNode]           # all op nodes
    order:     list[str]                   # node ids in execution order


# ---------------------------------------------------------------------------
# Type parsing helpers
# ---------------------------------------------------------------------------

def _parse_lhs_outputs(lhs: str) -> list[str]:
    """
    Parse MLIR LHS outputs.

    Handles:
      %0 = ...
      %13:2 = ...      -> %13#0, %13#1
      %a, %b = ...
    """
    outs = []

    for m in re.finditer(r"(%[\w.]+)(?::(\d+))?", lhs):
        base = m.group(1)
        count = m.group(2)

        if count is not None:
            for i in range(int(count)):
                outs.append(f"{base}#{i}")
        else:
            outs.append(base)

    return outs

def _parse_mlir_type(type_str: str) -> tuple[tuple[int, ...], str]:
    """
    Parse an MLIR type string.
    "tensor<4x8xf32>"  ->  ((4, 8), "f32")
    "tensor<f32>"      ->  ((),     "f32")   scalar tensor
    "f32"              ->  ((),     "f32")   bare scalar
    """
    type_str = type_str.strip()
    m = re.match(r"tensor<(.+)>", type_str)
    if m:
        inner = m.group(1)
        parts = inner.split("x")
        dtype = parts[-1]
        try:
            shape = tuple(int(p) for p in parts[:-1])
        except ValueError:
            shape = ()
        return shape, dtype
    return (), type_str


def _split_types(type_block: str) -> list[str]:
    """Split comma-separated MLIR types, respecting < > depth."""
    type_block = type_block.strip().strip("()")
    types, current, depth = [], "", 0
    for ch in type_block:
        if ch == "<": depth += 1
        elif ch == ">": depth -= 1
        if ch == "," and depth == 0:
            if t := current.strip(): types.append(t)
            current = ""
        else:
            current += ch
    if t := current.strip(): types.append(t)
    return types


# ---------------------------------------------------------------------------
# Attribute parsing
# ---------------------------------------------------------------------------
def _parse_slice_ranges(raw: str) -> dict[str, Any]:
    """
    Parse StableHLO positional slice syntax.

    Examples:
      [0:3, 0:7, 0:32]
      [0:3:1, 0:7:1, 32:64:1]
    """
    raw = raw.strip()
    m = re.search(r"\[([^\]]+)\]", raw)
    if not m:
        return {}

    body = m.group(1).strip()
    if ":" not in body:
        return {}

    starts: list[int] = []
    limits: list[int] = []
    strides: list[int] = []

    for part in body.split(","):
        part = part.strip()
        mm = re.fullmatch(r"(-?\d+)\s*:\s*(-?\d+)(?:\s*:\s*(-?\d+))?", part)
        if not mm:
            return {}
        starts.append(int(mm.group(1)))
        limits.append(int(mm.group(2)))
        strides.append(int(mm.group(3)) if mm.group(3) is not None else 1)

    return {
        "start_indices": starts,
        "limit_indices": limits,
        "strides": strides,
    }

def _parse_attrs(raw: str) -> dict[str, Any]:
    attrs: dict[str, Any] = {}
    if not raw:
        return attrs

    # Positional StableHLO slice syntax:
    #   stablehlo.slice %x [0:3, 0:7, 32:64]
    slice_attrs = _parse_slice_ranges(raw)
    if slice_attrs:
        attrs.update(slice_attrs)

    # Drop surrounding attr braces if present:
    #   {dimension = 2 : i64}
    raw_clean = raw.strip()
    if raw_clean.startswith("{") and raw_clean.endswith("}"):
        raw_clean = raw_clean[1:-1].strip()

    applies_m = re.search(r"applies\s+([\w.]+)", raw_clean)
    if applies_m:
        attrs["applies"] = applies_m.group(1)

    # StableHLO compare often prints direction positionally:
    #   stablehlo.compare GT, %x, %y, FLOAT : ...
    cmp_m = re.search(r"\b(EQ|NE|LT|LE|GT|GE)\b", raw)
    if cmp_m:
        attrs["comparison_direction"] = cmp_m.group(1)
    
    dir_m = re.search(
        r"comparison_direction\s*=\s*#?stablehlo<comparison_direction\s+(\w+)>",
        raw,
    )
    if dir_m:
        attrs["comparison_direction"] = dir_m.group(1).upper()
    # Parse key=value attrs, while respecting nested brackets/braces.
    i = 0
    while i < len(raw_clean):
        m = re.search(r"(\w+)\s*=\s*", raw_clean[i:])
        if not m:
            break

        key = m.group(1)
        val_start = i + m.end()

        depth = 0
        j = val_start
        while j < len(raw_clean):
            ch = raw_clean[j]
            if ch in "<([{":
                depth += 1
            elif ch in ">)]}":
                depth -= 1
            elif ch == "," and depth == 0:
                break
            j += 1

        val_raw = raw_clean[val_start:j].strip().rstrip(",").strip()
        attrs[key] = _parse_attr_value(val_raw)

        i = j + 1

    return attrs


def _parse_attr_value(raw: str) -> Any:
    """Parse a single MLIR attribute value into a Python object."""
    raw = raw.strip().strip("{}").strip()

    # Strip MLIR type suffixes:
    #   2 : i64       -> 2
    #   true : i1     -> true
    typed_scalar = re.match(r"^(.+?)\s*:\s*[\w!<>., x]+$", raw)
    if typed_scalar and not raw.startswith("array<"):
        raw = typed_scalar.group(1).strip()

    # [1] x [0] -> ([1], [0])
    xprod = re.match(r"(\[[^\]]*\])\s*x\s*(\[[^\]]*\])", raw)
    if xprod:
        return (
            [int(x) for x in re.findall(r"-?\d+", xprod.group(1))],
            [int(x) for x in re.findall(r"-?\d+", xprod.group(2))],
        )

    # array<i64: 1, 2, 3>
    arr_m = re.match(r"array<[^:>]+:\s*([^>]*)>", raw)
    if arr_m:
        inner = arr_m.group(1).strip()
        if not inner:
            return []
        return [int(x) for x in re.findall(r"-?\d+", inner)]

    # [1, 2, 3] or [DEFAULT, DEFAULT]
    list_m = re.match(r"\[([^\]]*)\]", raw)
    if list_m:
        inner = list_m.group(1).strip()
        if not inner:
            return []
        parts = [p.strip() for p in inner.split(",")]
        try:
            return [int(p) for p in parts]
        except ValueError:
            return parts

    if raw in ("true", "false"):
        return raw == "true"

    try:
        return int(raw)
    except ValueError:
        pass

    try:
        return float(raw)
    except ValueError:
        pass

    return raw

def _parse_constant_value(attrs_raw: str) -> Any:
    """Extract and parse the value from a dense<...> attribute."""
    m = re.search(r"dense<([^>]*)>", attrs_raw)
    if not m:
        return None
    inner = m.group(1).strip()
    try:
        return float(inner)
    except ValueError:
        pass
    try:
        return [float(x) for x in inner.strip("[]").split(",")]
    except ValueError:
        return inner


# ---------------------------------------------------------------------------
# Low-level raw parsers  (private)
# ---------------------------------------------------------------------------

@dataclass
class _RawOp:
    opcode:       str
    inputs:       list[str]
    outputs:      list[str]
    attrs_raw:    str
    input_types:  list[str]
    result_types: list[str]


@dataclass
class _RawFunc:
    name:        str
    arg_names:   list[str]
    arg_types:   list[str]
    ops:         list[_RawOp]
    return_vals: list[str]


def _parse_func_signature(line: str) -> Optional[_RawFunc]:
    m = re.match(r"\s*func\.func(?:\s+\w+)?\s+@([\w.$]+)\s*\((.*)\)", line)
    if not m:
        return None
    name = m.group(1)
    arg_block = m.group(2).strip()
    arg_names, arg_types = [], []

    # Split args by top-level commas only (ignore commas inside tensor<> / loc()).
    parts: list[str] = []
    cur = []
    depth = 0
    for ch in arg_block:
        if ch in "<({[":
            depth += 1
        elif ch in ">)}]":
            depth -= 1
        if ch == "," and depth == 0:
            p = "".join(cur).strip()
            if p:
                parts.append(p)
            cur = []
        else:
            cur.append(ch)
    tail = "".join(cur).strip()
    if tail:
        parts.append(tail)

    for part in parts:
        mm = re.match(r"(%[\w.]+)\s*:\s*([^\s]+)", part)
        if not mm:
            continue
        arg_names.append(mm.group(1))
        arg_types.append(mm.group(2).strip())
    return _RawFunc(name=name, arg_names=arg_names, arg_types=arg_types, ops=[], return_vals=[])

"""
def _parse_raw_op(line: str) -> Optional[_RawOp]:
    line = line.strip()
    if not line or line.startswith("//") or "func.func" in line or line == "}":
        return None

    if re.match(r"return\b", line):
        return _RawOp("return", re.findall(r"%[\w.]+", line), [], "", [], [])

    outputs, rhs = [], line
    lhs_m = re.match(r"((?:%[\w.,\s]+)\s*)=\s*(.*)", line)
    if lhs_m:
        outputs = re.findall(r"%[\w.]+", lhs_m.group(1))
        rhs = lhs_m.group(2).strip()

    op_m = re.match(r'(?:"([^"]+)"|([\w.@]+))\s*(.*)', rhs)
    if not op_m:
        return None

    opcode = op_m.group(1) or op_m.group(2)
    rest = op_m.group(3).strip()

    # Fast-path for huge call sites (e.g. @main -> @__call__ with hundreds of args).
    # We only need SSA wiring for inlining; detailed type parsing here is unnecessary
    # and can be very slow on very large signatures.
    if opcode == "call":
        inputs = re.findall(r"%[\w.]+", rest)
        return _RawOp(opcode, inputs, outputs, rest, [], [])

    # split at top-level colon to separate operands from types
    colon_idx, depth = None, 0
    for i, ch in enumerate(rest):
        if ch in "<([{":
            depth += 1
        elif ch in ">)]}":
            depth -= 1
        elif ch == ":" and depth == 0:
            colon_idx = i
            break

    operands_part = rest[:colon_idx].strip() if colon_idx is not None else rest
    type_part     = rest[colon_idx+1:].strip() if colon_idx is not None else ""

    inputs    = re.findall(r"%[\w.]+", operands_part)
    attrs_raw = re.sub(r"%[\w.]+", "", operands_part).strip().strip(",").strip()

    input_types, result_types = [], []
    arrow_m = re.search(r"->\s*(.+)$", type_part)
    if arrow_m:
        result_types = _split_types(arrow_m.group(1))
        input_types  = _split_types(type_part[:arrow_m.start()])
    elif type_part:
        result_types = _split_types(type_part)

    return _RawOp(opcode, inputs, outputs, attrs_raw, input_types, result_types)
"""


def _parse_raw_op(line: str) -> Optional[_RawOp]:
    line = line.strip()
    if not line or line.startswith("//") or "func.func" in line or line == "}":
        return None

    if re.match(r"return\b", line):
        return _RawOp("return", re.findall(SSA_RE, line), [], "", [], [])

    outputs, rhs = [], line

    # Robust LHS parse.
    # Handles normal:
    #   %0 = stablehlo.add ...
    # and multi-result:
    #   %13:2 = call ...
    if line.startswith("%"):
        lhs_m = re.match(r"(.+?)\s*=\s*(.*)", line)
        if lhs_m:
            outputs = _parse_lhs_outputs(lhs_m.group(1))
            rhs = lhs_m.group(2).strip()

    # Support quoted StableHLO ops:
    #   "stablehlo.gather"(...)
    quoted_m = re.match(r'"([^"]+)"\s*(.*)', rhs)
    if quoted_m:
        opcode = quoted_m.group(1)
        rest = quoted_m.group(2).strip()
    else:
        op_m = re.match(r"([\w.@]+)\s*(.*)", rhs)
        if not op_m:
            return None
        opcode, rest = op_m.group(1), op_m.group(2).strip()

    # split at top-level colon to separate operands from types
    colon_idx, depth = None, 0
    for i, ch in enumerate(rest):
        if ch in "<([{":
            depth += 1
        elif ch in ">)]}":
            depth -= 1
        elif ch == ":" and depth == 0:
            colon_idx = i
            break

    operands_part = rest[:colon_idx].strip() if colon_idx is not None else rest
    type_part     = rest[colon_idx + 1:].strip() if colon_idx is not None else ""

    inputs    = re.findall(SSA_RE, operands_part)
    attrs_raw = re.sub(SSA_RE, "", operands_part).strip().strip(",").strip()

    input_types, result_types = [], []
    arrow_m = re.search(r"->\s*(.+)$", type_part)
    if arrow_m:
        result_types = _split_types(arrow_m.group(1))
        input_types  = _split_types(type_part[:arrow_m.start()])
    elif type_part:
        result_types = _split_types(type_part)

    return _RawOp(opcode, inputs, outputs, attrs_raw, input_types, result_types)

def _parse_raw_funcs(mlir_text: str) -> dict[str, _RawFunc]:
    """First pass: parse all functions into _RawFunc/_RawOp."""
    funcs: dict[str, _RawFunc] = {}
    current: Optional[_RawFunc] = None

    lines = mlir_text.splitlines()
    i = 0

    def _is_complete_stmt(stmt: str) -> bool:
        pairs = {"<": ">", "(": ")", "[": "]", "{": "}"}
        rev = {v: k for k, v in pairs.items()}
        stack: list[str] = []
        for ch in stmt:
            if ch in pairs:
                stack.append(ch)
            elif ch in rev:
                if stack and stack[-1] == rev[ch]:
                    stack.pop()
        return len(stack) == 0

    def _is_complete_func_sig(sig: str) -> bool:
        start = sig.find("(")
        if start < 0:
            return False
        depth = 0
        for ch in sig[start:]:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    return True
        return False

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if "func.func" in stripped:
            sig = stripped
            # Some MLIR exports wrap function signatures across many lines.
            # Stitch until the top-level argument list closes.
            while not _is_complete_func_sig(sig) and i + 1 < len(lines):
                i += 1
                sig += " " + lines[i].strip()
            f = _parse_func_signature(sig)
            if f:
                current = f
                funcs[f.name] = f
            i += 1
            continue
        if current is None:
            i += 1
            continue
        if stripped == "}":
            current = None
            i += 1
            continue

        # Some StableHLO ops span multiple lines (attrs/types wrapped).
        stmt = stripped
        j = i
        while not _is_complete_stmt(stmt) and j + 1 < len(lines):
            j += 1
            stmt += " " + lines[j].strip()
        i = j

        op = _parse_raw_op(stmt)
        if op is None:
            i += 1
            continue
        if op.opcode == "return":
            current.return_vals = op.inputs
        else:
            current.ops.append(op)
        i += 1

    return funcs


# ---------------------------------------------------------------------------
# Inlining pass  (private)
# ---------------------------------------------------------------------------

def _inline(funcs: dict[str, _RawFunc]) -> list[_RawOp]:
    """
    Inline all call ops starting from @main, returning a flat op list.
    SSA names in inlined callees are suffixed to avoid collisions.
    """
    main = funcs.get("main")
    if not main:
        raise ValueError("No @main function found in MLIR module.")

    counter = [0]

    def inline_one(callee: _RawFunc, caller_args: list[str]) -> tuple[list[_RawOp], list[str]]:
        counter[0] += 1
        suffix = f"_{callee.name}{counter[0]}"

        arg_map = dict(zip(callee.arg_names, caller_args))
        local = {o for op in callee.ops for o in op.outputs}
        rename = {n: n + suffix for n in local}

        def remap(ssa: str) -> str:
            return arg_map.get(ssa) or rename.get(ssa) or ssa

        ops = [
            _RawOp(
                opcode       = op.opcode,
                inputs       = [remap(s) for s in op.inputs],
                outputs      = [rename.get(s, s) for s in op.outputs],
                attrs_raw    = op.attrs_raw,
                input_types  = list(op.input_types),
                result_types = list(op.result_types),
            )
            for op in callee.ops
        ]
        return ops, [remap(r) for r in callee.return_vals]

    def resolve(ops: list[_RawOp]) -> list[_RawOp]:
        flat: list[_RawOp] = []
        for op in ops:
            if op.opcode != "call":
                flat.append(op)
                continue

            m = re.search(r"@(\w+)", op.attrs_raw)
            if not m:
                raise ValueError(f"Cannot parse callee from: {op.attrs_raw!r}")
            callee_name = m.group(1)
            if callee_name not in funcs:
                raise ValueError(f"Unknown function @{callee_name}")

            callee = funcs[callee_name]
            resolved_callee = _RawFunc(
                name=callee.name, arg_names=callee.arg_names,
                arg_types=callee.arg_types,
                ops=resolve(callee.ops), return_vals=callee.return_vals,
            )
            inlined_ops, renamed_returns = inline_one(resolved_callee, op.inputs)
            flat.extend(inlined_ops)

            # bind call-site outputs to callee return values
            for call_out, ret_val in zip(op.outputs, renamed_returns):
                flat.append(_RawOp("_alias", [ret_val], [call_out], "", op.result_types, op.result_types))

        return flat

    return resolve(main.ops)


# ---------------------------------------------------------------------------
# IRGraph builder
# ---------------------------------------------------------------------------

def _build_graph(main: _RawFunc, flat_ops: list[_RawOp]) -> IRGraph:
    """
    Convert the flat inlined op list into a clean IRGraph:
      - stablehlo.constant  ->  IRConstant  (removed from node graph)
      - _alias              ->  transparent (resolved, no node emitted)
      - everything else     ->  IRNode with parsed attrs
    """
    values:    dict[str, IRValue]    = {}
    constants: dict[str, IRConstant] = {}
    nodes:     dict[str, IRNode]     = {}
    order:     list[str]             = []
    alias:     dict[str, str]        = {}
    node_counter = [0]

    def canonical(ssa: str) -> str:
        while ssa in alias: ssa = alias[ssa]
        return ssa

    def new_id() -> str:
        nid = f"node_{node_counter[0]}"
        node_counter[0] += 1
        return nid

    # register graph-level inputs
    graph_inputs: list[str] = []
    for arg_name, arg_type in zip(main.arg_names, main.arg_types):
        shape, dtype = _parse_mlir_type(arg_type)
        values[arg_name] = IRValue(name=arg_name, shape=shape, dtype=dtype, producer=None)
        graph_inputs.append(arg_name)

    for op in flat_ops:

        # transparent alias
        if op.opcode == "_alias":
            alias[op.outputs[0]] = canonical(op.inputs[0])
            continue

        # constant -> extract, skip node
        if op.opcode == "stablehlo.constant":
            for out, rtype in zip(op.outputs, op.result_types):
                shape, dtype = _parse_mlir_type(rtype)
                val = _parse_constant_value(op.attrs_raw)
                constants[out] = IRConstant(name=out, shape=shape, dtype=dtype, value=val)
                values[out] = IRValue(name=out, shape=shape, dtype=dtype, producer=None)
            continue

        # normal op -> IRNode
        nid = new_id()
        resolved_inputs  = [canonical(s) for s in op.inputs]
        resolved_outputs = list(op.outputs)

        node = IRNode(
            id           = nid,
            op           = op.opcode,
            inputs       = resolved_inputs,
            outputs      = resolved_outputs,
            attrs        = _parse_attrs(op.attrs_raw),
            input_types  = op.input_types,
            result_types = op.result_types,
        )
        nodes[nid] = node
        order.append(nid)

        for out, rtype in zip(resolved_outputs, op.result_types):
            shape, dtype = _parse_mlir_type(rtype)
            values[out] = IRValue(name=out, shape=shape, dtype=dtype, producer=nid)

    graph_outputs = [canonical(r) for r in main.return_vals]

    return IRGraph(
        inputs    = graph_inputs,
        outputs   = graph_outputs,
        constants = constants,
        values    = values,
        nodes     = nodes,
        order     = order,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def parse_mlir(mlir_text: str) -> IRGraph:
    """
    Parse a StableHLO MLIR module and return a flat, clean IRGraph.

    Internally:
      1. Parse all func.func blocks
      2. Inline all call sites from @main
      3. Extract constants into IRGraph.constants
      4. Build IRNodes with structured attrs and typed IRValues
    """
    raw_funcs = _parse_raw_funcs(mlir_text)
    flat_ops  = _inline(raw_funcs)
    return _build_graph(raw_funcs["main"], flat_ops)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import jax
    import jax.numpy as jnp
    from test import jax_to_mlir

    def simple_model(x, w):
        h = x @ w
        return jax.nn.relu(h)

    x = jnp.ones((4, 8),  dtype=jnp.float32)
    w = jnp.ones((8, 16), dtype=jnp.float32)

    g = parse_mlir(jax_to_mlir(simple_model, (x, w)))
    print(g)

    print("=== inputs ===")
    for name in g.inputs:
        v = g.values[name]
        print(f"  {name}  shape={v.shape}  dtype={v.dtype}")

    print("\n=== constants ===")
    for name, c in g.constants.items():
        print(f"  {name}  shape={c.shape}  dtype={c.dtype}  value={c.value}")

    print("\n=== nodes ===")
    for nid in g.order:
        n = g.nodes[nid]
        print(f"  [{nid}]  {n.op}")
        print(f"    inputs  = {n.inputs}")
        print(f"    outputs = {n.outputs}")
        print(f"    attrs   = {n.attrs}")
        for out in n.outputs:
            v = g.values[out]
            print(f"    value {out}: shape={v.shape}  dtype={v.dtype}")

    print("\n=== outputs ===")
    for name in g.outputs:
        v = g.values[name]
        print(f"  {name}  shape={v.shape}  dtype={v.dtype}")
