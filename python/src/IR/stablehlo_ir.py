from dataclasses import dataclass
from typing import Optional
import re
import numpy as np


@dataclass
class IRNode:
    name: str
    op: str
    inputs: list[str]
    shape: Optional[tuple]
    raw: str


@dataclass
class IRFunction:
    name: str
    args: list[str]
    nodes: list[IRNode]
    returns: list[str]


# -------------------------
# Shape parsing
# -------------------------
def parse_tensor_shape(shape_str):
    dims = []
    for p in shape_str.split("x"):
        if p.startswith("f") or p.startswith("i"):
            break
        try:
            dims.append(int(p))
        except:
            pass
    return tuple(dims)


def extract_shape(line):
    # prefer output shape
    m = re.search(r"->\s*tensor<([^>]+)>", line)
    if m:
        return parse_tensor_shape(m.group(1))

    # fallback
    matches = re.findall(r"tensor<([^>]+)>", line)
    if matches:
        return parse_tensor_shape(matches[-1])

    return None


# -------------------------
# Constant extraction (safe)
# -------------------------
def extract_constant(line):
    m = re.search(r"dense<([^>]+)>", line)
    if not m:
        return None

    s = m.group(1).strip()

    # ⚠️ Skip huge hex blobs (model weights)
    if s.startswith('"0x') or len(s) > 100:
        return None

    try:
        if s.startswith("0x"):
            val = int(s, 16)

            # Interpret hex literals as IEEE bit-patterns for floating tensors.
            if "tensor<f16>" in line:
                arr = np.array([val], dtype=np.uint16).view(np.float16)
                return float(arr[0])
            if "tensor<bf16>" in line:
                # bf16 -> float32 by placing bits in the upper 16 of f32.
                bits = (np.array([val], dtype=np.uint16).astype(np.uint32) << 16)
                return float(bits.view(np.float32)[0])
            if "tensor<f32>" in line:
                arr = np.array([val], dtype=np.uint32).view(np.float32)
                return float(arr[0])
            if "tensor<f64>" in line:
                arr = np.array([val], dtype=np.uint64).view(np.float64)
                return float(arr[0])

            return float(val)

        return float(s)

    except:
        return None


# -------------------------
# MAIN PARSER
# -------------------------
def _parse_function(func_name: str, sig_line: str, func_lines):
    arg_names = re.findall(r"%arg\d+", sig_line)
    nodes = []
    returns = []
    for line in func_lines:
        line = line.strip()

        # skip empty
        if not line:
            continue

        if line.startswith("return "):
            returns = re.findall(r"%[\w\d_]+", line.split(":", 1)[0])
            continue

        # -------------------------
        # ONLY lines that define ops
        # -------------------------
        if "=" not in line:
            continue

        if "stablehlo." not in line and "chlo." not in line and "call" not in line:
            continue

        # -------------------------
        # HANDLE REDUCE
        # -------------------------
        m_reduce = re.match(
            r"(%[\w\d_]+)\s*=\s*stablehlo\.reduce\((.*?)\)\s+applies\s+stablehlo\.([\w_]+)\s+across\s+dimensions\s*=\s*\[([0-9,\s]*)\]",
            line,
        )
        if m_reduce:
            name = m_reduce.group(1)
            inside = m_reduce.group(2)
            reduce_kind = m_reduce.group(3)
            axes_raw = m_reduce.group(4)

            inputs = re.findall(r"%[\w\d_]+", inside)
            axes = [int(x.strip()) for x in axes_raw.split(",") if x.strip()]

            nodes.append(IRNode(
                name=name,
                op=f"reduce_{reduce_kind}",
                inputs=inputs,
                shape=extract_shape(line),
                raw=line[:500],  # truncate giant lines
            ))
            continue

        # -------------------------
        # NORMAL OPS
        # -------------------------
        m = re.match(
            r"(%[\w\d_]+)\s*=\s*(?:stablehlo\.([\w_]+)|chlo\.([\w_]+)|call\s+@([\w_]+))\s*(.*)",
            line
        )
        if not m:
            # Quoted custom-form ops, e.g.:
            # %8 = "stablehlo.gather"(%arg3, %7) <...> : (...) -> tensor<...>
            m = re.match(
                r"(%[\w\d_]+)\s*=\s*\"stablehlo\.([\w_]+)\"\s*\((.*)\)\s*(.*)",
                line
            )
            if m:
                name = m.group(1)
                op = m.group(2)
                inside = m.group(3)
                inputs = re.findall(r"%[\w\d_]+", inside)
                shape = extract_shape(line)
                nodes.append(IRNode(
                    name=name,
                    op=op,
                    inputs=inputs,
                    shape=shape,
                    raw=line[:500],
                ))
                continue
        if not m:
            continue

        name = m.group(1)

        if m.group(2):
            op = m.group(2)  # stablehlo op
        elif m.group(3):
            op = m.group(3)  # chlo op
        elif m.group(4):
            op = m.group(4)  # call target
        else:
            continue

        rest = m.group(5)

        inputs = re.findall(r"%[\w\d_]+", rest)
        shape = extract_shape(line)

        nodes.append(IRNode(
            name=name,
            op=op,
            inputs=inputs,
            shape=shape,
            raw=line[:500],  # 🔥 truncate giant constants
        ))

    return IRFunction(name=func_name, args=arg_names, nodes=nodes, returns=returns)


def parse_stablehlo_ops(text: str):
    # Parse all function bodies first.
    funcs = {}
    cur_name = None
    cur_sig = None
    cur_lines = []
    depth = 0

    for raw in text.splitlines():
        line = raw.strip()
        # function start
        m_func = re.match(r"func\.func\s+(?:public|private)?\s*@([\w\d_]+)\(", line)
        if m_func:
            cur_name = m_func.group(1)
            cur_sig = line
            cur_lines = []
            depth = line.count("{") - line.count("}")
            continue

        if cur_name is None:
            continue

        depth += line.count("{") - line.count("}")
        # Keep only body lines (skip closing brace).
        if line != "}":
            cur_lines.append(line)

        if depth <= 0:
            funcs[cur_name] = _parse_function(cur_name, cur_sig or "", cur_lines)
            cur_name = None
            cur_sig = None
            cur_lines = []
            depth = 0

    if "main" not in funcs:
        raise Exception("❌ Parser failed: no @main function found")

    def _inline_func(func_name: str, arg_bindings: dict[str, str], prefix: str, stack: set[str]) -> tuple[list[IRNode], list[str]]:
        if func_name in stack:
            raise Exception(f"Recursive call cycle detected at @{func_name}")
        stack = set(stack)
        stack.add(func_name)
        f = funcs[func_name]
        out_nodes: list[IRNode] = []
        local_map: dict[str, str] = {}

        def map_name(n: str) -> str:
            if n in arg_bindings:
                return arg_bindings[n]
            if n in local_map:
                return local_map[n]
            return n

        for node in f.nodes:
            # call to known local function -> inline recursively
            if node.op in funcs and node.op != func_name:
                callee = funcs[node.op]
                callee_inputs = [map_name(x) for x in node.inputs]
                call_prefix = f"{prefix}{node.name.strip('%')}_"
                callee_bind = {}
                for i, an in enumerate(callee.args):
                    if i < len(callee_inputs):
                        callee_bind[an] = callee_inputs[i]
                sub_nodes, sub_returns = _inline_func(node.op, callee_bind, call_prefix, stack)
                out_nodes.extend(sub_nodes)
                if sub_returns:
                    local_map[node.name] = sub_returns[0]
                    # Preserve call-site SSA name (e.g. %0 from @main return)
                    # so downstream output selection can still reference it.
                    out_nodes.append(IRNode(
                        name=node.name,
                        op="alias",
                        inputs=[sub_returns[0]],
                        shape=node.shape,
                        raw=f"{node.name} = alias {sub_returns[0]}",
                    ))
                continue

            new_name = f"%{prefix}{node.name.strip('%')}"
            local_map[node.name] = new_name
            new_inputs = [map_name(x) for x in node.inputs]
            out_nodes.append(IRNode(
                name=new_name,
                op=node.op,
                inputs=new_inputs,
                shape=node.shape,
                raw=node.raw,
            ))

        return_names = [map_name(r) for r in f.returns]
        return out_nodes, return_names

    main_func = funcs["main"]
    main_bind = {arg: arg for arg in main_func.args}
    nodes, _ = _inline_func("main", main_bind, "main_", set())

    print(f"Parsed {len(nodes)} nodes")

    if len(nodes) == 0:
        raise Exception("❌ Parser failed: no nodes extracted")

    return nodes
