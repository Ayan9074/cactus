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
def parse_stablehlo_ops(text: str):
    nodes = []

    in_main = False

    for line in text.splitlines():
        line = line.strip()

        # -------------------------
        # Enter main function
        # -------------------------
        if "func.func" in line and "@main" in line:
            in_main = True
            continue

        # -------------------------
        # Exit main
        # -------------------------
        if in_main and line == "}":
            break

        if not in_main:
            continue

        # skip empty
        if not line:
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

    print(f"Parsed {len(nodes)} nodes")

    if len(nodes) == 0:
        raise Exception("❌ Parser failed: no nodes extracted")

    return nodes
