from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from src.graph import Graph
from src.IR.stablehlo_ir import parse_stablehlo_ops
from src.IR.lower_to_cactus import lower_to_cactus
from src.tensor_io import save_tensor_with_header


def _parse_arg_count(stablehlo_text: str) -> int:
    m = re.search(r"func\.func[^\n]*@main\s*\(", stablehlo_text)
    if not m:
        raise RuntimeError("Could not find @main signature start in StableHLO")
    start = m.end() - 1  # '('
    depth = 0
    end = -1
    for i in range(start, len(stablehlo_text)):
        ch = stablehlo_text[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                end = i
                break
    if end < 0:
        raise RuntimeError("Could not parse @main argument list")
    sig = stablehlo_text[start + 1 : end]
    return len(re.findall(r"%arg\d+", sig))


def _parse_return_names(stablehlo_text: str) -> List[str]:
    # Handles: return %123 : ... OR return %1, %2 : ...
    for line in stablehlo_text.splitlines():
        line = line.strip()
        if not line.startswith("return "):
            continue
        before_type = line.split(":", 1)[0]
        names = re.findall(r"%[\w\d_]+", before_type)
        if names:
            return names
    raise RuntimeError("Could not parse return value names from @main")


def _load_sample_inputs(inputs_path: Path, arg_count: int) -> Dict[str, np.ndarray]:
    result: Dict[str, np.ndarray] = {}

    if inputs_path.suffix == ".npz":
        with np.load(inputs_path, allow_pickle=False) as z:
            for i in range(arg_count):
                key_candidates = [f"arg{i}", f"%arg{i}", str(i)]
                arr = None
                for k in key_candidates:
                    if k in z:
                        arr = z[k]
                        break
                if arr is None:
                    raise RuntimeError(
                        f"Missing arg{i} in NPZ. Expected one of {key_candidates}."
                    )
                result[f"%arg{i}"] = np.asarray(arr)
        return result

    if inputs_path.is_dir():
        for i in range(arg_count):
            p = inputs_path / f"arg{i}.npy"
            if not p.exists():
                raise RuntimeError(f"Missing input file: {p}")
            result[f"%arg{i}"] = np.load(p, allow_pickle=False)
        return result

    raise RuntimeError("--inputs must be .npz or a directory containing arg0.npy, arg1.npy, ...")


def _should_mmap_arg(arg_name: str, arr: np.ndarray, weight_policy: str, weight_arg_regex: str | None) -> bool:
    if weight_policy == "inputs":
        return False
    if arg_name == "%arg0":
        # Safety default: keep primary activation input runtime.
        # Regex can still force mmap in mixed mode if explicitly requested.
        if weight_policy == "mmap":
            return False
        if weight_arg_regex and re.search(weight_arg_regex, arg_name):
            return True
        return False
    if weight_policy == "mmap":
        return True
    # mixed
    if weight_arg_regex and re.search(weight_arg_regex, arg_name):
        return True
    # Heuristic fallback for mixed mode: matrices and higher-rank tensors are usually static weights.
    return arr.ndim >= 2


def compile_mlir_to_cgraph(
    stablehlo_path: Path,
    inputs_path: Path,
    output_cgraph_path: Path,
    output_meta_path: Path,
    output_name: str | None,
    weight_policy: str = "inputs",
    weight_arg_regex: str | None = None,
) -> None:
    text = stablehlo_path.read_text()
    nodes = parse_stablehlo_ops(text)

    arg_count = _parse_arg_count(text)
    return_names = _parse_return_names(text)
    sample_inputs = _load_sample_inputs(inputs_path, arg_count)

    g = Graph()
    observed_inputs: Dict[int, np.ndarray] = {}
    orig_set_input = g.set_input

    def _recording_set_input(tensor, data, dtype=None):
        arr = np.ascontiguousarray(data)
        observed_inputs[int(tensor.id)] = arr.copy()
        return orig_set_input(tensor, arr, dtype=dtype)

    g.set_input = _recording_set_input  # type: ignore[assignment]

    input_map = {}
    input_shapes: Dict[str, Tuple[int, ...]] = {}
    runtime_arg_names: List[str] = []
    runtime_input_tensors = {}
    mmap_weight_files: Dict[str, str] = {}
    weights_dir = Path(str(output_cgraph_path) + ".weights")
    if weight_policy != "inputs":
        weights_dir.mkdir(parents=True, exist_ok=True)

    for i in range(arg_count):
        name = f"%arg{i}"
        arr = np.ascontiguousarray(sample_inputs[name])
        # Path-1 policy: treat non-primary args as weights/static tensors and
        # keep them at FP16 for Cactus-friendly memory/perf behavior.
        if name != "%arg0" and np.issubdtype(arr.dtype, np.floating):
            arr = arr.astype(np.float16, copy=False)
        use_mmap = _should_mmap_arg(name, arr, weight_policy, weight_arg_regex)
        if use_mmap:
            wpath = weights_dir / f"{name.replace('%', 'arg_')}.weights"
            # Keep generic path simple and stable: FP16 mmap weights for now.
            save_tensor_with_header(arr.astype(np.float16, copy=False), str(wpath), precision="FP16")
            t = g.mmap_weights(str(wpath))
            mmap_weight_files[name] = str(wpath.name)
        else:
            t = g.input(arr.shape, Graph.FP16 if arr.dtype == np.float16 else Graph.FP32)
            g.set_input(t, arr)
            runtime_arg_names.append(name)
            runtime_input_tensors[name] = t
        input_map[name] = t
        input_shapes[name] = tuple(int(x) for x in arr.shape)

    env = lower_to_cactus(
        nodes,
        g,
        input_map,
        input_shapes,
        raw_inputs=[],
        enable_attention_fusion=True,
        enable_rmsnorm_fusion=True,
        arg_specs=None,
    )

    out_name = output_name or return_names[0]
    if out_name not in env:
        raise RuntimeError(
            f"Requested output {out_name} not found in lowered env. "
            f"Try one of: {return_names[:8]}"
        )

    g.save(str(output_cgraph_path))

    runtime_arg_ids = {int(t.id) for t in runtime_input_tensors.values()}
    literal_inputs_dir = Path(str(output_cgraph_path) + ".literal_inputs")
    literal_inputs_dir.mkdir(parents=True, exist_ok=True)
    literal_node_ids: List[int] = []
    for node_id, arr in sorted(observed_inputs.items()):
        if node_id in runtime_arg_ids:
            continue
        np.save(literal_inputs_dir / f"node_{node_id}.npy", arr)
        literal_node_ids.append(int(node_id))

    meta = {
        "stablehlo_path": str(stablehlo_path),
        "arg_count": arg_count,
        "arg_names": [f"%arg{i}" for i in range(arg_count)],
        "runtime_arg_names": runtime_arg_names,
        "all_input_shapes": {k: list(v) for k, v in input_shapes.items()},
        "input_node_ids": {k: int(v.id) for k, v in runtime_input_tensors.items()},
        "input_precisions": {k: int(v.dtype) for k, v in runtime_input_tensors.items()},
        "input_shapes": {k: list(input_shapes[k]) for k in runtime_arg_names},
        "weight_policy": weight_policy,
        "mmap_weights_dir": weights_dir.name if weight_policy != "inputs" else None,
        "mmap_weight_files": mmap_weight_files,
        "literal_inputs_dir": literal_inputs_dir.name,
        "literal_node_ids": literal_node_ids,
        "return_names": return_names,
        "selected_output": out_name,
        "selected_output_node_id": int(env[out_name].id),
        "notes": "Generic JAX StableHLO compile (best effort).",
    }
    output_meta_path.write_text(json.dumps(meta, indent=2))

    print(f"Saved cgraph: {output_cgraph_path}")
    print(f"Saved meta:   {output_meta_path}")
    print(f"Saved literal inputs: {literal_inputs_dir} ({len(literal_node_ids)} tensors)")
    print(f"Selected output: {out_name}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Compile JAX StableHLO MLIR to Cactus .cgraph")
    ap.add_argument("--stablehlo", required=True, help="Path to StableHLO .mlir file")
    ap.add_argument("--inputs", required=True, help="NPZ or directory with arg*.npy sample inputs")
    ap.add_argument("--out", required=True, help="Output .cgraph path")
    ap.add_argument("--meta", default=None, help="Output metadata JSON path")
    ap.add_argument(
        "--output-name",
        default=None,
        help="Return node name to select (default: first return value)",
    )
    ap.add_argument(
        "--weight-policy",
        default="inputs",
        choices=["inputs", "mmap", "mixed"],
        help="How to bind %argN tensors: all runtime inputs, all mmap weights, or mixed.",
    )
    ap.add_argument(
        "--weight-arg-regex",
        default=None,
        help="Regex over arg names (e.g. '%arg[1-9][0-9]*') to force mmap in mixed mode.",
    )
    args = ap.parse_args()

    stablehlo_path = Path(args.stablehlo).resolve()
    inputs_path = Path(args.inputs).resolve()
    out_path = Path(args.out).resolve()
    meta_path = Path(args.meta).resolve() if args.meta else Path(str(out_path) + ".meta.json")

    compile_mlir_to_cgraph(
        stablehlo_path=stablehlo_path,
        inputs_path=inputs_path,
        output_cgraph_path=out_path,
        output_meta_path=meta_path,
        output_name=args.output_name,
        weight_policy=args.weight_policy,
        weight_arg_regex=args.weight_arg_regex,
    )


if __name__ == "__main__":
    main()
