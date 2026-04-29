from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np

from src.graph import Graph
from src.IR.stablehlo_ir import parse_stablehlo_ops
from src.IR.lower_to_cactus import lower_to_cactus


def _load_runtime_inputs(inputs_path: Path, arg_names: list[str]) -> Dict[str, np.ndarray]:
    result: Dict[str, np.ndarray] = {}

    if inputs_path.suffix == ".npz":
        with np.load(inputs_path, allow_pickle=False) as z:
            for name in arg_names:
                i = int(name.replace("%arg", ""))
                key_candidates = [f"arg{i}", f"%arg{i}", str(i), name]
                arr = None
                for k in key_candidates:
                    if k in z:
                        arr = z[k]
                        break
                if arr is None:
                    raise RuntimeError(
                        f"Missing {name} in NPZ. Expected one of {key_candidates}."
                    )
                result[name] = np.asarray(arr)
        return result

    if inputs_path.is_dir():
        for name in arg_names:
            i = int(name.replace("%arg", ""))
            p = inputs_path / f"arg{i}.npy"
            if not p.exists():
                raise RuntimeError(f"Missing input file: {p}")
            result[name] = np.load(p, allow_pickle=False)
        return result

    raise RuntimeError("--inputs must be .npz or a directory containing arg0.npy, arg1.npy, ...")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run generic compiled .cgraph with runtime inputs")
    ap.add_argument("--cgraph", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--out", required=True, help="Output .npy path")
    args = ap.parse_args()

    cgraph_path = Path(args.cgraph).resolve()
    meta_path = Path(args.meta).resolve()
    inputs_path = Path(args.inputs).resolve()
    out_path = Path(args.out).resolve()

    meta = json.loads(meta_path.read_text())
    arg_count = int(meta["arg_count"])
    runtime_arg_names = list(meta.get("runtime_arg_names", [f"%arg{i}" for i in range(arg_count)]))
    input_shapes = meta["input_shapes"]
    all_input_shapes = meta.get("all_input_shapes", input_shapes)
    input_node_ids = {k: int(v) for k, v in meta["input_node_ids"].items()}
    input_precisions = {k: int(v) for k, v in meta["input_precisions"].items()}
    literal_inputs_dir = None
    if meta.get("literal_inputs_dir"):
        lit = Path(meta["literal_inputs_dir"])
        if lit.is_absolute():
            literal_inputs_dir = lit
        else:
            literal_inputs_dir = (cgraph_path.parent / lit).resolve()
    literal_node_ids = [int(x) for x in meta.get("literal_node_ids", [])]
    output_node_id = int(meta["selected_output_node_id"])
    mmap_weight_files = dict(meta.get("mmap_weight_files", {}))

    runtime_inputs = _load_runtime_inputs(inputs_path, runtime_arg_names)

    # Safe path for mmap-backed artifacts:
    # some current Cactus builds can segfault on Graph.load(...).execute() when mmap weight
    # nodes are serialized. Rebuild from StableHLO + mmap files instead of loading the .cgraph.
    if mmap_weight_files:
        stablehlo_path = Path(meta.get("stablehlo_path", ""))
        if not stablehlo_path.is_absolute():
            stablehlo_path = (cgraph_path.parent / stablehlo_path).resolve()
        if not stablehlo_path.exists():
            raise RuntimeError(
                f"Missing stablehlo source for mmap artifact rebuild: {stablehlo_path}"
            )
        stablehlo_text = stablehlo_path.read_text()
        nodes = parse_stablehlo_ops(stablehlo_text)

        g = Graph()
        input_map = {}
        input_shapes_map = {}

        weights_dir = meta.get("mmap_weights_dir")
        if not weights_dir:
            raise RuntimeError("Missing mmap_weights_dir in metadata")
        weights_dir_path = Path(weights_dir)
        if not weights_dir_path.is_absolute():
            weights_dir_path = (cgraph_path.parent / weights_dir_path).resolve()

        arg_names = list(meta.get("arg_names", []))
        if not arg_names:
            raise RuntimeError("Missing arg_names in metadata")

        for name in arg_names:
            if name in mmap_weight_files:
                wpath = weights_dir_path / mmap_weight_files[name]
                if not wpath.exists():
                    raise RuntimeError(f"Missing mmap weight file: {wpath}")
                t = g.mmap_weights(str(wpath))
                # use shape from metadata if present
                shape = tuple(int(x) for x in all_input_shapes.get(name, []))
                if shape:
                    input_shapes_map[name] = shape
            else:
                arr = np.ascontiguousarray(runtime_inputs[name])
                expected = tuple(int(x) for x in input_shapes[name])
                if tuple(arr.shape) != expected:
                    raise RuntimeError(
                        f"Shape mismatch for {name}: expected {expected}, got {tuple(arr.shape)}"
                    )
                expected_precision = input_precisions[name]
                if expected_precision == Graph.FP16:
                    arr = arr.astype(np.float16, copy=False)
                elif expected_precision == Graph.FP32:
                    arr = arr.astype(np.float32, copy=False)
                t = g.input(arr.shape, Graph.FP16 if arr.dtype == np.float16 else Graph.FP32)
                g.set_input(t, np.ascontiguousarray(arr))
                input_shapes_map[name] = tuple(int(x) for x in arr.shape)
            input_map[name] = t

        env = lower_to_cactus(
            nodes,
            g,
            input_map,
            input_shapes_map,
            raw_inputs=[],
            enable_attention_fusion=True,
            enable_rmsnorm_fusion=True,
            arg_specs=None,
        )
        out_name = meta.get("selected_output")
        if out_name not in env:
            raise RuntimeError(
                f"Selected output {out_name} not found after rebuild. "
                f"Available sample: {list(env.keys())[:8]}"
            )
        g.execute()
        y = env[out_name].numpy()
        np.save(out_path, y)
        print(f"Saved output (rebuilt mmap path): {out_path} shape={tuple(y.shape)} dtype={y.dtype}")
        return

    g = Graph.load(str(cgraph_path))

    if literal_inputs_dir is not None and literal_node_ids:
        for node_id in literal_node_ids:
            p = literal_inputs_dir / f"node_{node_id}.npy"
            if not p.exists():
                raise RuntimeError(f"Missing literal input tensor: {p}")
            arr = np.load(p, allow_pickle=False)
            t = g._tensor_from_node(node_id)
            g.set_input(t, np.ascontiguousarray(arr))

    for name in runtime_arg_names:
        arr = np.ascontiguousarray(runtime_inputs[name])
        expected = tuple(int(x) for x in input_shapes[name])
        if tuple(arr.shape) != expected:
            raise RuntimeError(
                f"Shape mismatch for {name}: expected {expected}, got {tuple(arr.shape)}"
            )
        expected_precision = input_precisions[name]
        if expected_precision == Graph.FP16:
            arr = arr.astype(np.float16, copy=False)
        elif expected_precision == Graph.FP32:
            arr = arr.astype(np.float32, copy=False)
        t = g._tensor_from_node(input_node_ids[name])  # Use compiled graph's existing input node.
        g.set_input(t, np.ascontiguousarray(arr))

    g.execute()

    y = g._tensor_from_node(output_node_id).numpy()
    np.save(out_path, y)
    print(f"Saved output: {out_path} shape={tuple(y.shape)} dtype={y.dtype}")


if __name__ == "__main__":
    main()
