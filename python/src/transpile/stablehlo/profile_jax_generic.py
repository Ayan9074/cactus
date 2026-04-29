from __future__ import annotations

import argparse
import importlib
import json
import re
import tempfile
import time
from pathlib import Path
from typing import Dict, List

import jax
import jax.numpy as jnp
import numpy as np

from src.graph import Graph
from src.IR.stablehlo_ir import parse_stablehlo_ops
from src.IR.lower_to_cactus import lower_to_cactus
from src.tensor_io import save_tensor_with_header


def _load_callable(spec: str):
    if ":" not in spec:
        raise RuntimeError("--callable must be module.submodule:function_name")
    mod_name, fn_name = spec.split(":", 1)
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, fn_name, None)
    if fn is None or not callable(fn):
        raise RuntimeError(f"Callable not found or not callable: {spec}")
    return fn


def _load_args_npz(npz_path: Path) -> List[np.ndarray]:
    with np.load(npz_path, allow_pickle=False) as z:
        keys = list(z.keys())
        if not keys:
            raise RuntimeError("Input NPZ is empty")

        def key_order(k: str):
            if k.startswith("arg") and k[3:].isdigit():
                return int(k[3:])
            if k.startswith("%arg") and k[4:].isdigit():
                return int(k[4:])
            if k.isdigit():
                return int(k)
            return 10**9

        sorted_keys = sorted(keys, key=key_order)
        arrays: List[np.ndarray] = [np.asarray(z[k]) for k in sorted_keys]
    return arrays


def _parse_return_names(stablehlo_text: str) -> List[str]:
    for line in stablehlo_text.splitlines():
        s = line.strip()
        if s.startswith("return "):
            names = re.findall(r"%[\w\d_]+", s.split(":", 1)[0])
            if names:
                return names
    raise RuntimeError("Could not parse return names")


def _should_mmap_arg(arg_name: str, arr: np.ndarray, weight_policy: str, weight_arg_regex: str | None) -> bool:
    if weight_policy == "inputs":
        return False
    if arg_name == "%arg0":
        if weight_policy == "mmap":
            return False
        if weight_arg_regex and re.search(weight_arg_regex, arg_name):
            return True
        return False
    if weight_policy == "mmap":
        return True
    if weight_arg_regex and re.search(weight_arg_regex, arg_name):
        return True
    return arr.ndim >= 2


def _metrics(pred: np.ndarray, ref: np.ndarray) -> Dict[str, float]:
    p = pred.astype(np.float32)
    r = ref.astype(np.float32)
    d = p - r
    mae = float(np.mean(np.abs(d)))
    max_abs = float(np.max(np.abs(d)))
    rmse = float(np.sqrt(np.mean(d * d)))
    cosine = float(np.dot(p.ravel(), r.ravel()) / (np.linalg.norm(p.ravel()) * np.linalg.norm(r.ravel()) + 1e-12))
    k = min(5, p.shape[-1] if p.ndim > 1 else p.size)
    if p.ndim == 1:
        a = set(np.argpartition(p, -k)[-k:].tolist())
        b = set(np.argpartition(r, -k)[-k:].tolist())
        overlap = len(a & b)
        top5_mean = float(overlap)
        top5_min = int(overlap)
    else:
        p2 = p.reshape(-1, p.shape[-1])
        r2 = r.reshape(-1, r.shape[-1])
        overlaps = []
        for i in range(p2.shape[0]):
            a = set(np.argpartition(p2[i], -k)[-k:].tolist())
            b = set(np.argpartition(r2[i], -k)[-k:].tolist())
            overlaps.append(len(a & b))
        top5_mean = float(np.mean(overlaps))
        top5_min = int(np.min(overlaps))
    return {
        "mae": mae,
        "max_abs": max_abs,
        "rmse": rmse,
        "cosine": cosine,
        "top5_overlap_mean": top5_mean,
        "top5_overlap_min": top5_min,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Profile JAX->StableHLO->Cactus generic path (in-process)")
    ap.add_argument("--callable", required=True)
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--weight-policy", default="inputs", choices=["inputs", "mmap", "mixed"])
    ap.add_argument("--weight-arg-regex", default=None)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--output-name", default=None)
    args = ap.parse_args()

    fn = _load_callable(args.callable)
    np_args = _load_args_npz(Path(args.inputs).resolve())

    # 1 stablehlo export
    jax_args = [jnp.asarray(a) for a in np_args]
    t0 = time.perf_counter()
    lowered = jax.jit(fn).lower(*jax_args)
    stablehlo_text = str(lowered.compiler_ir(dialect="stablehlo"))
    t_export_ms = (time.perf_counter() - t0) * 1000.0

    # 2 parse
    t1 = time.perf_counter()
    nodes = parse_stablehlo_ops(stablehlo_text)
    t_parse_ms = (time.perf_counter() - t1) * 1000.0

    returns = _parse_return_names(stablehlo_text)
    out_name = args.output_name or returns[0]

    # 3 build graph
    g = Graph()
    input_map = {}
    input_shapes = {}
    runtime_arg_tensors = {}
    mmap_arg_names = []

    weights_tmpdir = Path(tempfile.mkdtemp(prefix="cactus_generic_weights_"))

    set_input_ms = 0.0
    for i, arr in enumerate(np_args):
        name = f"%arg{i}"
        arr_c = np.ascontiguousarray(arr)
        if _should_mmap_arg(name, arr_c, args.weight_policy, args.weight_arg_regex):
            wpath = weights_tmpdir / f"arg_{i}.weights"
            save_tensor_with_header(arr_c.astype(np.float16, copy=False), str(wpath), precision="FP16")
            t = g.mmap_weights(str(wpath))
            mmap_arg_names.append(name)
        else:
            t = g.input(arr_c.shape, Graph.FP16 if arr_c.dtype == np.float16 else Graph.FP32)
            s0 = time.perf_counter()
            g.set_input(t, arr_c)
            set_input_ms += (time.perf_counter() - s0) * 1000.0
            runtime_arg_tensors[name] = t
        input_map[name] = t
        input_shapes[name] = tuple(int(x) for x in arr_c.shape)

    b0 = time.perf_counter()
    env = lower_to_cactus(nodes, g, input_map, input_shapes, raw_inputs=[])
    t_build_ms = (time.perf_counter() - b0) * 1000.0

    if out_name not in env:
        raise RuntimeError(f"Output {out_name} not found; candidates={returns[:8]}")
    out_t = env[out_name]

    # 4 cactus compile (first execute)
    c0 = time.perf_counter()
    g.execute()
    t_cactus_compile_ms = (time.perf_counter() - c0) * 1000.0

    # 6 steady execute
    for _ in range(10):
        g.execute()
    x0 = time.perf_counter()
    for _ in range(args.iters):
        g.execute()
    t_execute_ms = ((time.perf_counter() - x0) * 1000.0) / args.iters

    # 7 output read
    r0 = time.perf_counter()
    cactus_out = out_t.numpy()
    t_output_read_ms = (time.perf_counter() - r0) * 1000.0

    # JAX reference + timing
    jit_fn = jax.jit(fn)
    _ = jit_fn(*jax_args).block_until_ready()
    tj0 = time.perf_counter()
    jax_out = jit_fn(*jax_args).block_until_ready()
    t_jax_ms = (time.perf_counter() - tj0) * 1000.0
    jax_out = np.asarray(jax_out)

    m = _metrics(cactus_out, jax_out)

    result = {
        "weight_policy": args.weight_policy,
        "mmap_arg_names": mmap_arg_names,
        "runtime_arg_names": sorted(runtime_arg_tensors.keys()),
        "stablehlo_export_ms": t_export_ms,
        "stablehlo_parse_ms": t_parse_ms,
        "cgraph_build_ms": t_build_ms,
        "cactus_compile_ms_first_execute": t_cactus_compile_ms,
        "set_input_ms": set_input_ms,
        "execute_ms_avg": t_execute_ms,
        "output_read_ms": t_output_read_ms,
        "jax_ref_ms": t_jax_ms,
        "num_nodes": len(nodes),
        "output_shape": list(cactus_out.shape),
        **m,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
