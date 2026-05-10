from __future__ import annotations

import argparse
import importlib.util
import json
import struct
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from bundle import write_bundle_manifest
from compatibility import analyze_ir
from lower_to_cactus import decode_stablehlo_const, lower_to_cactus
from parse import parse_mlir


PROFILE_TO_PATTERNS = {
    "generic": ["default"],
    "transformer": ["default"],
    "vision": ["default"],
}


def choose_patterns(profile: str) -> list[str]:
    return PROFILE_TO_PATTERNS.get(profile, ["default"])


def _load_py_fn(py_file: Path, fn_name: str):
    spec = importlib.util.spec_from_file_location("user_model_module", py_file)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import python file: {py_file}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, fn_name):
        raise AttributeError(f"{py_file} does not define function {fn_name!r}")
    return getattr(mod, fn_name)


def _load_args_npz(npz_path: Path) -> tuple[np.ndarray, ...]:
    z = np.load(npz_path, allow_pickle=False)
    keys = sorted(z.files)
    if not keys:
        raise ValueError("inputs npz is empty")

    # Preferred key format: arg0, arg1, ..., argN
    if all(k.startswith("arg") and k[3:].isdigit() for k in keys):
        keys = sorted(keys, key=lambda k: int(k[3:]))

    return tuple(np.asarray(z[k]) for k in keys)


def _tensor_dtype(t) -> int:
    try:
        return int(t.dtype)
    except Exception:
        return 1


def _set_tensor_input(graph, tensor, data: np.ndarray) -> None:
    dt = _tensor_dtype(tensor)
    if dt == 1:
        graph.set_input(tensor, np.asarray(data, dtype=np.float16), dtype=1)
    elif dt == 2:
        graph.set_input(tensor, np.asarray(data, dtype=np.float32), dtype=2)
    elif dt == 0:
        graph.set_input(tensor, np.asarray(data, dtype=np.int8), dtype=0)
    elif dt == 3:
        graph.set_input(tensor, np.asarray(data, dtype=np.uint8), dtype=3)
    else:
        graph.set_input(tensor, np.asarray(data, dtype=np.float16), dtype=dt)


CACTUS_MAGIC = 0x54434143
CACTUS_PREC_FP16 = 1
CACTUS_HEADER_SIZE = 84
CACTUS_ALIGNMENT = 32


def _align_offset(offset: int, alignment: int) -> int:
    rem = offset % alignment
    return offset if rem == 0 else offset + (alignment - rem)


def _write_cactus_fp16_weight(path: Path, arr: np.ndarray) -> None:
    arr = np.ascontiguousarray(np.asarray(arr, dtype=np.float16))
    shape = list(arr.shape)
    if len(shape) > 4:
        raise ValueError(f"mmap writer supports <=4D tensors, got {shape} for {path}")
    ndim = len(shape)
    raw_shape = shape + [0] * (4 - ndim)

    header = bytearray(CACTUS_HEADER_SIZE)
    off = 0
    struct.pack_into("<I", header, off, CACTUS_MAGIC)
    off += 4
    struct.pack_into("<I", header, off, 0)  # flags
    off += 4
    struct.pack_into("<I", header, off, CACTUS_ALIGNMENT)
    off += 4
    struct.pack_into("<I", header, off, ndim)
    off += 4
    for dim in raw_shape:
        struct.pack_into("<Q", header, off, int(dim))
        off += 8
    struct.pack_into("<I", header, off, CACTUS_PREC_FP16)
    off += 4
    struct.pack_into("<Q", header, off, int(arr.size * 2))  # byte_size
    off += 8
    struct.pack_into("<Q", header, off, 0)  # scales_bytes
    off += 8
    struct.pack_into("<I", header, off, 0)  # group_size
    off += 4
    struct.pack_into("<I", header, off, 0)  # num_groups
    off += 4
    struct.pack_into("<Q", header, off, int(shape[0] if shape else 1))  # original_N
    off += 8
    if off != CACTUS_HEADER_SIZE:
        raise RuntimeError(f"bad header size: {off}")

    data_offset = _align_offset(CACTUS_HEADER_SIZE, CACTUS_ALIGNMENT)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(header)
        f.write(b"\x00" * (data_offset - CACTUS_HEADER_SIZE))
        f.write(arr.tobytes())


def _export_mmap_arg_weights(weights: tuple[np.ndarray, ...], start_arg: int, out_dir: Path) -> dict[int, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    mapping: dict[int, Path] = {}
    manifest: list[dict[str, Any]] = []
    for i, w in enumerate(weights):
        arg_idx = int(start_arg + i)
        path = out_dir / f"arg{arg_idx}.weights"
        _write_cactus_fp16_weight(path, np.asarray(w))
        mapping[arg_idx] = path
        manifest.append(
            {
                "arg_index": arg_idx,
                "file": path.name,
                "shape": list(np.asarray(w).shape),
                "dtype": "fp16",
            }
        )
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return mapping


def _decode_const_array(const) -> np.ndarray | None:
    """Decode non-scalar StableHLO constants when possible.

    Supports dense hex blobs like dense<"0x..."> for common dtypes.
    """
    if not const.shape:
        return None

    shape = tuple(int(x) for x in const.shape)
    n = int(np.prod(shape))
    v = const.value

    if isinstance(v, np.ndarray):
        try:
            return np.asarray(v).reshape(shape)
        except Exception:
            return None

    if isinstance(v, (list, tuple)):
        try:
            return np.asarray(v).reshape(shape)
        except Exception:
            return None

    if isinstance(v, str):
        s = v.strip()
        if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
            s = s[1:-1].strip()
        low = s.lower()
        if low.startswith("0x"):
            h = low[2:]
            if len(h) % 2 != 0:
                h = "0" + h
            raw = bytes.fromhex(h)
            dtype_s = str(const.dtype).lower()
            try:
                if "f16" in dtype_s and "bf16" not in dtype_s:
                    arr = np.frombuffer(raw, dtype="<f2", count=n)
                elif "bf16" in dtype_s:
                    u16 = np.frombuffer(raw, dtype="<u2", count=n).astype(np.uint32)
                    arr = (u16 << 16).view(np.float32)
                elif "f32" in dtype_s:
                    arr = np.frombuffer(raw, dtype="<f4", count=n)
                elif "f64" in dtype_s:
                    arr = np.frombuffer(raw, dtype="<f8", count=n)
                elif "i8" in dtype_s:
                    arr = np.frombuffer(raw, dtype=np.int8, count=n)
                elif "i16" in dtype_s:
                    arr = np.frombuffer(raw, dtype="<i2", count=n)
                elif "i32" in dtype_s:
                    arr = np.frombuffer(raw, dtype="<i4", count=n)
                elif "i64" in dtype_s:
                    arr = np.frombuffer(raw, dtype="<i8", count=n)
                elif "ui8" in dtype_s or "u8" in dtype_s:
                    arr = np.frombuffer(raw, dtype=np.uint8, count=n)
                elif "ui16" in dtype_s or "u16" in dtype_s:
                    arr = np.frombuffer(raw, dtype="<u2", count=n)
                elif "ui32" in dtype_s or "u32" in dtype_s:
                    arr = np.frombuffer(raw, dtype="<u4", count=n)
                elif "ui64" in dtype_s or "u64" in dtype_s:
                    arr = np.frombuffer(raw, dtype="<u8", count=n)
                elif "i1" in dtype_s:
                    arr = np.frombuffer(raw, dtype=np.uint8, count=n).astype(bool)
                else:
                    return None
                return np.asarray(arr).reshape(shape)
            except Exception:
                return None
    return None


def _set_constants(graph, env, ir) -> None:
    mask_fill_fp16 = -65504.0
    mask_fill_fp32 = -65504.0
    for ssa, const in ir.constants.items():
        tensor = env.get(ssa)
        if tensor is None:
            continue
        arr_full = _decode_const_array(const)
        if arr_full is not None:
            _set_tensor_input(graph, tensor, arr_full)
            continue
        dt = _tensor_dtype(tensor)
        shape = tuple(const.shape) if const.shape else (1,)
        scalar = decode_stablehlo_const(const.value, const.dtype)
        scalar_f = float(scalar)

        if dt == 1:
            max_f16 = float(np.finfo(np.float16).max)
            if np.isnan(scalar_f) or np.isneginf(scalar_f) or scalar_f < -max_f16:
                scalar_f = mask_fill_fp16
            elif np.isposinf(scalar_f) or scalar_f > max_f16:
                scalar_f = max_f16
            arr = np.full(shape, scalar_f, dtype=np.float16)
        elif dt == 2:
            if np.isnan(scalar_f) or np.isneginf(scalar_f):
                scalar_f = mask_fill_fp32
            elif np.isposinf(scalar_f):
                scalar_f = float(np.finfo(np.float32).max)
            arr = np.full(shape, scalar_f, dtype=np.float32)
        elif dt == 0:
            arr = np.full(shape, int(scalar_f), dtype=np.int8)
        elif dt == 3:
            arr = np.full(shape, int(scalar_f), dtype=np.uint8)
        else:
            arr = np.full(shape, scalar_f, dtype=np.float16)

        graph.set_input(tensor, arr, dtype=dt)


def _execute_graph(
    graph,
    env,
    ir,
    arg_values: dict[int, np.ndarray],
    *,
    set_constants: bool = True,
) -> list[np.ndarray]:
    for arg_idx, arr in sorted(arg_values.items(), key=lambda kv: kv[0]):
        ssa = f"%arg{arg_idx}"
        if ssa not in env:
            raise RuntimeError(f"Missing runtime arg {ssa}; available keys start={list(env.keys())[:16]}")
        _set_tensor_input(graph, env[ssa], arr)
    if set_constants:
        _set_constants(graph, env, ir)
    graph.execute()
    outs: list[np.ndarray] = []
    for o in ir.outputs:
        v = env[o]
        if hasattr(v, "numpy"):
            outs.append(v.numpy())
            continue
        # Diagnostic path for lazy BroadcastView outputs used by probe mode.
        if hasattr(v, "tensor") and hasattr(v, "logical_shape"):
            base = v.tensor.numpy()
            try:
                outs.append(np.broadcast_to(base, tuple(int(x) for x in v.logical_shape)).copy())
            except Exception:
                outs.append(np.asarray(base))
            continue
        raise TypeError(f"Unsupported output value type for {o}: {type(v)}")
    return outs


def _compare_arrays(ref: np.ndarray, got: np.ndarray, atol=1e-3, rtol=1e-3) -> dict[str, Any]:
    r = np.asarray(ref, dtype=np.float32)
    g = np.asarray(got, dtype=np.float32)
    if r.shape != g.shape:
        return {"ok": False, "reason": f"shape mismatch ref={r.shape} got={g.shape}"}
    finite = np.isfinite(r) & np.isfinite(g)
    if not finite.any():
        return {
            "ok": False,
            "reason": "no finite overlap",
            "ref_finite_ratio": float(np.isfinite(r).mean()),
            "got_finite_ratio": float(np.isfinite(g).mean()),
        }
    diff = np.abs(r - g)
    ok = bool(np.allclose(r[finite], g[finite], atol=atol, rtol=rtol))
    rf = r[finite].reshape(-1)
    gf = g[finite].reshape(-1)
    denom = float(np.linalg.norm(rf) * np.linalg.norm(gf))
    cosine = float(np.dot(rf, gf) / denom) if denom > 0.0 else float("nan")
    ref_flat = r.reshape(-1)
    got_flat = g.reshape(-1)
    return {
        "ok": ok,
        "atol": atol,
        "rtol": rtol,
        "shape": list(r.shape),
        "cosine": cosine,
        "max_diff": float(np.max(diff[finite])),
        "mean_diff": float(np.mean(diff[finite])),
        "p99_diff": float(np.percentile(diff[finite], 99)),
        "expected_sample": ref_flat[:16].tolist(),
        "actual_sample": got_flat[:16].tolist(),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Convert StableHLO MLIR or a Python JAX function to Cactus graph, "
            "run it with provided inputs, and optionally test against JAX output."
        )
    )
    ap.add_argument("mlir", nargs="?", default="", help="Path to StableHLO MLIR file")
    ap.add_argument("--python-file", default="", help="Python file containing JAX function")
    ap.add_argument("--function", default="", help="Function name inside --python-file")
    ap.add_argument(
        "--gemma3-hf-model-id",
        default="",
        help="Optional HF id (e.g. google/gemma-3-270m) to run built-in Gemma3 path in convert_any.",
    )
    ap.add_argument(
        "--gemma3-tiny",
        action="store_true",
        help="Use a tiny random Gemma3 config for fast smoke tests (no HF download/weights).",
    )
    ap.add_argument(
        "--gemma3-tiny-layers",
        type=int,
        default=4,
        help="Number of layers for --gemma3-tiny (default: 4).",
    )
    ap.add_argument(
        "--gemma3-hf-dir",
        default="",
        help="Optional local HF dir; defaults to ~/.jaxgarden/hf_models/<model-id> for Gemma3 mode.",
    )
    ap.add_argument(
        "--gemma3-export-max-pos",
        type=int,
        default=0,
        help="Optional export-time override for Gemma3 max_position_embeddings to shrink StableHLO constants.",
    )
    ap.add_argument(
        "--gemma3-hf-max-layers",
        type=int,
        default=0,
        help="Optional override for Gemma3 HF config num_hidden_layers (for bisection/debug).",
    )
    ap.add_argument(
        "--llama-hf-model-id",
        default="",
        help="Optional HF id (e.g. TinyLlama/TinyLlama-1.1B-Chat-v1.0) to run built-in Llama path in convert_any.",
    )
    ap.add_argument(
        "--llama-hf-dir",
        default="",
        help="Optional local HF dir; defaults to ~/.jaxgarden/hf_models/<model-id> for Llama mode.",
    )
    ap.add_argument(
        "--llama-hf-max-layers",
        type=int,
        default=0,
        help="Optional override for Llama HF config num_hidden_layers (for bisection/debug).",
    )
    ap.add_argument(
        "--prompt",
        default="The capital of France is",
        help="Prompt text for built-in Gemma3 mode.",
    )
    ap.add_argument("--inputs-npz", default="", help="NPZ containing runtime inputs arg0,arg1,...")
    ap.add_argument("--weights-npz", default="", help="Optional NPZ of extra args appended after inputs")
    ap.add_argument(
        "--mmap-weights-dir",
        default="",
        help="Optional directory with argN.weights files + manifest.json for mmap weight binding at lower-time.",
    )
    ap.add_argument(
        "--mmap-export-dir",
        default="",
        help="Optional output dir to export model weight args to Cactus .weights and mmap them automatically.",
    )
    ap.add_argument(
        "--mmap-arg-start",
        type=int,
        default=0,
        help="First arg index to mmap from --mmap-weights-dir (default: 0).",
    )
    ap.add_argument(
        "--mmap-arg-end",
        type=int,
        default=-1,
        help="Exclusive arg index to mmap from --mmap-weights-dir; -1 means all indices present in manifest/files.",
    )
    ap.add_argument(
        "--mmap-runtime-tail-count",
        type=int,
        default=-1,
        help="When using --mmap-export-dir, number of trailing runtime args to keep as set_input (default: auto by mode).",
    )
    ap.add_argument("--profile", default="auto", help="auto|generic|transformer|vision")
    ap.add_argument("--allow-tier-c", action="store_true", help="Allow lowering even if compatibility tier is C.")
    ap.add_argument("--no-patterns", action="store_true", help="Disable pattern fusions and use only primitive op lowering.")
    ap.add_argument("--graph-out", default="converted.cactus", help="Output graph file path")
    ap.add_argument("--report-json", default="", help="Optional output path for compatibility JSON")
    ap.add_argument("--save-outputs-npz", default="", help="Optional NPZ path to save graph outputs as out0,out1,...")
    ap.add_argument("--bundle-dir", default="", help="Optional output dir for bundle manifest")
    ap.add_argument("--run", action="store_true", help="Execute cactus graph if inputs are provided")
    ap.add_argument("--test", action="store_true", help="Run JAX reference and compare with Cactus output")
    ap.add_argument("--atol", type=float, default=1e-3, help="allclose absolute tolerance for --test")
    ap.add_argument("--rtol", type=float, default=1e-3, help="allclose relative tolerance for --test")
    ap.add_argument("--verbose", action="store_true", help="Verbose lowering logs")
    ap.add_argument("--bench-runs", type=int, default=1, help="Number of repeated graph executes for run timing (>=1)")
    ap.add_argument(
        "--with-debug-locs",
        action="store_true",
        help="Emit StableHLO with debug location metadata (larger/slower to parse).",
    )
    ap.add_argument(
        "--stop-after",
        choices=["mlir", "parse", "lower", "run"],
        default="run",
        help="Stop pipeline after a stage.",
    )
    ap.add_argument(
        "--probe-output-node",
        type=int,
        default=-1,
        help="Debug: override graph output to the first output of IR node at this execution-order index.",
    )
    args = ap.parse_args()
    t_start = time.perf_counter()
    timings: dict[str, float] = {}

    if (
        not args.mlir
        and not args.python_file
        and not args.gemma3_hf_model_id
        and not args.gemma3_tiny
        and not args.llama_hf_model_id
    ):
        raise SystemExit(
            "Provide one of: <mlir>, --python-file + --function, --gemma3-hf-model-id, --gemma3-tiny, or --llama-hf-model-id"
        )

    args_np: tuple[np.ndarray, ...] = ()
    runtime_tail_count = 0
    full_arg_values: dict[int, np.ndarray] = {}
    jax_ref_out: list[np.ndarray] | None = None
    mmap_arg_to_path: dict[int, Path] = {}

    if args.inputs_npz:
        args_np = _load_args_npz(Path(args.inputs_npz))
    if args.weights_npz:
        args_np = args_np + _load_args_npz(Path(args.weights_npz))

    if args.gemma3_hf_model_id or args.gemma3_tiny:
        import jax
        import jax.numpy as jnp
        from flax import nnx
        from jaxgarden import Gemma3Config, Gemma3ForCausalLM, Tokenizer
        from run_gemma3_hf_smoke import config_from_hf_dir
        from test import jax_to_mlir

        if args.gemma3_tiny:
            model_id = "gemma3-tiny-random"
            hf_dir = None
            cfg = Gemma3Config(
                vocab_size=4096,
                hidden_size=256,
                intermediate_size=1024,
                num_hidden_layers=max(1, int(args.gemma3_tiny_layers)),
                num_attention_heads=4,
                num_key_value_heads=1,
                head_dim=64,
                sliding_window=128,
                sliding_window_pattern=2,
                # Gemma3 internals can use a non-zero base position offset
                # during decode-style flows; keep table comfortably larger.
                max_position_embeddings=8192,
                dtype=jnp.float16,
                param_dtype=jnp.float16,
            )
            model = Gemma3ForCausalLM(cfg, rngs=nnx.Rngs(0))
            # Simple tiny prompt tokens for smoke path.
            input_ids = np.asarray([[2, 11, 277, 256, 1, 283]], dtype=np.int32)
            attention_mask = np.ones_like(input_ids, dtype=np.bool_)
        else:
            model_id = str(args.gemma3_hf_model_id)
            if args.gemma3_hf_dir:
                hf_dir = Path(args.gemma3_hf_dir).expanduser()
            else:
                hf_dir = Path.home() / ".jaxgarden" / "hf_models" / Path(model_id)
            if not hf_dir.exists():
                raise SystemExit(f"Gemma3 local HF dir not found: {hf_dir}")

            cfg = config_from_hf_dir(hf_dir)
            if int(args.gemma3_export_max_pos) > 0:
                cfg.max_position_embeddings = int(args.gemma3_export_max_pos)
            cfg.dtype = jnp.float16
            cfg.param_dtype = jnp.float16
            model = Gemma3ForCausalLM(cfg, rngs=nnx.Rngs(0))
            model.from_hf(
                model_id,
                force_download=False,
                save_in_orbax=False,
                remove_hf_after_conversion=False,
            )
            if int(args.gemma3_hf_max_layers) > 0:
                keep = max(1, min(int(args.gemma3_hf_max_layers), len(model.layers)))
                model.layers = list(model.layers[:keep])
                model.config.num_hidden_layers = keep

            tok = Tokenizer.from_pretrained(model_id)
            enc = tok.encode(args.prompt)
            input_ids = np.asarray(enc["input_ids"], dtype=np.int32)
            attention_mask = np.asarray(enc["attention_mask"], dtype=np.bool_)

        # Export with model state as function inputs to avoid giant embedded hex constants.
        graphdef, state, rest = nnx.split(model, nnx.Param, ...)
        state_leaves, state_treedef = jax.tree_util.tree_flatten(state)
        n_state = len(state_leaves)

        def fn_flat(*flat_args):
            st = jax.tree_util.tree_unflatten(state_treedef, flat_args[:n_state])
            input_ids = flat_args[n_state]
            attention_mask = flat_args[n_state + 1]
            m = nnx.merge(graphdef, st, rest)
            out = m(
                input_ids=input_ids,
                attention_mask=attention_mask,
                deterministic=True,
            )
            if isinstance(out, tuple):
                return out[0]
            return out

        def _runtime_arr(x):
            a = np.asarray(x)
            if np.issubdtype(a.dtype, np.floating):
                return a.astype(np.float16)
            return a

        args_np = tuple(_runtime_arr(x) for x in state_leaves) + (input_ids, attention_mask)
        runtime_tail_count = 2
        args_jax = tuple(jnp.asarray(x) for x in args_np)
        fn = fn_flat
        t0 = time.perf_counter()
        mlir_text = jax_to_mlir(fn, args_jax, with_debug_locs=bool(args.with_debug_locs))
        timings["mlir_export_sec"] = time.perf_counter() - t0
        mlir_path = Path(args.mlir) if args.mlir else Path("out.mlir")
        mlir_path.write_text(mlir_text)

        if args.test:
            t1 = time.perf_counter()
            out = fn(*args_jax)
            if isinstance(out, tuple):
                jax_ref_out = [np.asarray(jax.device_get(x)) for x in out]
            else:
                jax_ref_out = [np.asarray(jax.device_get(out))]
            timings["jax_ref_sec"] = time.perf_counter() - t1

        # Helpful metadata for this mode.
        print(
            json.dumps(
                {
                    "gemma3_mode": {
                        "model_id": model_id,
                        "hf_dir": str(hf_dir) if hf_dir is not None else "",
                        "prompt": args.prompt,
                        "input_shape": list(input_ids.shape),
                        "num_state_tensors": int(n_state),
                        "tiny": bool(args.gemma3_tiny),
                    }
                },
                indent=2,
                sort_keys=True,
            )
        )
    elif args.llama_hf_model_id:
        import jax
        import jax.numpy as jnp
        from flax import nnx
        from jaxgarden import LlamaConfig, LlamaForCausalLM
        from test import jax_to_mlir
        from transformers import AutoConfig, AutoTokenizer

        model_id = str(args.llama_hf_model_id)
        if args.llama_hf_dir:
            hf_dir = Path(args.llama_hf_dir).expanduser()
        else:
            hf_dir = Path.home() / ".jaxgarden" / "hf_models" / Path(model_id)
        if not hf_dir.exists():
            raise SystemExit(f"Llama local HF dir not found: {hf_dir}")

        hf_cfg = AutoConfig.from_pretrained(model_id)
        hidden = int(hf_cfg.hidden_size)
        heads = int(hf_cfg.num_attention_heads)
        head_dim = int(getattr(hf_cfg, "head_dim", hidden // heads))
        cfg = LlamaConfig(
            dim=hidden,
            n_layers=int(hf_cfg.num_hidden_layers),
            n_heads=heads,
            n_kv_heads=int(getattr(hf_cfg, "num_key_value_heads", heads)),
            head_dim=head_dim,
            intermediate_size=int(hf_cfg.intermediate_size),
            vocab_size=int(hf_cfg.vocab_size),
            norm_eps=float(getattr(hf_cfg, "rms_norm_eps", 1e-5)),
            rope_theta=float(getattr(hf_cfg, "rope_theta", 10000.0)),
        )
        cfg.dtype = jnp.float16
        cfg.param_dtype = jnp.float16

        model = LlamaForCausalLM(cfg, rngs=nnx.Rngs(0))
        model.from_hf(
            model_id,
            force_download=False,
            save_in_orbax=False,
            remove_hf_after_conversion=False,
        )
        if int(args.llama_hf_max_layers) > 0:
            keep = max(1, min(int(args.llama_hf_max_layers), len(model.layers)))
            model.layers = list(model.layers[:keep])
            model.config.n_layers = keep

        tok = AutoTokenizer.from_pretrained(model_id)
        enc = tok(args.prompt, return_tensors="np", add_special_tokens=True)
        input_ids = np.asarray(enc["input_ids"], dtype=np.int32)
        attention_mask = np.asarray(enc["attention_mask"], dtype=np.bool_)

        graphdef, state, rest = nnx.split(model, nnx.Param, ...)
        state_leaves, state_treedef = jax.tree_util.tree_flatten(state)
        n_state = len(state_leaves)

        def fn_flat(*flat_args):
            st = jax.tree_util.tree_unflatten(state_treedef, flat_args[:n_state])
            input_ids = flat_args[n_state]
            attention_mask = flat_args[n_state + 1]
            m = nnx.merge(graphdef, st, rest)
            out = m(
                input_ids=input_ids,
                attention_mask=attention_mask,
                deterministic=True,
            )
            if isinstance(out, tuple):
                return out[0]
            return out

        def _runtime_arr(x):
            a = np.asarray(x)
            if np.issubdtype(a.dtype, np.floating):
                return a.astype(np.float16)
            return a

        args_np = tuple(_runtime_arr(x) for x in state_leaves) + (input_ids, attention_mask)
        runtime_tail_count = 2
        args_jax = tuple(jnp.asarray(x) for x in args_np)
        fn = fn_flat
        t0 = time.perf_counter()
        mlir_text = jax_to_mlir(fn, args_jax, with_debug_locs=bool(args.with_debug_locs))
        timings["mlir_export_sec"] = time.perf_counter() - t0
        mlir_path = Path(args.mlir) if args.mlir else Path("out.mlir")
        mlir_path.write_text(mlir_text)

        if args.test:
            t1 = time.perf_counter()
            out = fn(*args_jax)
            jax_ref_out = [np.asarray(jax.device_get(out))]
            timings["jax_ref_sec"] = time.perf_counter() - t1

        print(
            json.dumps(
                {
                    "llama_mode": {
                        "model_id": model_id,
                        "hf_dir": str(hf_dir),
                        "prompt": args.prompt,
                        "input_shape": list(input_ids.shape),
                        "num_state_tensors": int(n_state),
                    }
                },
                indent=2,
                sort_keys=True,
            )
        )

    elif args.python_file:
        if not args.function:
            raise SystemExit("--function is required when --python-file is used")
        if not args_np:
            raise SystemExit("--inputs-npz is required for python mode")

        fn = _load_py_fn(Path(args.python_file), args.function)
        import jax
        import jax.numpy as jnp
        from test import jax_to_mlir

        args_jax = tuple(jnp.asarray(x) for x in args_np)
        t0 = time.perf_counter()
        mlir_text = jax_to_mlir(fn, args_jax, with_debug_locs=bool(args.with_debug_locs))
        timings["mlir_export_sec"] = time.perf_counter() - t0
        mlir_path = Path(args.mlir) if args.mlir else Path("out.mlir")
        mlir_path.write_text(mlir_text)

        if args.test:
            t1 = time.perf_counter()
            out = fn(*args_jax)
            if isinstance(out, tuple):
                jax_ref_out = [np.asarray(jax.device_get(x)) for x in out]
            else:
                jax_ref_out = [np.asarray(jax.device_get(out))]
            timings["jax_ref_sec"] = time.perf_counter() - t1
    full_arg_values = {i: arr for i, arr in enumerate(args_np)}

    if args.mmap_export_dir:
        export_dir = Path(args.mmap_export_dir)
        tail = int(args.mmap_runtime_tail_count)
        if tail < 0:
            tail = int(runtime_tail_count)
        if tail < 0 or tail > len(args_np):
            raise SystemExit(f"Invalid --mmap-runtime-tail-count={tail} for {len(args_np)} args")
        weight_args = args_np[: len(args_np) - tail]
        mmap_arg_to_path = _export_mmap_arg_weights(weight_args, start_arg=0, out_dir=export_dir)
        full_arg_values = {i: arr for i, arr in enumerate(args_np)}

    if args.mmap_weights_dir:
        wdir = Path(args.mmap_weights_dir)
        manifest_path = wdir / "manifest.json"
        if manifest_path.exists():
            raw = json.loads(manifest_path.read_text())
            for item in raw:
                idx = int(item["arg_index"])
                mmap_arg_to_path[idx] = wdir / str(item["file"])
        else:
            for p in wdir.glob("arg*.weights"):
                stem = p.stem
                if stem.startswith("arg") and stem[3:].isdigit():
                    mmap_arg_to_path[int(stem[3:])] = p

    if mmap_arg_to_path:
        start = int(args.mmap_arg_start)
        end = int(args.mmap_arg_end)
        if end >= 0:
            mmap_arg_to_path = {k: v for k, v in mmap_arg_to_path.items() if start <= k < end}
        else:
            mmap_arg_to_path = {k: v for k, v in mmap_arg_to_path.items() if k >= start}

    # If no path produced MLIR text yet, read explicit MLIR file mode.
    if "mlir_text" not in locals():
        mlir_path = Path(args.mlir)
        mlir_text = mlir_path.read_text()

    if args.stop_after == "mlir":
        timings["total_sec"] = time.perf_counter() - t_start
        payload = {
            "stage": "mlir",
            "mlir_path": str(mlir_path),
            "mlir_chars": len(mlir_text),
            "timings": {k: float(v) for k, v in timings.items()},
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    t0 = time.perf_counter()
    ir = parse_mlir(mlir_text)
    timings["parse_sec"] = time.perf_counter() - t0
    t1 = time.perf_counter()
    report = analyze_ir(ir)
    timings["analyze_sec"] = time.perf_counter() - t1

    if args.profile != "auto":
        report.profile = args.profile

    if int(args.probe_output_node) >= 0:
        idx = int(args.probe_output_node)
        if idx < 0 or idx >= len(ir.order):
            raise SystemExit(f"--probe-output-node {idx} out of range [0, {len(ir.order)-1}]")
        nid = ir.order[idx]
        n = ir.nodes[nid]
        if not n.outputs:
            raise SystemExit(f"node at index {idx} has no outputs: {nid} {n.op}")
        ir.outputs = [n.outputs[0]]

    if report.tier.startswith("C") and not args.allow_tier_c:
        print("Compatibility tier is C-blocked; refusing to lower.")
        print(json.dumps(asdict(report), indent=2, sort_keys=True))
        raise SystemExit(2)

    if args.stop_after == "parse":
        timings["total_sec"] = time.perf_counter() - t_start
        payload = asdict(report)
        payload["stage"] = "parse"
        payload["mlir_path"] = str(mlir_path)
        payload["timings"] = {k: float(v) for k, v in timings.items()}
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    patterns = [] if args.no_patterns else choose_patterns(report.profile)
    t2 = time.perf_counter()
    def _resolver(graph, ssa, shape, dtype):
        if not ssa.startswith("%arg"):
            return None
        sid = ssa[4:]
        if not sid.isdigit():
            return None
        idx = int(sid)
        p = mmap_arg_to_path.get(idx)
        if p is None:
            return None
        t = graph.mmap_weights(str(p))
        got = tuple(int(x) for x in getattr(t, "shape", ()))
        want = tuple(int(x) for x in shape)
        if got and want and got != want:
            raise ValueError(f"mmap shape mismatch for {ssa}: got={got} want={want} file={p}")
        return t

    g, _env = lower_to_cactus(
        ir,
        patterns=patterns,
        verbose=args.verbose,
        input_resolver=_resolver if mmap_arg_to_path else None,
    )
    timings["lower_sec"] = time.perf_counter() - t2

    graph_out = Path(args.graph_out)
    t3 = time.perf_counter()
    g.save(str(graph_out))
    timings["save_sec"] = time.perf_counter() - t3

    payload = asdict(report)
    payload["graph_out"] = str(graph_out)
    payload["patterns"] = patterns
    payload["stage"] = "lower"

    if args.stop_after == "lower":
        timings["total_sec"] = time.perf_counter() - t_start
        payload["timings"] = {k: float(v) for k, v in timings.items()}
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    runtime_arg_values = {i: v for i, v in full_arg_values.items() if i not in mmap_arg_to_path}

    should_run = args.stop_after == "run" and (args.run or args.test or bool(runtime_arg_values))
    if should_run:
        payload["stage"] = "run"
        if not runtime_arg_values:
            raise SystemExit("Need runtime inputs: provide --inputs-npz (and optional --weights-npz)")
        bench_runs = max(1, int(args.bench_runs))
        run_times: list[float] = []
        cactus_outs = None
        for i in range(bench_runs):
            tr = time.perf_counter()
            cactus_outs = _execute_graph(g, _env, ir, runtime_arg_values, set_constants=(i == 0))
            run_times.append(time.perf_counter() - tr)
        assert cactus_outs is not None
        timings["run_sec"] = float(run_times[-1])
        timings["run_sec_avg"] = float(sum(run_times) / len(run_times))
        payload["run"] = {
            "executed": True,
            "num_outputs": len(cactus_outs),
            "output_shapes": [list(np.asarray(o).shape) for o in cactus_outs],
            "bench_runs": bench_runs,
        }
        try:
            o0 = np.asarray(cactus_outs[0], dtype=np.float32)
            payload["run"]["output_finite_ratio"] = float(np.isfinite(o0).mean())
            payload["run"]["output_absmax"] = float(np.nanmax(np.abs(o0)))
        except Exception:
            pass
        if args.test:
            if jax_ref_out is None:
                raise SystemExit("--test requires python mode (--python-file + --function)")
            tc = time.perf_counter()
            cmp_items = []
            for oi, (r, c) in enumerate(zip(jax_ref_out, cactus_outs)):
                cmp_items.append(
                    {
                        "output_index": oi,
                        **_compare_arrays(r, c, atol=args.atol, rtol=args.rtol),
                    }
                )
            timings["compare_sec"] = time.perf_counter() - tc
            payload["test"] = {
                "ok": all(bool(x.get("ok", False)) for x in cmp_items),
                "outputs": cmp_items,
            }

        if args.save_outputs_npz:
            out_path = Path(args.save_outputs_npz)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(out_path, **{f"out{i}": np.asarray(o) for i, o in enumerate(cactus_outs)})
            payload.setdefault("run", {})
            payload["run"]["saved_outputs_npz"] = str(out_path)

    timings["total_sec"] = time.perf_counter() - t_start
    payload["timings"] = {k: float(v) for k, v in timings.items()}
    if mmap_arg_to_path:
        payload["mmap"] = {
            "enabled": True,
            "mapped_arg_count": len(mmap_arg_to_path),
            "runtime_arg_count": int(len(runtime_arg_values)),
        }

    print(json.dumps(payload, indent=2, sort_keys=True))

    if args.report_json:
        Path(args.report_json).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    if args.bundle_dir:
        bundle_path = write_bundle_manifest(
            Path(args.bundle_dir),
            source_mlir=mlir_path,
            graph_file=graph_out,
            profile=report.profile,
            compatibility_tier=report.tier,
            extra={"patterns": patterns},
        )
        print(f"bundle manifest: {bundle_path}")


if __name__ == "__main__":
    main()
