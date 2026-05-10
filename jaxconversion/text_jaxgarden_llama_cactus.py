#!/usr/bin/env python3
"""
test_jaxgarden_llama_cactus.py
==============================

End-to-end test:

  JAXgarden Llama tiny random model
    -> JAX reference logits
    -> StableHLO export
    -> parse/lower to Cactus
    -> execute Cactus
    -> compare logits

This is meant to prove that a real JAXgarden model implementation can go through
your JAX -> StableHLO -> Cactus pipeline.

Run:
  python test_jaxgarden_llama_cactus.py --seq 8

Optional:
  python test_jaxgarden_llama_cactus.py --seq 8 --strict-math false
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

import jax
import jax.numpy as jnp
from flax import nnx


ROOT = Path(__file__).resolve().parent
JAXGARDEN_ROOT = ROOT / "third_party" / "jaxgarden"
if JAXGARDEN_ROOT.exists():
    sys.path.insert(0, str(JAXGARDEN_ROOT))

from jaxgarden  import LlamaConfig, LlamaForCausalLM

# Your local pipeline imports.
from test import jax_to_mlir
from parse import parse_mlir
from lower_to_cactus import lower_to_cactus
from transformers import AutoTokenizer
from transformers import AutoConfig


def print_decoded_outputs(tokenizer, input_ids, jax_logits, cactus_logits, prompt_len=None):
    if tokenizer is None:
        print("\nNo tokenizer available, skipping decoded text.")
        return

    ids_np = np.asarray(input_ids, dtype=np.int32)
    seq = ids_np.shape[1]

    if prompt_len is None:
        prompt_len = seq

    prompt_len = max(1, min(int(prompt_len), seq))
    eval_pos = prompt_len - 1

    display_prompt_ids = ids_np[0, :prompt_len].tolist()

    jax_next = int(np.argmax(np.asarray(jax_logits)[0, eval_pos]))
    cactus_next = int(np.argmax(np.asarray(cactus_logits)[0, eval_pos]))

    print("\n================ DECODED OUTPUT ================")
    print("eval position:", eval_pos)
    print("prompt token ids:", display_prompt_ids)
    print("prompt text:", repr(tokenizer.decode(display_prompt_ids, skip_special_tokens=True)))

    print("JAX next token:", jax_next, repr(tokenizer.decode([jax_next], skip_special_tokens=False)))
    print("Cactus next token:", cactus_next, repr(tokenizer.decode([cactus_next], skip_special_tokens=False)))

    print(
        "JAX prompt + next:",
        repr(tokenizer.decode(display_prompt_ids + [jax_next], skip_special_tokens=True)),
    )
    print(
        "Cactus prompt + next:",
        repr(tokenizer.decode(display_prompt_ids + [cactus_next], skip_special_tokens=True)),
    )

def llama_config_from_hf(model_id: str):
    hf = AutoConfig.from_pretrained(model_id)

    hidden = int(hf.hidden_size)
    heads = int(hf.num_attention_heads)
    head_dim = int(getattr(hf, "head_dim", hidden // heads))

    return LlamaConfig(
        dim=hidden,
        n_layers=int(hf.num_hidden_layers),
        n_heads=heads,
        n_kv_heads=int(getattr(hf, "num_key_value_heads", heads)),
        head_dim=head_dim,
        intermediate_size=int(hf.intermediate_size),
        vocab_size=int(hf.vocab_size),
        norm_eps=float(getattr(hf, "rms_norm_eps", 1e-5)),
        rope_theta=float(getattr(hf, "rope_theta", 10000.0)),
    )
def fix_zero_norm_weights(flat_state, hidden_size=64, force_all_norms=False):
    fixed = []

    for x in flat_state:
        try:
            if hasattr(x, "dtype") and "key<" in str(x.dtype):
                fixed.append(x)
                continue
        except Exception:
            fixed.append(x)
            continue

        try:
            arr = np.asarray(x)
        except Exception:
            fixed.append(x)
            continue

        is_float = np.issubdtype(arr.dtype, np.floating) or "bfloat16" in str(arr.dtype)
        if not is_float:
            fixed.append(x)
            continue

        arr32 = arr.astype(np.float32) if "bfloat16" in str(arr.dtype) else arr

        is_norm = arr.ndim == 1 and arr.shape[0] == hidden_size

        if is_norm and (force_all_norms or np.count_nonzero(arr32) == 0):
            print("Setting norm weight to ones:", arr.shape, arr.dtype)
            fixed.append(jnp.ones(arr.shape, dtype=x.dtype))
        else:
            fixed.append(x)

    return tuple(fixed)

def numeric_arr(x):
    arr = np.asarray(x)

    # ml_dtypes.bfloat16 often fails np.issubdtype checks.
    if "bfloat16" in str(arr.dtype):
        return arr.astype(np.float32)

    return arr

def print_state_stats(name, flat_state):
    print(f"\n{name} stats:")
    for i, x in enumerate(flat_state):
        try:
            arr = numeric_arr(x)
        except Exception:
            continue

        if not np.issubdtype(arr.dtype, np.number):
            print(f"  [{i:02d}] shape={arr.shape} dtype={arr.dtype} nonnumeric")
            continue

        if arr.size == 0:
            continue

        finite = np.isfinite(arr)
        if not finite.any():
            print(f"  [{i:02d}] shape={arr.shape} dtype={arr.dtype} all nonfinite")
            continue

        print(
            f"  [{i:02d}] shape={arr.shape} dtype={arr.dtype} "
            f"min={float(np.nanmin(arr)):.6g} "
            f"max={float(np.nanmax(arr)):.6g} "
            f"mean={float(np.nanmean(arr)):.6g} "
            f"std={float(np.nanstd(arr)):.6g} "
            f"nonzero={int(np.count_nonzero(arr))}/{arr.size}"
        )

def randomize_param_leaves(flat_state, seed=123):
    rng = np.random.default_rng(seed)
    out = []

    for x in flat_state:
        # Keep JAX PRNG keys untouched.
        try:
            if hasattr(x, "dtype") and "key<" in str(x.dtype):
                out.append(x)
                continue
        except Exception:
            pass

        try:
            arr = np.asarray(x)
        except Exception:
            out.append(x)
            continue

        is_float = np.issubdtype(arr.dtype, np.floating) or "bfloat16" in str(arr.dtype)

        if not is_float:
            out.append(x)
            continue

        scale = 0.02
        gen_dtype = np.float32 if "bfloat16" in str(arr.dtype) else arr.dtype
        new_arr = rng.normal(0.0, scale, size=arr.shape).astype(gen_dtype)

        # Preserve original JAX dtype, especially bf16/fp16.
        out.append(jnp.asarray(new_arr, dtype=x.dtype))

    assert len(out) == len(flat_state), f"leaf count changed: {len(flat_state)} -> {len(out)}"
    return tuple(out)


def const_to_runtime_array(x):
    """
    Convert parsed MLIR constants into NumPy arrays.

    Handles hex float constants like:
      0xFF800000 = -inf float32
      0x7F800000 = +inf float32
    """
    value = x.value if hasattr(x, "value") else x
    arr = np.asarray(value)

    if arr.dtype.kind in ("U", "S", "O"):
        def decode_one(v):
            if isinstance(v, bytes):
                v = v.decode("utf-8")
            if isinstance(v, np.generic):
                v = v.item()

            if isinstance(v, str):
                s = v.strip().lower()

                if s == "true":
                    return 1.0
                if s == "false":
                    return 0.0

                if s.startswith("0x"):
                    h = int(s, 16)

                    # 32-bit float hex encoding.
                    if len(s) <= 10:
                        return np.array([h], dtype=np.uint32).view(np.float32)[0]

                    # 64-bit float hex encoding.
                    return np.array([h], dtype=np.uint64).view(np.float64)[0]

                return float(s)

            return float(v)

        vec = np.vectorize(decode_one, otypes=[np.float32])
        arr = vec(arr)

    if arr.shape == ():
        arr = arr.reshape((1,))

    return arr

def to_runtime_array(x):
    """
    Convert JAX/NNX leaves into NumPy arrays for Cactus.

    PRNGKey dtype cannot be np.asarray(...) directly, so expose its underlying
    uint32 key data. These RNG leaves are not numerically used by the model
    forward, but they may still appear in the exported function signature.
    """
    try:
        if hasattr(x, "dtype") and "key<" in str(x.dtype):
            return np.asarray(jax.random.key_data(x), dtype=np.uint32)
    except Exception:
        pass

    return np.asarray(x)

def summarize(name: str, x):
    arr = np.asarray(x)
    print(f"\n{name}:")
    print("  shape:", arr.shape)
    print("  dtype:", arr.dtype)

    if np.issubdtype(arr.dtype, np.number):
        finite = np.isfinite(arr)
        print("  finite:", bool(finite.all()))
        print("  nan/inf:", int(np.isnan(arr).sum()), "/", int(np.isinf(arr).sum()))
        print("  min/max:", float(np.nanmin(arr)), float(np.nanmax(arr)))
        arr_stats = arr.astype(np.float32) if np.issubdtype(arr.dtype, np.floating) else arr
        print("  mean/std:", float(np.nanmean(arr_stats)), float(np.nanstd(arr_stats)))
        print("  sample:", arr.reshape(-1)[:16])


def compare(name: str, ref, got, atol=3e-2, rtol=3e-2, eval_pos=None):
    ref = np.asarray(ref, dtype=np.float32)
    got = np.asarray(got, dtype=np.float32)

    print(f"\nCOMPARE {name}")
    print("  ref:", ref.shape, ref.dtype)
    print("  got:", got.shape, got.dtype)

    if ref.shape != got.shape:
        print("  SHAPE MISMATCH")
        return False

    mask = np.isfinite(ref) & np.isfinite(got)
    finite_overlap = int(mask.sum())
    print(f"  finite_overlap: {finite_overlap}/{ref.size}")
    print("  ref_all_finite:", bool(np.isfinite(ref).all()))
    print("  got_all_finite:", bool(np.isfinite(got).all()))

    if finite_overlap == 0:
        print("  no finite overlap")
        return False

    diff = np.abs(ref[mask] - got[mask])
    max_diff = float(np.max(diff))
    mean_diff = float(np.mean(diff))
    p95 = float(np.percentile(diff, 95))
    p99 = float(np.percentile(diff, 99))

    ref_f = ref[mask].reshape(-1)
    got_f = got[mask].reshape(-1)
    denom = float(np.linalg.norm(ref_f) * np.linalg.norm(got_f))
    cosine = float(np.dot(ref_f, got_f) / denom) if denom != 0 else 1.0

    print("  max_diff:", max_diff)
    print("  mean_diff:", mean_diff)
    print("  p95_diff:", p95)
    print("  p99_diff:", p99)
    print("  cosine:", cosine)

    close = bool(np.allclose(ref, got, atol=atol, rtol=rtol))
    print("  allclose:", close, f"(atol={atol}, rtol={rtol})")

    # Also compare last-token top-k.
    if ref.ndim == 3:
        if eval_pos is None:
            pos = ref.shape[1] - 1
        else:
            pos = max(0, min(int(eval_pos), ref.shape[1] - 1))

        print("  eval_pos:", pos)

        ref_last = ref[0, pos]
        got_last = got[0, pos]

        ref_argmax = int(np.argmax(ref_last))
        got_argmax = int(np.argmax(got_last))

        print("  ref argmax last:", ref_argmax, float(ref_last[ref_argmax]))
        print("  got argmax last:", got_argmax, float(got_last[got_argmax]))

        ref_top = np.argsort(-ref_last)[:10]
        got_top = np.argsort(-got_last)[:10]

        print("  ref top10:", [(int(i), float(ref_last[i])) for i in ref_top])
        print("  got top10:", [(int(i), float(got_last[i])) for i in got_top])

    same_argmax = True
    if ref.ndim == 3:
        if eval_pos is None:
            pos = ref.shape[1] - 1
        else:
            pos = max(0, min(int(eval_pos), ref.shape[1] - 1))

        same_argmax = int(np.argmax(ref[0, pos])) == int(np.argmax(got[0, pos]))

    return close or (
        cosine > 0.98
        and mean_diff < 0.20
        and p99 < 1.50
        and same_argmax
    )

def make_ids(batch=1, seq=8, vocab=320):
    base = np.array([[2, 11, 277, 256, 1, 283, 44, 91]], dtype=np.int32)

    if seq > base.shape[1]:
        extra = np.arange(base.shape[1], seq, dtype=np.int32)[None, :] + 100
        base = np.concatenate([base, extra], axis=1)

    ids = base[:, :seq]
    ids = np.mod(ids, vocab).astype(np.int32)

    if batch != 1:
        ids = np.broadcast_to(ids, (batch, seq)).copy()

    return jnp.asarray(ids)


def flatten_state_for_export(state):
    flat_state, treedef = jax.tree_util.tree_flatten(state)
    return tuple(flat_state), treedef


def build_model(
    seq: int,
    hf_model_id: str | None = None,
    prompt: str | None = None,
    tokenizer_id: str | None = None,
    config_dim: int | None = None,
    config_layers: int | None = None,
    config_heads: int | None = None,
    config_kv_heads: int | None = None,
    config_head_dim: int | None = None,
    config_ff: int | None = None,
    config_vocab_size: int | None = None,
):
    decode_tokenizer = None
    prompt_len = seq
    print("\n================ BUILD JAXGARDEN LLAMA ================")

    # 1. Build config first.
    if hf_model_id is not None:
        config = llama_config_from_hf(hf_model_id)
    else:
        dim = 128 if config_dim is None else config_dim
        heads = 4 if config_heads is None else config_heads
        head_dim = dim // heads if config_head_dim is None else config_head_dim

        config = LlamaConfig(
            dim=dim,
            n_layers=4 if config_layers is None else config_layers,
            n_heads=heads,
            n_kv_heads=heads if config_kv_heads is None else config_kv_heads,
            head_dim=head_dim,
            intermediate_size=4 * dim if config_ff is None else config_ff,
            vocab_size=32000 if config_vocab_size is None else config_vocab_size,
            norm_eps=1e-5,
            rope_theta=10000.0,
        )

    # 2. Build model.
    model = LlamaForCausalLM(
        config,
        dtype=jnp.float16,
        param_dtype=jnp.float16,
        rngs=nnx.Rngs(0),
    )

    if hf_model_id is not None:
        print("Loading HF weights via JAXgarden:", hf_model_id)
        model.from_hf(
            hf_model_id,
            save_in_orbax=False,
            remove_hf_after_conversion=False,
            force_download=False,
        )

    # 3. Build input IDs.
    tok_source = hf_model_id or tokenizer_id

    if tok_source is not None and prompt is not None:
        print("Tokenizing prompt:", repr(prompt))
        print("Using tokenizer:", tok_source)
        tok = AutoTokenizer.from_pretrained(tok_source)
        decode_tokenizer = tok
        encoded = tok(prompt, return_tensors="np", add_special_tokens=True)
        ids_np = encoded["input_ids"].astype(np.int32)

        print("raw token ids:", ids_np.tolist())
        print("tokenizer vocab size:", getattr(tok, "vocab_size", None))

        raw_prompt_len = int(ids_np.shape[1])

        # Keep fixed seq length for StableHLO.
        if ids_np.shape[1] > seq:
            ids_np = ids_np[:, :seq]
        elif ids_np.shape[1] < seq:
            pad_id = tok.pad_token_id
            if pad_id is None:
                pad_id = tok.eos_token_id
            if pad_id is None:
                pad_id = 0

            pad = np.full((1, seq - ids_np.shape[1]), pad_id, dtype=np.int32)
            ids_np = np.concatenate([ids_np, pad], axis=1)

        prompt_len = min(raw_prompt_len, seq)

        if ids_np.max() >= config.vocab_size:
            raise ValueError(
                f"Tokenizer produced id {ids_np.max()} but model vocab_size={config.vocab_size}."
            )

        input_ids = jnp.asarray(ids_np)
    else:
        input_ids = make_ids(seq=seq, vocab=config.vocab_size)
    # Shape [T,T] works with the model implementation.
    attention_mask = jnp.asarray(np.tril(np.ones((seq, seq), dtype=bool)))

    # Split params separately from RNG/internal state.
    graphdef, param_state, rest_state = nnx.split(model, nnx.Param, ...)

    flat_param_state, param_treedef = flatten_state_for_export(param_state)

    if hf_model_id is None:
        print("Randomizing local test params...")
        flat_param_state = randomize_param_leaves(flat_param_state, seed=123)
        flat_param_state = fix_zero_norm_weights(flat_param_state, config.dim, force_all_norms=True)
    else:
        flat_param_state = fix_zero_norm_weights(flat_param_state, config.dim, force_all_norms=False)

    # Rebuild param_state from the modified flat params.
    param_state = jax.tree_util.tree_unflatten(param_treedef, flat_param_state)

    # Merge once to create a modified model, then split full state for export.
    model = nnx.merge(graphdef, param_state, rest_state)
    graphdef, state = nnx.split(model)
    flat_state, state_treedef = flatten_state_for_export(state)

    # Re-split params from the modified model for Cactus runtime args.
    _, param_state, _rest_state = nnx.split(model, nnx.Param, ...)
    flat_param_state, _param_treedef = flatten_state_for_export(param_state)

    #print_state_stats("FULL STATE", flat_state)
    #print_state_stats("PARAM STATE", flat_param_state)

    print("config:")
    print("  vocab_size:", config.vocab_size)
    print("  dim:", config.dim)
    print("  layers:", config.n_layers)
    print("  heads:", config.n_heads)
    print("  kv_heads:", config.n_kv_heads)
    print("  head_dim:", config.head_dim)
    print("  seq:", seq)

    print("\ninputs:")
    summarize("input_ids", input_ids)
    summarize("attention_mask", attention_mask)

    print("\nstate leaves:", len(flat_state))
    total_params = 0
    for x in flat_state:
        try:
            total_params += int(np.prod(np.asarray(x).shape))
        except Exception:
            pass
    print("state scalar count:", total_params)

    return config, graphdef, state_treedef, flat_state, flat_param_state, input_ids, attention_mask, decode_tokenizer, prompt_len

def run_jax_reference(graphdef, state_treedef, flat_state, input_ids, attention_mask):
    print("\n================ JAX REFERENCE ================")

    def forward_flat(*args):
        *state_args, input_ids_arg, attention_mask_arg = args
        state_arg = jax.tree_util.tree_unflatten(state_treedef, state_args)
        m = nnx.merge(graphdef, state_arg)
        return m(input_ids_arg, attention_mask=attention_mask_arg)

    example_args = tuple(flat_state) + (input_ids, attention_mask)

    t0 = time.perf_counter()
    logits = forward_flat(*example_args)
    jax.block_until_ready(logits)
    t1 = time.perf_counter()

    summarize("JAX logits", logits)
    print("JAX time:", f"{(t1 - t0) * 1000:.3f} ms")

    if np.asarray(logits).ndim == 3:
        last = np.asarray(logits)[0, -1]
        top = np.argsort(-last)[:10]
        print("JAX top10 last:", [(int(i), float(last[i])) for i in top])

    return forward_flat, example_args, np.asarray(logits)


def export_stablehlo(forward_flat, example_args, out_path: str):
    print("\n================ EXPORT STABLEHLO ================")
    print("exporting...")

    t0 = time.perf_counter()
    mlir = jax_to_mlir(forward_flat, example_args)
    t1 = time.perf_counter()

    Path(out_path).write_text(mlir)

    print("saved:", out_path)
    print("MLIR chars:", len(mlir))
    print("export time:", f"{(t1 - t0) * 1000:.3f} ms")

    return mlir


def lower_and_execute_cactus(mlir_path: str, example_args, strict_math: bool):
    print("\n================ PARSE MLIR ================")

    mlir_text = Path(mlir_path).read_text()
    ir = parse_mlir(mlir_text)

    print("IR summary:")
    print("  inputs:", len(ir.inputs))
    print("  outputs:", ir.outputs)
    print("  nodes:", len(ir.order))
    print("  constants:", len(getattr(ir, "constants", {})))

    op_counts = {}
    for nid in ir.order:
        op = ir.nodes[nid].op
        op_counts[op] = op_counts.get(op, 0) + 1

    print("\nOps:")
    for op, count in sorted(op_counts.items()):
        print(f"  {op:35s} {count}")

    print("\n================ LOWER TO CACTUS ================")

    runtime_args = [to_runtime_array(x) for x in example_args]

    g, env = lower_to_cactus(
        ir,
        patterns=["default"],
        verbose=False,
    )

    if len(runtime_args) < len(ir.inputs):
        raise RuntimeError(
            f"Not enough runtime args: have {len(runtime_args)}, MLIR needs {len(ir.inputs)}"
        )
    

    print("runtime args:", len(runtime_args))
    print("MLIR inputs:", len(ir.inputs))
    for i, (ssa, arr) in enumerate(zip(ir.inputs, runtime_args)):
        print(f"  input[{i:02d}] {ssa:8s} shape={arr.shape} dtype={arr.dtype}")
    for ssa, arr in zip(ir.inputs, runtime_args):
        g.set_input(env[ssa], arr)

    for ssa, const in ir.constants.items():
        if ssa not in env:
            continue
        value = const_to_runtime_array(const)
        g.set_input(env[ssa], value)

    print("set MLIR inputs:", len(ir.inputs))
    print("set constants:", len(ir.constants))

    print("\n================ EXECUTE CACTUS ================")

    t0 = time.perf_counter()


    print("\n================ GRAPH DEBUG BEFORE EXECUTE ================")

    # Try to inspect graph/tensor/node APIs without assuming exact names.
    print("Graph debug methods:", [x for x in dir(g) if "node" in x.lower() or "debug" in x.lower() or "dump" in x.lower()])
    print("Graph tensor-ish methods:", [x for x in dir(g) if "tensor" in x.lower() or "input" in x.lower() or "output" in x.lower()])

    # Print output tensor info.
    try:
        out_ssa = ir.outputs[0]
        out_tensor = env[out_ssa]
        print("Output tensor:", out_tensor)
        print("Output shape:", getattr(out_tensor, "shape", None))
        print("Output dtype:", getattr(out_tensor, "dtype", None))
    except Exception as e:
        print("Could not inspect output tensor:", e)
    g.execute()
    t1 = time.perf_counter()

    print("execute time:", f"{(t1 - t0) * 1000:.3f} ms")

    print("\n================ READ OUTPUT ================")

    out_ssa = ir.outputs[0]
    out_tensor = env[out_ssa]

    t0 = time.perf_counter()
    if hasattr(out_tensor, "read"):
        cactus_logits = np.asarray(out_tensor.read())
    elif hasattr(out_tensor, "numpy"):
        cactus_logits = np.asarray(out_tensor.numpy())
    elif hasattr(g, "get_tensor"):
        cactus_logits = np.asarray(g.get_tensor(out_tensor))
    elif hasattr(g, "read_output"):
        cactus_logits = np.asarray(g.read_output(out_tensor))
    else:
        raise AttributeError(
            f"Don't know how to read Cactus tensor. "
            f"Graph has: {[x for x in dir(g) if 'read' in x or 'tensor' in x or 'output' in x]}; "
            f"Tensor has: {[x for x in dir(out_tensor) if 'read' in x or 'numpy' in x or 'data' in x]}"
        )
    t1 = time.perf_counter()

    summarize("Cactus logits", cactus_logits)
    print("readback time:", f"{(t1 - t0) * 1000:.3f} ms")

    return cactus_logits


def benchmark_cactus(mlir_path: str, example_args, strict_math: bool, iters: int):
    if iters <= 0:
        return

    print("\n================ CACTUS BENCHMARK ================")

    mlir_text = Path(mlir_path).read_text()
    ir = parse_mlir(mlir_text)
    runtime_args = [np.asarray(x) for x in example_args]

    try:
        g, env = lower_to_cactus(ir, runtime_args=runtime_args, strict_math=strict_math)
    except TypeError:
        try:
            g, env = lower_to_cactus(ir, runtime_inputs=runtime_args, strict_math=strict_math)
        except TypeError:
            try:
                g, env = lower_to_cactus(ir, runtime_args, strict_math=strict_math)
            except TypeError:
                g, env = lower_to_cactus(ir, runtime_args)

    out_tensor = env[ir.outputs[0]]

    # Warmup.
    for _ in range(5):
        g.execute()

    exec_times = []
    exec_read_times = []

    for _ in range(iters):
        t0 = time.perf_counter()
        g.execute()
        t1 = time.perf_counter()
        _ = np.asarray(g.read_tensor(out_tensor))
        t2 = time.perf_counter()

        exec_times.append((t1 - t0) * 1000)
        exec_read_times.append((t2 - t0) * 1000)

    print("iters:", iters)
    print("execute mean ms:", float(np.mean(exec_times)))
    print("execute median ms:", float(np.median(exec_times)))
    print("execute p95 ms:", float(np.percentile(exec_times, 95)))
    print("exec+read mean ms:", float(np.mean(exec_read_times)))
    print("exec+read median ms:", float(np.median(exec_read_times)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=8)
    ap.add_argument("--mlir-out", default="jaxgarden_llama_tiny.stablehlo.mlir")
    ap.add_argument("--atol", type=float, default=3e-2)
    ap.add_argument("--rtol", type=float, default=3e-2)
    ap.add_argument("--bench-iters", type=int, default=0)
    ap.add_argument(
        "--strict-math",
        default="false",
        choices=["true", "false", "1", "0", "yes", "no"],
    )
    ap.add_argument("--hf-model-id", default=None)
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--dim", type=int, default=None)
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--heads", type=int, default=None)
    ap.add_argument("--kv-heads", type=int, default=None)
    ap.add_argument("--head-dim", type=int, default=None)
    ap.add_argument("--ff", type=int, default=None)
    ap.add_argument("--vocab-size", type=int, default=None)
    ap.add_argument("--tokenizer-id", default=None)
    args = ap.parse_args()

    strict_math = args.strict_math.lower() in ("true", "1", "yes")

    print("\n================ JAXGARDEN LLAMA CACTUS TEST ================")
    print("seq:", args.seq)
    print("strict_math:", strict_math)

    (
        config,
        graphdef,
        state_treedef,
        flat_state,
        flat_param_state,
        input_ids,
        attention_mask,
        decode_tokenizer,
        prompt_len,
    ) = build_model(
            args.seq,
            args.hf_model_id,
            args.prompt,
            tokenizer_id=args.tokenizer_id,
            config_dim=args.dim,
            config_layers=args.layers,
            config_heads=args.heads,
            config_kv_heads=args.kv_heads,
            config_head_dim=args.head_dim,
            config_ff=args.ff,
            config_vocab_size=args.vocab_size,
        )
    
    forward_flat, example_args, jax_logits = run_jax_reference(
        graphdef,
        state_treedef,
        flat_state,
        input_ids,
        attention_mask,
    )

    export_stablehlo(forward_flat, example_args, args.mlir_out)
    cactus_runtime_args = tuple(flat_param_state) + (input_ids, attention_mask)
    cactus_logits = lower_and_execute_cactus(
        args.mlir_out,
        cactus_runtime_args,
        strict_math=strict_math,
    )

    ok = compare(
        "JAXgarden Llama JAX vs Cactus",
        jax_logits,
        cactus_logits,
        atol=args.atol,
        rtol=args.rtol,
        eval_pos=prompt_len - 1,
    )
    print_decoded_outputs(
        decode_tokenizer,
        input_ids,
        jax_logits,
        cactus_logits,
        prompt_len=prompt_len,
    )

    benchmark_cactus(
        args.mlir_out,
        cactus_runtime_args,
        strict_math=strict_math,
        iters=args.bench_iters,
    )

    print("\n================ FINAL ================")
    print("JAX/Cactus matched:", ok)

    if ok:
        print("JAXGARDEN LLAMA CACTUS TEST PASSED ✅")
    else:
        print("JAXGARDEN LLAMA CACTUS TEST FAILED ❌")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
