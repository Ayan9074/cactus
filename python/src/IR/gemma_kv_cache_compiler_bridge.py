import os
import re
import time
from pathlib import Path
import numpy as np
import torch
import jax
import jax.numpy as jnp
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.graph import Graph
from src.tensor_io import save_tensor_with_header
from src.IR.stablehlo_ir import parse_stablehlo_ops
from src.IR.lower_to_cactus_rewrite import (
    lower_to_cactus as lower_to_cactus_rewrite,
    find_attention_blocks as find_attention_blocks_rewrite,
)
from src.IR.lower_to_cactus import (
    lower_to_cactus as lower_to_cactus_legacy,
    find_attention_blocks as find_attention_blocks_legacy,
)


MODEL_NAME = os.getenv("MODEL_NAME", "google/gemma-2b")
PROMPT = os.getenv("PROMPT", "In a surprising discovery, scientists found that")
GEN_STEPS = int(os.getenv("GEN_STEPS", "8"))
MAX_SEQ_LEN = int(os.getenv("MAX_SEQ_LEN", "64"))
TOP_K = int(os.getenv("TOP_K", "50"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.8"))
REPETITION_PENALTY = float(os.getenv("REPETITION_PENALTY", "1.2"))
FP32_LAYERNORM = os.getenv("FP32_LAYERNORM", "1") == "1"
FP32_QKV_MATMUL = os.getenv("FP32_QKV_MATMUL", "0") == "1"
ENABLE_ATTENTION_FUSION = os.getenv("ENABLE_ATTENTION_FUSION", "0") == "1"
USE_KV_CACHE = os.getenv("USE_KV_CACHE", "1") == "1"
ENABLE_RMSNORM_FUSION = os.getenv("ENABLE_RMSNORM_FUSION", "1") == "1"
SANITIZE_LOGITS = os.getenv("SANITIZE_LOGITS", "0") == "1"
DEBUG_NUMERICS = os.getenv("DEBUG_NUMERICS", "0") == "1"
SKIP_REF_COMPARE = os.getenv("SKIP_REF_COMPARE", "0") == "1"
BENCH_DETAIL = os.getenv("BENCH_DETAIL", "1") == "1"
COMPARE_PREFILL = os.getenv("COMPARE_PREFILL", "0") == "1"
COMPARE_JAX_SINGLE = os.getenv("COMPARE_JAX_SINGLE", "0") == "1"
COMPARE_JAX_FP16_SINGLE = os.getenv("COMPARE_JAX_FP16_SINGLE", "0") == "1"
GRAPH_FP32 = os.getenv("GRAPH_FP32", "0") == "1"
POST_MATMUL_FP32 = os.getenv("POST_MATMUL_FP32", "0") == "1"
RECOMPILE_PER_TOKEN = os.getenv("RECOMPILE_PER_TOKEN", "0") == "1"
USE_ADDITIVE_KV_MASK = os.getenv("USE_ADDITIVE_KV_MASK", "1") == "1"
WEIGHT_CACHE_ROOT = Path(os.getenv("WEIGHT_CACHE_ROOT", "/tmp/cactus_ir_mmap_weights"))
FORCE_REBUILD_WEIGHT_FILES = os.getenv("FORCE_REBUILD_WEIGHT_FILES", "0") == "1"
LOWERER_MODE = os.getenv("LOWERER", "auto").strip().lower()  # auto|rewrite|legacy
NORM_WEIGHT_OFFSET_MODE = os.getenv("NORM_WEIGHT_OFFSET", "auto").strip().lower()  # auto|on|off


def extract_return_name(ir):
    for line in ir.splitlines():
        line = line.strip()
        if line.startswith("return "):
            m = re.search(r"return\s+(%[\w\d_]+)", line)
            if m:
                return m.group(1)
            raise RuntimeError(f"Bad return line: {line}")
    raise RuntimeError("No return found")


def sample_logits(logits, generated):
    logits = logits.astype(np.float32).copy()
    if not np.all(np.isfinite(logits)):
        n_nan = int(np.isnan(logits).sum())
        n_inf = int(np.isinf(logits).sum())
        bad_idx = np.where(~np.isfinite(logits))[0]
        first_bad = int(bad_idx[0]) if bad_idx.size else -1
        finite = logits[np.isfinite(logits)]
        fmin = float(finite.min()) if finite.size else float("nan")
        fmax = float(finite.max()) if finite.size else float("nan")
        if SANITIZE_LOGITS:
            logits = np.nan_to_num(logits, nan=0.0, posinf=80.0, neginf=-80.0)
        else:
            raise RuntimeError(
                f"Non-finite logits before sampling: nan={n_nan}, inf={n_inf}, "
                f"first_bad_index={first_bad}, finite_min={fmin:.5f}, finite_max={fmax:.5f}"
            )
    for tok in set(generated[-50:]):
        if logits[tok] > 0:
            logits[tok] /= REPETITION_PENALTY
        else:
            logits[tok] *= REPETITION_PENALTY

    logits = logits / TEMPERATURE
    top_idx = np.argpartition(logits, -TOP_K)[-TOP_K:]
    top_logits = logits[top_idx]
    probs = np.exp(top_logits - np.max(top_logits))
    probs /= probs.sum()
    return int(np.random.choice(top_idx, p=probs))


def compare_logits(label, ref, out):
    ref = ref.astype(np.float32)
    out = out.astype(np.float32)
    diff = np.abs(ref - out)
    cos = float(np.dot(ref, out) / (np.linalg.norm(ref) * np.linalg.norm(out) + 1e-8))
    ref_top = np.argsort(ref)[-5:][::-1]
    out_top = np.argsort(out)[-5:][::-1]
    overlap = len(set(ref_top.tolist()) & set(out_top.tolist()))


    ref_norm = np.linalg.norm(ref)
    out_norm = np.linalg.norm(out)

    if ref_norm < 1e-8 and out_norm < 1e-8:
        cos = 1.0
    elif ref_norm < 1e-8 or out_norm < 1e-8:
        cos = 0.0
    else:
        cos = float(np.dot(ref, out) / (ref_norm * out_norm))
    print(
        f"{label}: max={float(diff.max()):.5f} "
        f"mean={float(diff.mean()):.5f} cos={cos:.6f} top5={overlap}/5"
    )
    return float(overlap) / 5.0


def rope_cos_sin_np(position, dim, theta):
    inv_freq = 1.0 / (theta ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
    freqs = np.outer(np.array([position], dtype=np.float32), inv_freq)
    emb = np.concatenate([freqs, freqs], axis=-1)
    return np.cos(emb).astype(np.float32), np.sin(emb).astype(np.float32)


def _sanitize_filename_part(value):
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", str(value))


def _weight_cache_dir():
    d = WEIGHT_CACHE_ROOT / _sanitize_filename_part(MODEL_NAME)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _weight_cache_filename(arg_key, spec, arr):
    layer = spec.get("layer")
    name = _sanitize_filename_part(spec.get("name", "weight"))
    shape = "x".join(str(int(x)) for x in arr.shape)
    if layer is None:
        stem = f"{arg_key[1:]}_{name}_{shape}"
    else:
        stem = f"{arg_key[1:]}_layer_{int(layer):03d}_{name}_{shape}"
    return f"{stem}.weights"


def _materialize_weight_file(arg_key, spec, arr):
    out_path = _weight_cache_dir() / _weight_cache_filename(arg_key, spec, arr)
    if FORCE_REBUILD_WEIGHT_FILES or not out_path.exists():
        arr_fp16 = np.ascontiguousarray(arr.astype(np.float16))
        save_tensor_with_header(
            arr_fp16,
            out_path,
            precision="FP16",
            transpose=False,
        )
    return out_path


def rotate_half_jnp(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return jnp.concatenate([-x2, x1], axis=-1)


def apply_rope_2d_jnp(x, cos, sin, num_heads, head_dim):
    # x: (1, D_MODEL), cos/sin: (1, HEAD_DIM)
    xh = jnp.reshape(x, (num_heads, head_dim))
    cosh = jnp.broadcast_to(cos, (num_heads, head_dim))
    sinh = jnp.broadcast_to(sin, (num_heads, head_dim))
    xh = (xh * cosh) + (rotate_half_jnp(xh) * sinh)
    return jnp.reshape(xh, (1, num_heads * head_dim))


def rms_norm_jnp(x, w, eps):
    var = jnp.mean(x * x, axis=-1, keepdims=True)
    return x * jax.lax.rsqrt(var + eps) * w


def gelu_tanh_jnp(x):
    return 0.5 * x * (1.0 + jnp.tanh(jnp.sqrt(2.0 / jnp.pi) * (x + 0.044715 * x * x * x)))


print("Loading Gemma model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,
    local_files_only=False,
)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=False)
config = model.config
state = model.state_dict()

NUM_LAYERS = config.num_hidden_layers
NUM_HEADS = config.num_attention_heads
NUM_KV_HEADS = config.num_key_value_heads
HEAD_DIM = config.head_dim
D_MODEL = config.hidden_size
ATTN_PROJ_DIM = NUM_HEADS * HEAD_DIM
RMS_EPS = float(config.rms_norm_eps)
VOCAB = config.vocab_size
LAYER_TYPES = list(getattr(config, "layer_types", ["full_attention"] * NUM_LAYERS))
if len(LAYER_TYPES) != NUM_LAYERS:
    LAYER_TYPES = (LAYER_TYPES + ["full_attention"] * NUM_LAYERS)[:NUM_LAYERS]
ROPE_PARAMS = getattr(config, "rope_parameters", None)
if isinstance(ROPE_PARAMS, dict):
    LOCAL_ROPE_THETA = float(ROPE_PARAMS.get("sliding_attention", {}).get("rope_theta", 10000.0))
    GLOBAL_ROPE_THETA = float(ROPE_PARAMS.get("full_attention", {}).get("rope_theta", LOCAL_ROPE_THETA))
else:
    theta = float(getattr(config, "rope_theta", 10000.0))
    LOCAL_ROPE_THETA = theta
    GLOBAL_ROPE_THETA = theta
LAYER_IS_GLOBAL = [lt == "full_attention" for lt in LAYER_TYPES]
_query_pre_attn_scalar = getattr(config, "query_pre_attn_scalar", None)
if _query_pre_attn_scalar is None:
    _query_pre_attn_scalar = HEAD_DIM
ATTN_SCALING = 1.0 / np.sqrt(float(_query_pre_attn_scalar))

if NORM_WEIGHT_OFFSET_MODE == "on":
    USE_NORM_WEIGHT_OFFSET = True
elif NORM_WEIGHT_OFFSET_MODE == "off":
    USE_NORM_WEIGHT_OFFSET = False
else:
    # Gemma-3 family stores RMSNorm as offset weights; Gemma-2 uses direct scales.
    USE_NORM_WEIGHT_OFFSET = str(getattr(config, "model_type", "")) in {"gemma3", "gemma3_text"}

HAS_QK_NORM = (
    f"model.layers.0.self_attn.q_norm.weight" in state
    and f"model.layers.0.self_attn.k_norm.weight" in state
)
HAS_PRE_FFN_NORM = f"model.layers.0.pre_feedforward_layernorm.weight" in state
HAS_POST_FFN_NORM = f"model.layers.0.post_feedforward_layernorm.weight" in state


def to_np(name):
    return state[name].detach().float().cpu().numpy()


def one_plus_or_ones(name, shape):
    if name in state:
        w = to_np(name)
        return (1.0 + w) if USE_NORM_WEIGHT_OFFSET else w
    return np.ones(shape, dtype=np.float32)


embed = to_np("model.embed_tokens.weight")
_final_norm = to_np("model.norm.weight")
final_norm_w = (1.0 + _final_norm) if USE_NORM_WEIGHT_OFFSET else _final_norm
lm_src = to_np("lm_head.weight") if "lm_head.weight" in state else embed
lm_head = lm_src.T.copy()  # (D_MODEL, VOCAB)


def expand_kv_to_q_heads(w):
    # w: (NUM_KV_HEADS * HEAD_DIM, D_MODEL) -> (NUM_HEADS * HEAD_DIM, D_MODEL)
    if NUM_KV_HEADS == NUM_HEADS:
        return np.ascontiguousarray(w)
    reps = NUM_HEADS // NUM_KV_HEADS
    w3 = w.reshape(NUM_KV_HEADS, HEAD_DIM, D_MODEL)
    w3 = np.repeat(w3, reps, axis=0)
    return np.ascontiguousarray(w3.reshape(NUM_HEADS * HEAD_DIM, D_MODEL))


blocks = []
for i in range(NUM_LAYERS):
    ln1 = one_plus_or_ones(f"model.layers.{i}.input_layernorm.weight", (D_MODEL,))
    ln2 = one_plus_or_ones(f"model.layers.{i}.post_attention_layernorm.weight", (D_MODEL,))
    ln3 = one_plus_or_ones(f"model.layers.{i}.pre_feedforward_layernorm.weight", (D_MODEL,))
    ln4 = one_plus_or_ones(f"model.layers.{i}.post_feedforward_layernorm.weight", (D_MODEL,))
    qn_name = f"model.layers.{i}.self_attn.q_norm.weight"
    kn_name = f"model.layers.{i}.self_attn.k_norm.weight"
    qn = one_plus_or_ones(qn_name, (HEAD_DIM,))
    kn = one_plus_or_ones(kn_name, (HEAD_DIM,))

    q = np.ascontiguousarray(to_np(f"model.layers.{i}.self_attn.q_proj.weight"))
    k = expand_kv_to_q_heads(to_np(f"model.layers.{i}.self_attn.k_proj.weight"))
    v = expand_kv_to_q_heads(to_np(f"model.layers.{i}.self_attn.v_proj.weight"))
    o = np.ascontiguousarray(to_np(f"model.layers.{i}.self_attn.o_proj.weight"))

    if q.shape[0] != ATTN_PROJ_DIM or k.shape[0] != ATTN_PROJ_DIM or v.shape[0] != ATTN_PROJ_DIM:
        raise RuntimeError(
            f"Layer {i}: unexpected q/k/v proj rows "
            f"(q={q.shape[0]}, k={k.shape[0]}, v={v.shape[0]}), expected {ATTN_PROJ_DIM}"
        )
    if o.shape[1] != ATTN_PROJ_DIM:
        raise RuntimeError(
            f"Layer {i}: unexpected o_proj input dim {o.shape[1]}, expected {ATTN_PROJ_DIM}"
        )

    gate = np.ascontiguousarray(to_np(f"model.layers.{i}.mlp.gate_proj.weight"))
    up = np.ascontiguousarray(to_np(f"model.layers.{i}.mlp.up_proj.weight"))
    down = np.ascontiguousarray(to_np(f"model.layers.{i}.mlp.down_proj.weight"))

    blocks.append((ln1, ln2, ln3, ln4, qn, kn, q, k, v, o, gate, up, down))


def gemma_one_token_graph(token_embed, pos_embed, rope_cos_local, rope_sin_local, rope_cos_global, rope_sin_global, *params):
    block_params = params[:-2]
    final_w, lm = params[-2:]

    x = token_embed
    x = x * jnp.sqrt(jnp.array(float(D_MODEL), dtype=x.dtype))
    x = x + pos_embed

    for i in range(NUM_LAYERS):
        o = i * 13
        ln1, ln2, ln3, ln4, qn, kn, wq, wk, wv, wo, w_gate, w_up, w_down = block_params[o:o + 13]

        residual = x
        h = rms_norm_jnp(x, ln1, RMS_EPS)

        q = h @ wq.T
        k = h @ wk.T
        v = h @ wv.T

        # CRITICAL: reshape before attention
        q = jnp.reshape(q, (1, NUM_HEADS, HEAD_DIM))
        k = jnp.reshape(k, (1, NUM_HEADS, HEAD_DIM))
        v = jnp.reshape(v, (1, NUM_HEADS, HEAD_DIM))

        # Gemma-3 variants apply q_norm/k_norm before RoPE; Gemma-2 does not.
        if HAS_QK_NORM:
            q = rms_norm_jnp(q, qn, RMS_EPS)
            k = rms_norm_jnp(k, kn, RMS_EPS)

        if LAYER_IS_GLOBAL[i]:
            rope_cos = rope_cos_global
            rope_sin = rope_sin_global
        else:
            rope_cos = rope_cos_local
            rope_sin = rope_sin_local
        cos = jnp.broadcast_to(rope_cos, (NUM_HEADS, HEAD_DIM))
        sin = jnp.broadcast_to(rope_sin, (NUM_HEADS, HEAD_DIM))

        q0 = q[0]
        k0 = k[0]

        q0 = (q0 * cos) + (rotate_half_jnp(q0) * sin)
        k0 = (k0 * cos) + (rotate_half_jnp(k0) * sin)

        q = jnp.reshape(q0, (1, NUM_HEADS, HEAD_DIM))
        k = jnp.reshape(k0, (1, NUM_HEADS, HEAD_DIM))

        # Real per-head attention:
        # q: (T,H,D), k: (S,H,D) -> scores: (H,T,S)
        # q/k/v: (T, H, D), here T=1
        q_b = jnp.transpose(q, (1, 0, 2))   # (H, T, D)
        k_b = jnp.transpose(k, (1, 2, 0))   # (H, D, T)
        v_b = jnp.transpose(v, (1, 0, 2))   # (H, T, D)

        scores = jnp.matmul(q_b, k_b) * jnp.array(float(ATTN_SCALING), dtype=q.dtype)  # (H, T, T)

        causal = jnp.tril(jnp.ones((1, 1), dtype=scores.dtype))
        scores = scores + (1.0 - causal)[None, :, :] * -1e4

        probs = jax.nn.softmax(scores, axis=-1)

        attn_b = jnp.matmul(probs, v_b)     # (H, T, D)
        attn = jnp.transpose(attn_b, (1, 0, 2))  # (T, H, D)
        # Do not assume hidden_size == num_heads * head_dim (false for Gemma-3-270M).
        attn_out = jnp.reshape(attn, (1, ATTN_PROJ_DIM))

        attn_proj = attn_out @ wo.T
        if HAS_PRE_FFN_NORM and HAS_POST_FFN_NORM:
            # Gemma-3-style block path.
            attn_proj = rms_norm_jnp(attn_proj, ln2, RMS_EPS)
            x = residual + attn_proj

            residual_ff = x
            h2 = rms_norm_jnp(x, ln3, RMS_EPS)
            gate = h2 @ w_gate.T
            up = h2 @ w_up.T
            mlp = gelu_tanh_jnp(gate) * up
            mlp_proj = mlp @ w_down.T
            mlp_proj = rms_norm_jnp(mlp_proj, ln4, RMS_EPS)
            x = residual_ff + mlp_proj
        else:
            # Gemma-2-style block path.
            x = residual + attn_proj
            residual_ff = x
            h2 = rms_norm_jnp(x, ln2, RMS_EPS)
            gate = h2 @ w_gate.T
            up = h2 @ w_up.T
            mlp = gelu_tanh_jnp(gate) * up
            mlp_proj = mlp @ w_down.T
            x = residual_ff + mlp_proj

    x = rms_norm_jnp(x, final_w, RMS_EPS)
    return x


def hf_logits(tokens):
    input_ids = torch.tensor([tokens], dtype=torch.long)
    with torch.no_grad():
        out = model(input_ids=input_ids)
    return out.logits[0, -1].detach().float().cpu().numpy()


flat = []
for b in blocks:
    flat.extend(b)


def build_jax_inputs(token_id, pos):
    token_one_hot = np.zeros((1, VOCAB), dtype=np.float32)
    token_one_hot[0, int(token_id)] = 1.0
    pos_embed = embed[int(pos):int(pos) + 1] * 0.0
    # actual position embedding is additive learned for GPT style; Gemma uses no learned pos emb.
    # keep arg for dynamic position path; zero vector keeps semantics.
    pos_embed = np.zeros((1, D_MODEL), dtype=np.float32)
    rope_cos_local, rope_sin_local = rope_cos_sin_np(int(pos), HEAD_DIM, LOCAL_ROPE_THETA)
    rope_cos_global, rope_sin_global = rope_cos_sin_np(int(pos), HEAD_DIM, GLOBAL_ROPE_THETA)
    token_embed = embed[int(token_id):int(token_id) + 1].astype(np.float32)

    return (
        token_embed,
        pos_embed,
        rope_cos_local,
        rope_sin_local,
        rope_cos_global,
        rope_sin_global,
        *flat,
        final_norm_w,
        lm_head,
    )


def build_arg_specs():
    specs = {
        "%arg0": {"kind": "runtime", "name": "token_embed"},
        "%arg1": {"kind": "runtime", "name": "pos_embed"},
        "%arg2": {"kind": "runtime", "name": "rope_cos_local"},
        "%arg3": {"kind": "runtime", "name": "rope_sin_local"},
        "%arg4": {"kind": "runtime", "name": "rope_cos_global"},
        "%arg5": {"kind": "runtime", "name": "rope_sin_global"},
    }

    arg = 6
    layer_names = [
        "input_norm",
        "post_attn_norm",
        "pre_ffn_norm",
        "post_ffn_norm",
        "attn_q_norm",
        "attn_k_norm",
        "attn_q",
        "attn_k",
        "attn_v",
        "attn_output",
        "ffn_gate",
        "ffn_up",
        "ffn_down",
    ]
    matmul_weight_names = {
        "attn_q",
        "attn_k",
        "attn_v",
        "attn_output",
        "ffn_gate",
        "ffn_up",
        "ffn_down",
    }

    for layer in range(NUM_LAYERS):
        for name in layer_names:
            is_matmul_weight = name in matmul_weight_names
            specs[f"%arg{arg}"] = {
                "kind": "weight",
                "layer": layer,
                "name": name,
                "matmul_weight": is_matmul_weight,
                "pretransposed_rhs": is_matmul_weight,
            }
            arg += 1

    specs[f"%arg{arg}"] = {
        "kind": "weight",
        "name": "output_norm",
        "matmul_weight": False,
        "pretransposed_rhs": False,
    }
    arg += 1

    specs[f"%arg{arg}"] = {
        "kind": "weight",
        "name": "lm_head",
        "matmul_weight": True,
        # lm_head is already (D_MODEL, VOCAB) and should not be treated as pretransposed.
        "pretransposed_rhs": False,
    }
    return specs


print("Exporting Gemma StableHLO (one-token decode graph)...")
jit_decode = jax.jit(gemma_one_token_graph)
sample = build_jax_inputs(0, 0)
sample_jax = [jnp.array(x, dtype=jnp.float32) for x in sample]
lowered = jit_decode.lower(*sample_jax)
ir = str(lowered.compiler_ir(dialect="stablehlo"))
return_name = extract_return_name(ir)
nodes = parse_stablehlo_ops(ir)
compile_args = getattr(getattr(lowered, "_lowering", None), "compile_args", {}) or {}
kept_var_idx = sorted(int(i) for i in compile_args.get("kept_var_idx", set(range(len(sample_jax)))))
if not kept_var_idx:
    kept_var_idx = list(range(len(sample_jax)))

full_arg_specs = build_arg_specs()
compact_to_full_arg_idx = {ci: fi for ci, fi in enumerate(kept_var_idx)}
compact_arg_specs = {}
for compact_idx, full_idx in compact_to_full_arg_idx.items():
    full_key = f"%arg{full_idx}"
    spec = dict(full_arg_specs.get(full_key, {"kind": "runtime", "name": f"arg{full_idx}"}))
    spec["original_arg_index"] = full_idx
    spec["original_arg_key"] = full_key
    compact_arg_specs[f"%arg{compact_idx}"] = spec

rewrite_attn_blocks = find_attention_blocks_rewrite(nodes)
legacy_attn_blocks = None
selected_lower_to_cactus = lower_to_cactus_rewrite
selected_find_attention_source = "rewrite"

if LOWERER_MODE == "legacy":
    selected_lower_to_cactus = lower_to_cactus_legacy
    legacy_attn_blocks = find_attention_blocks_legacy(nodes)
    attn_blocks = legacy_attn_blocks
    selected_find_attention_source = "legacy"
elif LOWERER_MODE == "rewrite":
    attn_blocks = rewrite_attn_blocks
else:
    # auto: keep rewrite by default, but if KV decode is requested and rewrite
    # cannot find fused attention, fallback to legacy lowerer for decode correctness.
    if ENABLE_ATTENTION_FUSION and USE_KV_CACHE and len(rewrite_attn_blocks) == 0:
        legacy_attn_blocks = find_attention_blocks_legacy(nodes)
        if len(legacy_attn_blocks) > 0:
            selected_lower_to_cactus = lower_to_cactus_legacy
            attn_blocks = legacy_attn_blocks
            selected_find_attention_source = "legacy(auto-fallback)"
        else:
            attn_blocks = rewrite_attn_blocks
    else:
        attn_blocks = rewrite_attn_blocks

attn_out_names = [nodes[b["out"]].name for b in attn_blocks]
FUSION_KV_ACTIVE = ENABLE_ATTENTION_FUSION and USE_KV_CACHE and (len(attn_out_names) > 0)
print(f"Parsed {len(nodes)} nodes, fused attention candidates={len(attn_out_names)}")
print(f"Selected lowerer: {selected_find_attention_source}")
if kept_var_idx != list(range(len(sample_jax))):
    print(
        f"StableHLO pruned unused args: kept {len(kept_var_idx)}/{len(sample_jax)} "
        f"(first kept indices: {kept_var_idx[:12]})"
    )
print(f"Attention fusion enabled: {ENABLE_ATTENTION_FUSION}")
print(f"KV cache enabled: {USE_KV_CACHE}")
print(f"KV fusion active: {FUSION_KV_ACTIVE}")
print(f"Norm weight offset: {USE_NORM_WEIGHT_OFFSET} (mode={NORM_WEIGHT_OFFSET_MODE})")
print(
    f"Block features: qk_norm={HAS_QK_NORM}, "
    f"pre_ffn_norm={HAS_PRE_FFN_NORM}, post_ffn_norm={HAS_POST_FFN_NORM}"
)
if ENABLE_ATTENTION_FUSION and USE_KV_CACHE and not FUSION_KV_ACTIVE:
    print("Warning: rewrite lowerer reported no fused attention blocks; running without fused KV append.")
print(f"RMSNorm fusion enabled: {ENABLE_RMSNORM_FUSION}")
print(f"Recompile per token: {RECOMPILE_PER_TOKEN}")
print(f"Use additive KV mask: {USE_ADDITIVE_KV_MASK}")
print(f"Skip HF reference compare: {SKIP_REF_COMPARE}")

cur_tokens = tokenizer.encode(PROMPT)
if MAX_SEQ_LEN <= len(cur_tokens):
    MAX_SEQ_LEN = len(cur_tokens) + GEN_STEPS + 8

k_cache_buffers = [
    np.zeros((1, MAX_SEQ_LEN, NUM_HEADS, HEAD_DIM), dtype=np.float16)
    for _ in range(NUM_LAYERS)
]
v_cache_buffers = [
    np.zeros((1, MAX_SEQ_LEN, NUM_HEADS, HEAD_DIM), dtype=np.float16)
    for _ in range(NUM_LAYERS)
]
cache_len = 0

compiled_graph = None
compiled_env = None
compiled_input_tensors = {}
compiled_kv_cache_tensors = []
compiled_mask_tensor = None
compiled_fused_attn_out_names = []
compiled_debug_taps = None
compiled_input_np_dtype = np.float16


def build_kv_additive_mask(cur_cache_len):
    # Match the StableHLO/JAX masking constant used in the decode graph.
    m = np.full((1, NUM_HEADS, 1, MAX_SEQ_LEN + 1), np.float16(-10000.0), dtype=np.float16)
    if cur_cache_len > 0:
        m[..., :cur_cache_len] = 0.0
    m[..., MAX_SEQ_LEN] = 0.0
    return m


def init_compiled_graph(cur_cache_len=0):
    global compiled_graph, compiled_env
    global compiled_input_tensors, compiled_kv_cache_tensors, compiled_mask_tensor
    global compiled_fused_attn_out_names, compiled_debug_taps, compiled_input_np_dtype

    if compiled_graph is not None:
        return

    g = Graph()
    input_map = {}
    input_shapes = {}
    full_build_inputs = build_jax_inputs(0, 0)
    build_inputs = [full_build_inputs[i] for i in kept_var_idx]
    arg_specs = compact_arg_specs
    compiled_input_np_dtype = np.float32 if GRAPH_FP32 else np.float16
    compiled_input_g_dtype = Graph.FP32 if GRAPH_FP32 else Graph.FP16

    compiled_input_tensors = {}
    compiled_kv_cache_tensors = []

    for i, arr in enumerate(build_inputs):
        key = f"%arg{i}"
        spec = arg_specs.get(key, {"kind": "runtime"})

        if spec.get("kind") == "runtime":
            arr2 = arr.astype(compiled_input_np_dtype)
            t = g.input(arr2.shape, compiled_input_g_dtype)
            g.set_input(t, np.ascontiguousarray(arr2))
            compiled_input_tensors[spec.get("name", key)] = t
        else:
            weight_path = _materialize_weight_file(spec.get("original_arg_key", key), spec, arr)
            t = g.mmap_weights(str(weight_path))

        input_map[key] = t
        input_shapes[key] = arr.shape

    for li in range(NUM_LAYERS):
        k_t = g.input((1, MAX_SEQ_LEN, NUM_HEADS, HEAD_DIM), Graph.FP16)
        v_t = g.input((1, MAX_SEQ_LEN, NUM_HEADS, HEAD_DIM), Graph.FP16)
        g.set_input(k_t, np.ascontiguousarray(k_cache_buffers[li]))
        g.set_input(v_t, np.ascontiguousarray(v_cache_buffers[li]))
        compiled_kv_cache_tensors.append((k_t, v_t))

    compiled_mask_tensor = None
    if USE_ADDITIVE_KV_MASK:
        compiled_mask_tensor = g.input((1, NUM_HEADS, 1, MAX_SEQ_LEN + 1), Graph.FP16)
        g.set_input(compiled_mask_tensor, build_kv_additive_mask(cur_cache_len))
    compiled_fused_attn_out_names = []
    compiled_debug_taps = {} if DEBUG_NUMERICS else None

    compiled_env = selected_lower_to_cactus(
        nodes,
        g,
        input_map,
        input_shapes,
        raw_inputs=[],
        enable_attention_fusion=FUSION_KV_ACTIVE,
        enable_rmsnorm_fusion=ENABLE_RMSNORM_FUSION,
        use_kv_cache=FUSION_KV_ACTIVE,
        kv_cache_provider=None,
        kv_cache_tensors=compiled_kv_cache_tensors,
        kv_cache_len=cur_cache_len,
        kv_num_heads=NUM_HEADS,
        kv_head_dim=HEAD_DIM,
        kv_cache_additive_mask=compiled_mask_tensor,
        position_offset=cur_cache_len,
        force_dense_kv_attention=True,
        debug_taps=compiled_debug_taps,
        fp32_layernorm=FP32_LAYERNORM,
        fp32_qkv_matmul=FP32_QKV_MATMUL,
        post_matmul_fp32=POST_MATMUL_FP32,
        attention_scale=ATTN_SCALING,
        arg_specs=arg_specs,
        fused_attn_out_names=compiled_fused_attn_out_names,
    )
    if FUSION_KV_ACTIVE and len(compiled_fused_attn_out_names) != NUM_LAYERS:
        raise RuntimeError(
            f"Expected {NUM_LAYERS} fused attention outputs, got {len(compiled_fused_attn_out_names)}"
        )
    compiled_graph = g


def run_bridge_one_token(token_id, pos):
    global cache_len, compiled_input_np_dtype, compiled_graph, compiled_env
    if cache_len != pos:
        raise RuntimeError(f"cache_len mismatch: cache_len={cache_len}, pos={pos}")
    if cache_len >= MAX_SEQ_LEN:
        raise RuntimeError(f"cache_len {cache_len} reached MAX_SEQ_LEN {MAX_SEQ_LEN}")

    if RECOMPILE_PER_TOKEN:
        compiled_graph = None
        compiled_env = None
    if compiled_graph is None:
        init_compiled_graph(cache_len)

    g = compiled_graph
    env = compiled_env
    t_total0 = time.perf_counter()

    token_embed = embed[int(token_id):int(token_id) + 1].astype(np.float32)
    pos_embed = np.zeros((1, D_MODEL), dtype=np.float32)
    rope_cos_local, rope_sin_local = rope_cos_sin_np(int(pos), HEAD_DIM, LOCAL_ROPE_THETA)
    rope_cos_global, rope_sin_global = rope_cos_sin_np(int(pos), HEAD_DIM, GLOBAL_ROPE_THETA)

    t_set0 = time.perf_counter()
    if "token_embed" in compiled_input_tensors:
        g.set_input(
            compiled_input_tensors["token_embed"],
            np.ascontiguousarray(token_embed.astype(compiled_input_np_dtype)),
        )
    if "pos_embed" in compiled_input_tensors:
        g.set_input(
            compiled_input_tensors["pos_embed"],
            np.ascontiguousarray(pos_embed.astype(compiled_input_np_dtype)),
        )
    if "rope_cos_local" in compiled_input_tensors:
        g.set_input(
            compiled_input_tensors["rope_cos_local"],
            np.ascontiguousarray(rope_cos_local.astype(compiled_input_np_dtype)),
        )
    if "rope_sin_local" in compiled_input_tensors:
        g.set_input(
            compiled_input_tensors["rope_sin_local"],
            np.ascontiguousarray(rope_sin_local.astype(compiled_input_np_dtype)),
        )
    if "rope_cos_global" in compiled_input_tensors:
        g.set_input(
            compiled_input_tensors["rope_cos_global"],
            np.ascontiguousarray(rope_cos_global.astype(compiled_input_np_dtype)),
        )
    if "rope_sin_global" in compiled_input_tensors:
        g.set_input(
            compiled_input_tensors["rope_sin_global"],
            np.ascontiguousarray(rope_sin_global.astype(compiled_input_np_dtype)),
        )

    for li, (k_t, v_t) in enumerate(compiled_kv_cache_tensors):
        g.set_input(k_t, np.ascontiguousarray(k_cache_buffers[li]))
        g.set_input(v_t, np.ascontiguousarray(v_cache_buffers[li]))
    if compiled_mask_tensor is not None:
        g.set_input(compiled_mask_tensor, build_kv_additive_mask(cache_len))
    t_set = time.perf_counter() - t_set0

    t0 = time.perf_counter()
    g.execute()
    t_exec = time.perf_counter() - t0

    t_post0 = time.perf_counter()
    raw_out = env[return_name].numpy().astype(np.float32).reshape(-1)

    if raw_out.shape[0] == D_MODEL:
        # Graph returned hidden state. Apply final LM projection in FP32 outside Cactus.
        hidden = raw_out
        out = hidden @ lm_head.astype(np.float32)
    elif raw_out.shape[0] == VOCAB:
        # Graph returned logits directly.
        out = raw_out
    else:
        raise RuntimeError(
            f"Unexpected bridge output shape: got {raw_out.shape}, "
            f"expected hidden D_MODEL={D_MODEL} or logits VOCAB={VOCAB}"
        )
    if not np.all(np.isfinite(out)):
        first_bad_node = None
        if DEBUG_NUMERICS:
            for idx, n in enumerate(nodes):
                t = env.get(n.name)
                if t is None:
                    continue
                a = t.numpy().astype(np.float32)
                if not np.all(np.isfinite(a)):
                    first_bad_node = (idx, n.name, n.op, int(np.isnan(a).sum()), int(np.isinf(a).sum()))
                    break
            if first_bad_node is not None:
                bi = first_bad_node[0]
                lo = max(0, bi - 10)
                hi = min(len(nodes), bi + 11)
                print(f"[first_bad_node_context] idx={bi} name={first_bad_node[1]} op={first_bad_node[2]}")
                for j in range(lo, hi):
                    n = nodes[j]
                    t = env.get(n.name)
                    if t is None:
                        print(f"  {j:04d} {n.name} {n.op} (missing in env)")
                        continue
                    a = t.numpy().astype(np.float32)
                    n_nan_j = int(np.isnan(a).sum())
                    n_inf_j = int(np.isinf(a).sum())
                    finite = a[np.isfinite(a)]
                    vmin = float(finite.min()) if finite.size else float('nan')
                    vmax = float(finite.max()) if finite.size else float('nan')
                    print(
                        f"  {j:04d} {n.name} {n.op} shape={a.shape} "
                        f"nan={n_nan_j} inf={n_inf_j} min={vmin:.5f} max={vmax:.5f}"
                    )
        bad_tap = None
        if DEBUG_NUMERICS and compiled_debug_taps:
            for name, t in compiled_debug_taps.items():
                a = t.numpy().astype(np.float32)
                if not np.all(np.isfinite(a)):
                    n_nan_t = int(np.isnan(a).sum())
                    n_inf_t = int(np.isinf(a).sum())
                    bad_tap = (name, n_nan_t, n_inf_t)
                    break
        n_nan = int(np.isnan(out).sum())
        n_inf = int(np.isinf(out).sum())
        bad_idx = np.where(~np.isfinite(out))[0]
        first_bad = int(bad_idx[0]) if bad_idx.size else -1
        finite = out[np.isfinite(out)]
        fmin = float(finite.min()) if finite.size else float("nan")
        fmax = float(finite.max()) if finite.size else float("nan")
        if SANITIZE_LOGITS:
            out = np.nan_to_num(out, nan=0.0, posinf=80.0, neginf=-80.0)
        else:
            tap_msg = ""
            if bad_tap is not None:
                tap_msg = f", first_bad_tap={bad_tap[0]}(nan={bad_tap[1]},inf={bad_tap[2]})"
            node_msg = ""
            if first_bad_node is not None:
                node_msg = (
                    f", first_bad_node={first_bad_node[1]}@{first_bad_node[0]}"
                    f"/{first_bad_node[2]}(nan={first_bad_node[3]},inf={first_bad_node[4]})"
                )
            raise RuntimeError(
                f"Bridge produced non-finite logits at pos={pos}: "
                f"nan={n_nan}, inf={n_inf}, first_bad_index={first_bad}, "
                f"finite_min={fmin:.5f}, finite_max={fmax:.5f}{tap_msg}{node_msg}"
            )
    if FUSION_KV_ACTIVE:
        for li, out_name in enumerate(compiled_fused_attn_out_names):
            k_new = env[out_name + "_k"].numpy().astype(np.float16).reshape(1, 1, NUM_HEADS, HEAD_DIM)
            v_new = env[out_name + "_v"].numpy().astype(np.float16).reshape(1, 1, NUM_HEADS, HEAD_DIM)
            k_cache_buffers[li][:, cache_len:cache_len + 1, :, :] = k_new
            v_cache_buffers[li][:, cache_len:cache_len + 1, :, :] = v_new

    cache_len += 1
    t_post = time.perf_counter() - t_post0
    t_total = time.perf_counter() - t_total0
    timing = {
        "set_input": float(t_set),
        "execute": float(t_exec),
        "post": float(t_post),
        "total": float(t_total),
    }
    return out, timing


print("Prefilling bridge cache...")
prefill_bridge_times = []
prefill_set_times = []
prefill_exec_times = []
prefill_post_times = []
for pos, tok in enumerate(cur_tokens):
    last_logits, t = run_bridge_one_token(int(tok), pos)
    prefill_bridge_times.append(t["total"])
    prefill_set_times.append(t["set_input"])
    prefill_exec_times.append(t["execute"])
    prefill_post_times.append(t["post"])

if COMPARE_PREFILL:
    ref_prefill = hf_logits(cur_tokens)
    compare_logits("prefill", ref_prefill, last_logits)
    if COMPARE_JAX_SINGLE and len(cur_tokens) == 1:
        jax_inputs = [jnp.array(x, dtype=jnp.float32) for x in build_jax_inputs(cur_tokens[0], 0)]
        jax_hidden = np.array(jit_decode(*jax_inputs), dtype=np.float32).reshape(-1)
        if jax_hidden.shape[0] == D_MODEL:
            jax_out = jax_hidden @ lm_head.astype(np.float32)
        else:
            jax_out = jax_hidden
        compare_logits("jax_one_token_vs_hf", ref_prefill, jax_out)
        compare_logits("bridge_vs_jax_one_token", jax_out, last_logits)
        if COMPARE_JAX_FP16_SINGLE:
            jax_inputs16 = [jnp.array(x, dtype=jnp.float16) for x in build_jax_inputs(cur_tokens[0], 0)]
            jax_hidden16 = np.array(jit_decode(*jax_inputs16), dtype=np.float32).reshape(-1)
            if jax_hidden16.shape[0] == D_MODEL:
                jax_out16 = jax_hidden16 @ lm_head.astype(np.float32)
            else:
                jax_out16 = jax_hidden16
            compare_logits("jax_fp16_one_token_vs_hf", ref_prefill, jax_out16)
            compare_logits("bridge_vs_jax_fp16_one_token", jax_out16, last_logits)

jax_times = []
bridge_times = []
bridge_set_times = []
bridge_exec_times = []
bridge_post_times = []
top5_fracs = []

for step in range(GEN_STEPS):
    next_tok = sample_logits(last_logits, cur_tokens)
    cur_tokens.append(next_tok)
    pos = len(cur_tokens) - 1

    ref_logits = None
    hf_t = None
    if not SKIP_REF_COMPARE:
        t0 = time.perf_counter()
        ref_logits = hf_logits(cur_tokens)
        hf_t = time.perf_counter() - t0
        jax_times.append(hf_t)

    out, bridge_t = run_bridge_one_token(next_tok, pos)
    bridge_times.append(bridge_t["total"])
    bridge_set_times.append(bridge_t["set_input"])
    bridge_exec_times.append(bridge_t["execute"])
    bridge_post_times.append(bridge_t["post"])
    last_logits = out

    if ref_logits is not None:
        top5 = compare_logits(f"decode s={step}", ref_logits, out)
        top5_fracs.append(top5)
        if BENCH_DETAIL:
            print(
                f"  times: hf_full={hf_t:.4f}s bridge_total={bridge_t['total']:.4f}s "
                f"(set={bridge_t['set_input']:.4f}s exec={bridge_t['execute']:.4f}s post={bridge_t['post']:.4f}s)"
            )
        else:
            print(f"  times: hf_full={hf_t:.4f}s bridge={bridge_t['total']:.4f}s")
    else:
        if BENCH_DETAIL:
            print(
                f"decode s={step}: bridge_total={bridge_t['total']:.4f}s "
                f"(set={bridge_t['set_input']:.4f}s exec={bridge_t['execute']:.4f}s post={bridge_t['post']:.4f}s)"
            )
        else:
            print(f"decode s={step}: bridge={bridge_t['total']:.4f}s")

print("\n=== Summary ===")
if bridge_times:
    bridge_avg = float(np.mean(bridge_times))
    if prefill_bridge_times:
        prefill_total = float(np.sum(prefill_bridge_times))
        prefill_tokens = len(prefill_bridge_times)
        prefill_tps = (prefill_tokens / prefill_total) if prefill_total > 0 else float("inf")
        print(
            f"prefill: tokens={prefill_tokens} total={prefill_total:.4f}s "
            f"tps={prefill_tps:.4f} tok/s"
        )
        if BENCH_DETAIL:
            print(
                f"prefill breakdown avg: total={float(np.mean(prefill_bridge_times)):.4f}s "
                f"set={float(np.mean(prefill_set_times)):.4f}s "
                f"exec={float(np.mean(prefill_exec_times)):.4f}s "
                f"post={float(np.mean(prefill_post_times)):.4f}s"
            )
    if jax_times:
        hf_avg = float(np.mean(jax_times))
        decode_tokens = len(bridge_times)
        decode_total = float(np.sum(bridge_times))
        decode_tps = (decode_tokens / decode_total) if decode_total > 0 else float("inf")
        hf_decode_total = float(np.sum(jax_times))
        hf_decode_tps = (len(jax_times) / hf_decode_total) if hf_decode_total > 0 else float("inf")
        print(f"hf_full avg={hf_avg:.4f}s")
        print(f"bridge avg={bridge_avg:.4f}s")
        print(f"hf_full decode tps={hf_decode_tps:.4f} tok/s")
        print(f"bridge decode tps={decode_tps:.4f} tok/s")
        print(f"speedup={hf_avg / bridge_avg:.2f}x")
        if BENCH_DETAIL:
            print(
                f"decode breakdown avg: total={bridge_avg:.4f}s "
                f"set={float(np.mean(bridge_set_times)):.4f}s "
                f"exec={float(np.mean(bridge_exec_times)):.4f}s "
                f"post={float(np.mean(bridge_post_times)):.4f}s"
            )
        if top5_fracs:
            print(f"top5 average={float(np.mean(top5_fracs)):.4f}")
    else:
        decode_tokens = len(bridge_times)
        decode_total = float(np.sum(bridge_times))
        decode_tps = (decode_tokens / decode_total) if decode_total > 0 else float("inf")
        print(f"bridge avg={bridge_avg:.4f}s")
        print(f"bridge decode tps={decode_tps:.4f} tok/s")
        if BENCH_DETAIL:
            print(
                f"decode breakdown avg: total={bridge_avg:.4f}s "
                f"set={float(np.mean(bridge_set_times)):.4f}s "
                f"exec={float(np.mean(bridge_exec_times)):.4f}s "
                f"post={float(np.mean(bridge_post_times)):.4f}s"
            )
print("Final text:", repr(tokenizer.decode(cur_tokens)))
