compiled_graph = None
compiled_env = None
USE_CACTUS_SAMPLING = True

import re
import time
import os
import sys
import resource
import numpy as np
import jax
import jax.numpy as jnp
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from src.graph import Graph
global_graph = Graph()

from src.IR.stablehlo_ir import parse_stablehlo_ops
from src.IR.lower_to_cactus import lower_to_cactus, find_attention_blocks


PROMPT = "In a surprising discovery, scientists found that"
GEN_STEPS = int(os.getenv("GEN_STEPS", "8"))

KV_QUANT_GROUP_SIZE = 32
FORCE_DENSE_KV_ATTENTION = os.getenv("FORCE_DENSE_KV_ATTENTION", "1") == "1"
PREFILL_PROMPT_CACHE = os.getenv("PREFILL_PROMPT_CACHE", "1") == "1"
COMPARE_MANUAL_KV = os.getenv("COMPARE_MANUAL_KV", "1") == "1"
CHECK_LAYER_PERM = os.getenv("CHECK_LAYER_PERM", "0") == "1"
BRIDGE_FP32_INPUTS = os.getenv("BRIDGE_FP32_INPUTS", "0") == "1"
TRACE_LAYER_DIFF = os.getenv("TRACE_LAYER_DIFF", "1") == "1"
FP32_LAYERNORM = os.getenv("FP32_LAYERNORM", "1") == "1"
FP32_QKV_MATMUL = os.getenv("FP32_QKV_MATMUL", "0") == "1"
CACHE_CHECK = os.getenv("CACHE_CHECK", "0") == "1"

def sample_logits(logits, generated, temperature=0.8, top_k=50, top_p=0.9, repetition_penalty=1.2):
    logits = logits.astype(np.float32).copy()

    for tok in generated[-50:]:
        if logits[tok] > 0:
            logits[tok] /= repetition_penalty
        else:
            logits[tok] *= repetition_penalty

    logits = logits / temperature

    top_idx = np.argpartition(logits, -top_k)[-top_k:]
    top_logits = logits[top_idx]

    probs = np.exp(top_logits - np.max(top_logits))
    probs /= probs.sum()

    sorted_order = np.argsort(-probs)
    sorted_probs = probs[sorted_order]
    sorted_idx = top_idx[sorted_order]

    cumulative = np.cumsum(sorted_probs)
    keep = cumulative <= top_p
    keep[0] = True

    final_idx = sorted_idx[keep]
    final_probs = sorted_probs[keep]
    final_probs /= final_probs.sum()

    return int(np.random.choice(final_idx, p=final_probs))


def sample_top_k(logits, k=40, temperature=0.8):
    logits = logits.astype(np.float32) / temperature
    top_idx = np.argpartition(logits, -k)[-k:]
    top_logits = logits[top_idx]
    probs = np.exp(top_logits - np.max(top_logits))
    probs /= probs.sum()
    return int(np.random.choice(top_idx, p=probs))

def extract_return_name(ir):
    for line in ir.splitlines():
        line = line.strip()
        if line.startswith("return "):
            m = re.search(r"return\s+(%[\w\d_]+)", line)
            if m:
                return m.group(1)
            raise RuntimeError(f"Bad return line: {line}")
    raise RuntimeError("No return found")


hf_model = GPT2LMHeadModel.from_pretrained("GPT2-large")
hf_model.eval()
config = hf_model.config

NUM_LAYERS = config.n_layer
NUM_HEADS  = config.n_head
HEAD_DIM   = config.n_embd // config.n_head
MAX_SEQ_LEN = int(os.getenv("MAX_SEQ_LEN", "0"))
tokenizer = GPT2Tokenizer.from_pretrained("GPT2-large")
state = hf_model.state_dict()

wte = state["transformer.wte.weight"].cpu().numpy().astype(np.float32)
wpe = state["transformer.wpe.weight"].cpu().numpy().astype(np.float32)
D_MODEL = wte.shape[1]





blocks = []
for i in range(NUM_LAYERS):
    ln1_g = state[f"transformer.h.{i}.ln_1.weight"].cpu().numpy().astype(np.float32)
    ln1_b = state[f"transformer.h.{i}.ln_1.bias"].cpu().numpy().astype(np.float32)

    c_attn = state[f"transformer.h.{i}.attn.c_attn.weight"].cpu().numpy().astype(np.float32)
    wq = np.ascontiguousarray(c_attn[:, :D_MODEL])
    wk = np.ascontiguousarray(c_attn[:, D_MODEL:2 * D_MODEL])
    wv = np.ascontiguousarray(c_attn[:, 2 * D_MODEL:3 * D_MODEL])
    wo = np.ascontiguousarray(
        state[f"transformer.h.{i}.attn.c_proj.weight"].cpu().numpy().astype(np.float32)
    )

    ln2_g = state[f"transformer.h.{i}.ln_2.weight"].cpu().numpy().astype(np.float32)
    ln2_b = state[f"transformer.h.{i}.ln_2.bias"].cpu().numpy().astype(np.float32)
    w1 = np.ascontiguousarray(state[f"transformer.h.{i}.mlp.c_fc.weight"].cpu().numpy().astype(np.float32))
    b1 = state[f"transformer.h.{i}.mlp.c_fc.bias"].cpu().numpy().astype(np.float32)
    w2 = np.ascontiguousarray(state[f"transformer.h.{i}.mlp.c_proj.weight"].cpu().numpy().astype(np.float32))
    b2 = state[f"transformer.h.{i}.mlp.c_proj.bias"].cpu().numpy().astype(np.float32)
    blocks.append((ln1_g, ln1_b, wq, wk, wv, wo, ln2_g, ln2_b, w1, b1, w2, b2))

ln_f_g = state["transformer.ln_f.weight"].cpu().numpy().astype(np.float32)
ln_f_b = state["transformer.ln_f.bias"].cpu().numpy().astype(np.float32)
lm_head = np.ascontiguousarray(wte.T.copy())


def gelu(x):
    return 0.5 * x * (1.0 + jnp.tanh(jnp.sqrt(2.0 / jnp.pi) * (x + 0.044715 * x * x * x)))


def layer_norm(x, g, b):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean((x - mean) ** 2, axis=-1, keepdims=True)
    return (x - mean) / jnp.sqrt(var + 1e-5) * g + b


def gelu_np(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x * x * x)))


def layer_norm_np(x, g, b):
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.mean((x - mean) ** 2, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + 1e-5) * g + b


def gpt2_one_token_graph(token_one_hot, pos_embed, wte, *params):
    block_params = params[:-3]
    lnf_g, lnf_b, lm = params[-3:]

    x = token_one_hot @ wte
    x = x + pos_embed

    for i in range(NUM_LAYERS):
        o = i * 12
        ln1_g, ln1_b, wq, wk, wv, wo, ln2_g, ln2_b, w1, b1, w2, b2 = block_params[o:o + 12]

        h = layer_norm(x, ln1_g, ln1_b)
        q = h @ wq
        k = h @ wk
        v = h @ wv

        scores = (q @ k.T) / jnp.sqrt(h.shape[-1])

        # ADD THIS
        causal = jnp.tril(jnp.ones_like(scores))
        scores = scores + (1.0 - causal) * -1e4

        probs = jax.nn.softmax(scores, axis=-1)
        attn_out = probs @ v

        x = x + attn_out @ wo
        h = layer_norm(x, ln2_g, ln2_b)
        h = gelu(h @ w1 + b1)
        x = x + (h @ w2 + b2)

    x = layer_norm(x, lnf_g, lnf_b)
    return x @ lm


def gpt2_full_jax(tokens):
    t = tokens.shape[0]
    x = wte[tokens] + wpe[:t]

    for i in range(NUM_LAYERS):
        ln1_g, ln1_b, wq, wk, wv, wo, ln2_g, ln2_b, w1, b1, w2, b2 = blocks[i]
        h = (x - x.mean(-1, keepdims=True)) / np.sqrt(((x - x.mean(-1, keepdims=True)) ** 2).mean(-1, keepdims=True) + 1e-5)
        h = h * ln1_g + ln1_b
        q = h @ wq
        k = h @ wk
        v = h @ wv
        q4 = q.reshape(t, NUM_HEADS, HEAD_DIM)
        k4 = k.reshape(t, NUM_HEADS, HEAD_DIM)
        v4 = v.reshape(t, NUM_HEADS, HEAD_DIM)
        scores = np.einsum("thd,shd->hts", q4, k4) / np.sqrt(float(HEAD_DIM))
        causal = np.tril(np.ones((t, t), dtype=np.float32))
        scores = scores + (1.0 - causal)[None, :, :] * -1e4
        probs = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        probs = probs / np.sum(probs, axis=-1, keepdims=True)
        attn = np.einsum("hts,shd->thd", probs, v4).reshape(t, D_MODEL)
        x = x + attn @ wo
        h = (x - x.mean(-1, keepdims=True)) / np.sqrt(((x - x.mean(-1, keepdims=True)) ** 2).mean(-1, keepdims=True) + 1e-5)
        h = h * ln2_g + ln2_b
        h = 0.5 * (h @ w1 + b1) * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * ((h @ w1 + b1) + 0.044715 * (h @ w1 + b1) ** 3)))
        x = x + (h @ w2 + b2)

    x = (x - x.mean(-1, keepdims=True)) / np.sqrt(((x - x.mean(-1, keepdims=True)) ** 2).mean(-1, keepdims=True) + 1e-5)
    x = x * ln_f_g + ln_f_b
    return x @ lm_head


flat = []
for b in blocks:
    flat.extend(b)


def build_jax_inputs(token_id, pos):
    token_one_hot = np.zeros((1, wte.shape[0]), dtype=np.float32)
    token_one_hot[0, int(token_id)] = 1.0
    pos_embed = wpe[int(pos):int(pos) + 1].astype(np.float32)
    return (
        token_one_hot,
        pos_embed,
        wte,
        *flat,
        ln_f_g,
        ln_f_b,
        lm_head,
    )


def quantize_kv(cache):
    cache = np.ascontiguousarray(cache, dtype=np.float32)
    if cache.shape[0] == 0:
        return np.zeros((0,), dtype=np.int8), np.zeros((0,), dtype=np.float32)
    seq_len, kv_heads, head_dim = cache.shape
    groups = (head_dim + KV_QUANT_GROUP_SIZE - 1) // KV_QUANT_GROUP_SIZE
    q = np.zeros_like(cache, dtype=np.int8)
    scales = np.zeros((seq_len, kv_heads, groups), dtype=np.float32)
    for t in range(seq_len):
        for h in range(kv_heads):
            for g in range(groups):
                s = g * KV_QUANT_GROUP_SIZE
                e = min(s + KV_QUANT_GROUP_SIZE, head_dim)
                blk = cache[t, h, s:e]
                mx = float(np.max(np.abs(blk))) if blk.size else 0.0
                sc = max(mx / 127.0, 1e-10)
                scales[t, h, g] = sc
                q[t, h, s:e] = np.clip(np.round(blk / sc), -128, 127).astype(np.int8)
    return np.ascontiguousarray(q.reshape(-1)), np.ascontiguousarray(scales.reshape(-1))


def cactus_attention_decode(q_new4, k_new4, v_new4, position, k_cache, v_cache):
    g = global_graph
    tq = g.input(q_new4.shape, Graph.FP16)
    tk = g.input(k_new4.shape, Graph.FP16)
    tv = g.input(v_new4.shape, Graph.FP16)
    g.set_input(tq, np.ascontiguousarray(q_new4.astype(np.float16)))
    g.set_input(tk, np.ascontiguousarray(k_new4.astype(np.float16)))
    g.set_input(tv, np.ascontiguousarray(v_new4.astype(np.float16)))

    cache_len = int(k_cache.shape[0])
    if cache_len == 0:
        out = g.attention(
            tq, tk, tv,
            1.0 / np.sqrt(HEAD_DIM),
            is_causal=True,
            position_offset=position,
        )
    else:
        ck, ks = quantize_kv(k_cache)
        cv, vs = quantize_kv(v_cache)
        out = g.attention_int8_hybrid(
            tq, tk, tv,
            1.0 / np.sqrt(HEAD_DIM),
            position,
            ck, cv, ks, vs,
            cache_len, NUM_HEADS, HEAD_DIM,
        )
    g.execute()
    return out.numpy().astype(np.float32)


def manual_decode_one_token(token_id, position, k_caches_m, v_caches_m):
    x = (wte[token_id] + wpe[position]).astype(np.float32)
    k_new_all = []
    v_new_all = []
    checkpoints = {}
    for i in range(NUM_LAYERS):
        ln1_g, ln1_b, wq, wk, wv, wo, ln2_g, ln2_b, w1, b1, w2, b2 = blocks[i]

        h = layer_norm_np(x, ln1_g, ln1_b)
        checkpoints[f"ln1_{i}"] = h.astype(np.float32)
        q_new = h @ wq
        k_new = h @ wk
        v_new = h @ wv

        q_new4 = q_new.reshape(1, 1, NUM_HEADS, HEAD_DIM)
        k_new4 = k_new.reshape(1, 1, NUM_HEADS, HEAD_DIM)
        v_new4 = v_new.reshape(1, 1, NUM_HEADS, HEAD_DIM)
        attn_new4 = cactus_attention_decode(
            q_new4, k_new4, v_new4, position, k_caches_m[i], v_caches_m[i]
        )
        attn_new = attn_new4.reshape(D_MODEL)
        checkpoints[f"attn_raw_{i}"] = attn_new.astype(np.float32)
        x = x + attn_new @ wo

        h2 = layer_norm_np(x, ln2_g, ln2_b)
        checkpoints[f"ln2_{i}"] = h2.astype(np.float32)
        h2 = gelu_np(h2 @ w1 + b1)
        checkpoints[f"gelu_{i}"] = h2.astype(np.float32)
        x = x + (h2 @ w2 + b2)

        k_new_all.append(k_new.reshape(NUM_HEADS, HEAD_DIM).astype(np.float32))
        v_new_all.append(v_new.reshape(NUM_HEADS, HEAD_DIM).astype(np.float32))

    x = layer_norm_np(x, ln_f_g, ln_f_b)
    logits = x @ lm_head
    return logits.astype(np.float32), k_new_all, v_new_all, checkpoints


def compare_logits(label, ref, out):
    ref = ref.astype(np.float32)
    out = out.astype(np.float32)
    diff = np.abs(ref - out)
    cos = float(np.dot(ref, out) / (np.linalg.norm(ref) * np.linalg.norm(out) + 1e-8))
    ref_top = np.argsort(ref)[-5:][::-1]
    out_top = np.argsort(out)[-5:][::-1]
    overlap = len(set(ref_top.tolist()) & set(out_top.tolist()))
    print(
        f"{label}: max={float(diff.max()):.5f} "
        f"mean={float(diff.mean()):.5f} cos={cos:.6f} top5={overlap}/5"
    )
    return {
        "max": float(diff.max()),
        "mean": float(diff.mean()),
        "cos": cos,
        "top5": overlap,
    }


def get_memory_mb():
    try:
        import psutil  # type: ignore
        proc = psutil.Process(os.getpid())
        return float(proc.memory_info().rss) / (1024.0 * 1024.0)
    except Exception:
        ru = resource.getrusage(resource.RUSAGE_SELF)
        if sys.platform == "darwin":
            return float(ru.ru_maxrss) / (1024.0 * 1024.0)
        return float(ru.ru_maxrss) / 1024.0


jit_decode = jax.jit(gpt2_one_token_graph)
sample = build_jax_inputs(0, 0)
sample_jax = [
    jnp.array(x, dtype=jnp.int32) if x.dtype == np.int32 else jnp.array(x)
    for x in sample
]
ir = str(jit_decode.lower(*sample_jax).compiler_ir(dialect="stablehlo"))
return_name = extract_return_name(ir)
nodes = parse_stablehlo_ops(ir)
attn_blocks = find_attention_blocks(nodes)
attn_out_names = [nodes[b["out"]].name for b in attn_blocks]

cur_tokens = tokenizer.encode(PROMPT)
if MAX_SEQ_LEN <= 0:
    # Keep static KV axis close to active decode range for better FP16 masking stability.
    MAX_SEQ_LEN = len(cur_tokens) + GEN_STEPS + 8
MAX_SEQ_LEN = min(MAX_SEQ_LEN, int(config.n_positions))
if MAX_SEQ_LEN <= len(cur_tokens):
    raise RuntimeError(
        f"MAX_SEQ_LEN must exceed prompt length: max_seq={MAX_SEQ_LEN}, prompt_len={len(cur_tokens)}"
    )

k_cache_buffers = [
    np.zeros((1, MAX_SEQ_LEN, NUM_HEADS, HEAD_DIM), dtype=np.float16)
    for _ in range(NUM_LAYERS)
]
v_cache_buffers = [
    np.zeros((1, MAX_SEQ_LEN, NUM_HEADS, HEAD_DIM), dtype=np.float16)
    for _ in range(NUM_LAYERS)
]
cache_len = 0

jax_times = []
bridge_times = []
bridge_set_times = []
bridge_exec_times = []
top5_fracs = []

compiled_input_tensors = {}
compiled_kv_cache_tensors = []
compiled_mask_tensor = None


def build_kv_additive_mask(cur_cache_len):
    # Shape must match attention logits [B, H, T, S].
    m = np.full((1, NUM_HEADS, 1, MAX_SEQ_LEN + 1), -1e4, dtype=np.float16)
    if cur_cache_len > 0:
        m[..., :cur_cache_len] = 0.0
    # Last slot corresponds to appended current-token K/V in cat([cache, k_new]).
    m[..., MAX_SEQ_LEN] = 0.0
    return m


def init_compiled_bridge_graph():
    global compiled_graph, compiled_env
    global compiled_input_tensors, compiled_kv_cache_tensors, compiled_mask_tensor

    if compiled_graph is not None:
        return

    g = Graph()
    input_map = {}
    input_shapes = {}
    compiled_input_tensors = {}

    # Build once with representative inputs; token/position are overwritten per step.
    build_inputs = build_jax_inputs(0, 0)
    for i, arr in enumerate(build_inputs):
        if arr.dtype == np.int32:
            arr2 = arr.astype(np.float32)
            t = g.input(arr2.shape, Graph.FP32)
        else:
            if BRIDGE_FP32_INPUTS:
                arr2 = arr.astype(np.float32)
                t = g.input(arr2.shape, Graph.FP32)
            else:
                arr2 = arr.astype(np.float16)
                t = g.input(arr2.shape, Graph.FP16)

        g.set_input(t, np.ascontiguousarray(arr2))
        key = f"%arg{i}"
        input_map[key] = t
        input_shapes[key] = arr2.shape
        compiled_input_tensors[key] = t

    compiled_kv_cache_tensors = []
    for li in range(NUM_LAYERS):
        k_cache_t = g.input((1, MAX_SEQ_LEN, NUM_HEADS, HEAD_DIM), Graph.FP16)
        v_cache_t = g.input((1, MAX_SEQ_LEN, NUM_HEADS, HEAD_DIM), Graph.FP16)
        g.set_input(k_cache_t, np.ascontiguousarray(k_cache_buffers[li]))
        g.set_input(v_cache_t, np.ascontiguousarray(v_cache_buffers[li]))
        compiled_kv_cache_tensors.append((k_cache_t, v_cache_t))

    compiled_mask_tensor = g.input((1, NUM_HEADS, 1, MAX_SEQ_LEN + 1), Graph.FP16)
    g.set_input(compiled_mask_tensor, build_kv_additive_mask(0))

    compiled_env = lower_to_cactus(
        nodes,
        g,
        input_map,
        input_shapes,
        # Compile-once graph must keep token/position dependent paths dynamic.
        raw_inputs=[],
        enable_attention_fusion=True,
        use_kv_cache=True,
        kv_cache_provider=None,
        kv_cache_tensors=compiled_kv_cache_tensors,
        kv_cache_len=0,
        kv_num_heads=NUM_HEADS,
        kv_head_dim=HEAD_DIM,
        kv_cache_additive_mask=compiled_mask_tensor,
        position_offset=0,
        force_dense_kv_attention=True,
        debug_taps=None,
        fp32_layernorm=FP32_LAYERNORM,
        fp32_qkv_matmul=FP32_QKV_MATMUL,
    )
    compiled_graph = g


def run_bridge_one_token(token_id, pos):
    global cache_len
    global compiled_graph, compiled_env

    if cache_len != pos:
        raise RuntimeError(f"cache_len mismatch: cache_len={cache_len}, pos={pos}")
    if cache_len >= MAX_SEQ_LEN:
        raise RuntimeError(f"cache_len {cache_len} reached MAX_SEQ_LEN {MAX_SEQ_LEN}")

    if compiled_graph is None:
        init_compiled_bridge_graph()

    g = compiled_graph
    env = compiled_env

    t_set0 = time.perf_counter()

    # Dynamic per-token inputs.
    token_one_hot = np.zeros((1, wte.shape[0]), dtype=np.float32)
    token_one_hot[0, int(token_id)] = 1.0
    pos_embed = wpe[int(pos):int(pos) + 1].astype(np.float32)
    g.set_input(compiled_input_tensors["%arg0"], np.ascontiguousarray(token_one_hot))
    g.set_input(compiled_input_tensors["%arg1"], np.ascontiguousarray(pos_embed))

    # Graph-managed cache tensors (static shape, runtime-updated content).
    for li, (k_cache_t, v_cache_t) in enumerate(compiled_kv_cache_tensors):
        g.set_input(k_cache_t, np.ascontiguousarray(k_cache_buffers[li]))
        g.set_input(v_cache_t, np.ascontiguousarray(v_cache_buffers[li]))
    g.set_input(compiled_mask_tensor, build_kv_additive_mask(cache_len))
    set_elapsed = time.perf_counter() - t_set0

    if CACHE_CHECK:
        print(
            "CACHE CHECK",
            f"layer=0",
            f"cache_len={cache_len}",
            f"k_buf_shape={k_cache_buffers[0].shape}",
            f"v_buf_shape={v_cache_buffers[0].shape}",
        )

    t_exec0 = time.perf_counter()
    g.execute()
    exec_elapsed = time.perf_counter() - t_exec0
    total_elapsed = set_elapsed + exec_elapsed

    out = env[return_name].numpy().astype(np.float32)[0]

    k_new_bridge = []
    v_new_bridge = []

    for li, out_name in enumerate(attn_out_names):
        k_new4 = env[out_name + "_k"].numpy().astype(np.float16)
        v_new4 = env[out_name + "_v"].numpy().astype(np.float16)

        k_new = k_new4.reshape(1, 1, NUM_HEADS, HEAD_DIM)
        v_new = v_new4.reshape(1, 1, NUM_HEADS, HEAD_DIM)

        k_new_bridge.append(k_new[0, 0].astype(np.float32))
        v_new_bridge.append(v_new[0, 0].astype(np.float32))

        k_cache_buffers[li][:, cache_len:cache_len + 1, :, :] = k_new
        v_cache_buffers[li][:, cache_len:cache_len + 1, :, :] = v_new

    cache_len += 1

    return out, total_elapsed, set_elapsed, exec_elapsed, k_new_bridge, v_new_bridge, {}


if PREFILL_PROMPT_CACHE:
    manual_k_caches = [np.zeros((0, NUM_HEADS, HEAD_DIM), dtype=np.float32) for _ in range(NUM_LAYERS)]
    manual_v_caches = [np.zeros((0, NUM_HEADS, HEAD_DIM), dtype=np.float32) for _ in range(NUM_LAYERS)]
    for pos, tok in enumerate(cur_tokens):
        _, _, _, _, bk_all, bv_all, _ = run_bridge_one_token(int(tok), pos)
        if COMPARE_MANUAL_KV:
            _, mk_all, mv_all, _ = manual_decode_one_token(int(tok), pos, manual_k_caches, manual_v_caches)
            for li in range(NUM_LAYERS):
                manual_k_caches[li] = np.concatenate([manual_k_caches[li], mk_all[li][None, ...]], axis=0)
                manual_v_caches[li] = np.concatenate([manual_v_caches[li], mv_all[li][None, ...]], axis=0)
else:
    manual_k_caches = [np.zeros((0, NUM_HEADS, HEAD_DIM), dtype=np.float32) for _ in range(NUM_LAYERS)]
    manual_v_caches = [np.zeros((0, NUM_HEADS, HEAD_DIM), dtype=np.float32) for _ in range(NUM_LAYERS)]
decode_tokens = 0

# last prompt token was already processed during prefill,
# so use its bridge logits as the first next-token distribution
last_logits = None
if PREFILL_PROMPT_CACHE:
    # rerun last prompt token only for logits would duplicate cache,
    # so easiest safe path: use JAX for the first sample
    last_logits = gpt2_full_jax(np.array(cur_tokens, dtype=np.int32))[-1]

for step in range(GEN_STEPS):
    # Use bridge logits for sampling after prefill; JAX only if unavailable.
    if last_logits is None:
        base_logits = gpt2_full_jax(np.array(cur_tokens, dtype=np.int32))[-1]
    else:
        base_logits = last_logits

    next_tok = sample_logits(
        base_logits,
        cur_tokens,
        temperature=0.8,
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.2,
    )

    cur_tokens.append(next_tok)
    pos = len(cur_tokens) - 1

    # Single JAX reference timing, aligned to the same sequence as bridge output.
    t0 = time.perf_counter()
    ref_logits = gpt2_full_jax(np.array(cur_tokens, dtype=np.int32))[-1]
    jax_time = time.perf_counter() - t0
    jax_times.append(jax_time)

    out, bridge_elapsed, bridge_set, bridge_exec, *_ = run_bridge_one_token(next_tok, pos)
    bridge_times.append(bridge_elapsed)
    bridge_set_times.append(bridge_set)
    bridge_exec_times.append(bridge_exec)
    last_logits = out

    decode_tokens += 1

    cmp = compare_logits(f"decode s={step}", ref_logits, out)
    top5_fracs.append(float(cmp["top5"]) / 5.0)

    print(
        f"  times: jax_full={jax_time:.4f}s "
        f"bridge_total={bridge_elapsed:.4f}s "
        f"(set={bridge_set:.4f}s exec={bridge_exec:.4f}s)"
    )
    
print("\n=== BENCHMARK SUMMARY ===")

if len(jax_times) > 0:
    jax_avg = float(np.mean(jax_times))
    bridge_avg = float(np.mean(bridge_times))
    bridge_set_avg = float(np.mean(bridge_set_times)) if bridge_set_times else 0.0
    bridge_exec_avg = float(np.mean(bridge_exec_times)) if bridge_exec_times else 0.0
    decode_tps = (float(decode_tokens) / float(np.sum(bridge_times))) if bridge_times else 0.0
    top5_avg = float(np.mean(top5_fracs)) if top5_fracs else 0.0

    print(f"JAX avg: {jax_avg:.4f}s")
    print(f"Bridge total avg: {bridge_avg:.4f}s")
    print(f"set_input avg: {bridge_set_avg:.4f}s")
    print(f"execute avg: {bridge_exec_avg:.4f}s")

    print("\nTokens/sec:")
    print(f"JAX: {1.0 / jax_avg:.2f}")
    print(f"Bridge: {1.0 / bridge_avg:.2f}")
    print(f"Decode TPS avg: {decode_tps:.2f}")

    print("\nSpeedup:")
    print(f"{jax_avg / bridge_avg:.2f}x")
    print(f"Top5 average: {top5_avg:.4f} ({top5_avg * 100.0:.2f}%)")
    print(f"Final text: {repr(tokenizer.decode(cur_tokens))}")
    print(f"Memory (RSS): {get_memory_mb():.2f} MB")
else:
    print("No timing data collected.")
