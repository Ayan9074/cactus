import os
import time
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.graph import Graph


MODEL_NAME = os.getenv("MODEL_NAME", "google/gemma-2b")
PROMPT = os.getenv("PROMPT", "In a surprising discovery, scientists found that")
GEN_STEPS = int(os.getenv("GEN_STEPS", "8"))
TOP_K = int(os.getenv("TOP_K", "50"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.8"))
REPETITION_PENALTY = float(os.getenv("REPETITION_PENALTY", "1.2"))
KV_QUANT_GROUP_SIZE = int(os.getenv("KV_QUANT_GROUP_SIZE", "32"))
USE_HYBRID_KV = os.getenv("USE_HYBRID_KV", "1") == "1"


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def gelu_pytorch_tanh(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x * x * x)))


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return np.concatenate([-x2, x1], axis=-1)


def sample_logits(logits, generated, temperature=TEMPERATURE, top_k=TOP_K, repetition_penalty=REPETITION_PENALTY):
    logits = logits.astype(np.float32).copy()

    for tok in set(generated[-50:]):
        if logits[tok] > 0:
            logits[tok] /= repetition_penalty
        else:
            logits[tok] *= repetition_penalty

    logits = logits / temperature
    top_idx = np.argpartition(logits, -top_k)[-top_k:]
    top_logits = logits[top_idx]
    probs = np.exp(top_logits - np.max(top_logits))
    probs /= probs.sum()
    return int(np.random.choice(top_idx, p=probs))


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


print("Loading Gemma...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,
    local_files_only=True,
)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
config = model.config
state = model.state_dict()

NUM_LAYERS = config.num_hidden_layers
NUM_HEADS = config.num_attention_heads
NUM_KV_HEADS = config.num_key_value_heads
HEAD_DIM = config.head_dim
D_MODEL = config.hidden_size
ROPE_THETA = getattr(config, "rope_theta", 10000.0)
RMS_EPS = config.rms_norm_eps


def to_np(name):
    return state[name].detach().float().cpu().numpy()


embed = to_np("model.embed_tokens.weight")
final_norm_w = to_np("model.norm.weight")
lm_head = embed.T.copy()

blocks = []
for i in range(NUM_LAYERS):
    ln1 = to_np(f"model.layers.{i}.input_layernorm.weight")
    ln2 = to_np(f"model.layers.{i}.post_attention_layernorm.weight")

    q = to_np(f"model.layers.{i}.self_attn.q_proj.weight")
    k = to_np(f"model.layers.{i}.self_attn.k_proj.weight")
    v = to_np(f"model.layers.{i}.self_attn.v_proj.weight")
    o = to_np(f"model.layers.{i}.self_attn.o_proj.weight")

    gate = to_np(f"model.layers.{i}.mlp.gate_proj.weight")
    up = to_np(f"model.layers.{i}.mlp.up_proj.weight")
    down = to_np(f"model.layers.{i}.mlp.down_proj.weight")

    blocks.append((ln1, q, k, v, o, ln2, gate, up, down))


def rms_norm(x, w):
    x = x.astype(np.float32)
    var = np.mean(x * x, axis=-1, keepdims=True)
    return x * (1.0 / np.sqrt(var + RMS_EPS)) * (1.0 + w)


def rope_cos_sin(position, dim):
    inv_freq = 1.0 / (ROPE_THETA ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
    freqs = np.outer(np.array([position], dtype=np.float32), inv_freq)
    emb = np.concatenate([freqs, freqs], axis=-1)
    return np.cos(emb).astype(np.float32), np.sin(emb).astype(np.float32)


def apply_rope(x, position):
    cos, sin = rope_cos_sin(position, x.shape[-1])
    return (x * cos) + (rotate_half(x) * sin)


def cactus_attention_decode(q_new4, k_new4, v_new4, position, k_cache, v_cache):
    g = Graph()
    tq = g.input(q_new4.shape, Graph.FP16)
    tk = g.input(k_new4.shape, Graph.FP16)
    tv = g.input(v_new4.shape, Graph.FP16)
    g.set_input(tq, np.ascontiguousarray(q_new4.astype(np.float16)))
    g.set_input(tk, np.ascontiguousarray(k_new4.astype(np.float16)))
    g.set_input(tv, np.ascontiguousarray(v_new4.astype(np.float16)))

    scale = 1.0 / np.sqrt(float(HEAD_DIM))
    cache_len = int(k_cache.shape[0])

    if cache_len == 0:
        out = g.attention(
            tq,
            tk,
            tv,
            scale,
            is_causal=True,
            position_offset=position,
        )
    elif USE_HYBRID_KV:
        ck, ks = quantize_kv(k_cache)
        cv, vs = quantize_kv(v_cache)
        out = g.attention_int8_hybrid(
            tq,
            tk,
            tv,
            scale,
            position,
            ck,
            cv,
            ks,
            vs,
            cache_len,
            NUM_KV_HEADS,
            HEAD_DIM,
        )
    else:
        k_old4 = g.input((1, cache_len, NUM_KV_HEADS, HEAD_DIM), Graph.FP16)
        v_old4 = g.input((1, cache_len, NUM_KV_HEADS, HEAD_DIM), Graph.FP16)
        g.set_input(k_old4, np.ascontiguousarray(k_cache[None].astype(np.float16)))
        g.set_input(v_old4, np.ascontiguousarray(v_cache[None].astype(np.float16)))
        k_full4 = g.cat([k_old4, tk], axis=1)
        v_full4 = g.cat([v_old4, tv], axis=1)
        out = g.attention(
            tq,
            k_full4,
            v_full4,
            scale,
            is_causal=True,
            position_offset=position,
        )

    g.execute()
    return out.numpy().astype(np.float32)


def manual_decode_one_token_cactus(token_id, position, k_caches, v_caches):
    x = embed[int(token_id)].astype(np.float32) * np.sqrt(float(D_MODEL))
    new_ks = []
    new_vs = []

    for li, (ln1, q_w, k_w, v_w, o_w, ln2, gate_w, up_w, down_w) in enumerate(blocks):
        h = rms_norm(x[None, :], ln1)[0]

        q = h @ q_w.T
        k = h @ k_w.T
        v = h @ v_w.T

        q = q.reshape(NUM_HEADS, HEAD_DIM)
        k = k.reshape(NUM_KV_HEADS, HEAD_DIM)
        v = v.reshape(NUM_KV_HEADS, HEAD_DIM)

        q = apply_rope(q, position)
        k = apply_rope(k, position)

        q_new4 = q.reshape(1, 1, NUM_HEADS, HEAD_DIM)
        k_new4 = k.reshape(1, 1, NUM_KV_HEADS, HEAD_DIM)
        v_new4 = v.reshape(1, 1, NUM_KV_HEADS, HEAD_DIM)

        attn_new4 = cactus_attention_decode(
            q_new4,
            k_new4,
            v_new4,
            position,
            k_caches[li],
            v_caches[li],
        )

        attn_new = attn_new4.reshape(D_MODEL)
        x = x + (attn_new @ o_w.T)

        h2 = rms_norm(x[None, :], ln2)[0]
        gate = h2 @ gate_w.T
        up = h2 @ up_w.T
        mlp = gelu_pytorch_tanh(gate) * up
        x = x + (mlp @ down_w.T)

        new_ks.append(k.astype(np.float32))
        new_vs.append(v.astype(np.float32))

    x = rms_norm(x[None, :], final_norm_w)[0]
    logits = x @ lm_head
    return logits.astype(np.float32), new_ks, new_vs


def hf_logits(tokens):
    input_ids = torch.tensor([tokens], dtype=torch.long)
    with torch.no_grad():
        out = model(input_ids=input_ids)
    return out.logits[0, -1].detach().float().cpu().numpy()


def compare(label, ref, out):
    ref = ref.astype(np.float32)
    out = out.astype(np.float32)
    diff = np.abs(ref - out)
    cos = float(np.dot(ref, out) / (np.linalg.norm(ref) * np.linalg.norm(out) + 1e-8))
    ref_top = np.argsort(ref)[-5:][::-1]
    out_top = np.argsort(out)[-5:][::-1]
    overlap = len(set(ref_top.tolist()) & set(out_top.tolist()))
    print(
        f"{label}: max={float(diff.max()):.5f} mean={float(diff.mean()):.5f} "
        f"cos={cos:.6f} top5={overlap}/5"
    )
    return overlap


tokens = tokenizer.encode(PROMPT)
print("Prompt:", repr(PROMPT))
print("Tokens:", tokens)
print(f"USE_HYBRID_KV={USE_HYBRID_KV}")

k_caches = [np.zeros((0, NUM_KV_HEADS, HEAD_DIM), dtype=np.float32) for _ in range(NUM_LAYERS)]
v_caches = [np.zeros((0, NUM_KV_HEADS, HEAD_DIM), dtype=np.float32) for _ in range(NUM_LAYERS)]

print("\n=== Prefill ===")
for pos, tok in enumerate(tokens):
    logits, ks, vs = manual_decode_one_token_cactus(tok, pos, k_caches, v_caches)
    for li in range(NUM_LAYERS):
        k_caches[li] = np.concatenate([k_caches[li], ks[li][None, :, :]], axis=0)
        v_caches[li] = np.concatenate([v_caches[li], vs[li][None, :, :]], axis=0)

ref = hf_logits(tokens)
compare("prefill last token", ref, logits)

cur = list(tokens)
last_logits = logits
hf_times = []
bridge_times = []
top5_hits = []

print("\n=== Decode ===")
for step in range(GEN_STEPS):
    next_tok = sample_logits(last_logits, cur)
    cur.append(next_tok)

    t0 = time.perf_counter()
    ref = hf_logits(cur)
    hf_t = time.perf_counter() - t0
    hf_times.append(hf_t)

    t1 = time.perf_counter()
    last_logits, ks, vs = manual_decode_one_token_cactus(next_tok, len(cur) - 1, k_caches, v_caches)
    bridge_t = time.perf_counter() - t1
    bridge_times.append(bridge_t)

    for li in range(NUM_LAYERS):
        k_caches[li] = np.concatenate([k_caches[li], ks[li][None, :, :]], axis=0)
        v_caches[li] = np.concatenate([v_caches[li], vs[li][None, :, :]], axis=0)

    hit = compare(f"decode s={step}", ref, last_logits)
    top5_hits.append(float(hit) / 5.0)
    print(f"  times: hf_full={hf_t:.4f}s gemma_bridge={bridge_t:.4f}s")

print("\n=== Summary ===")
if hf_times and bridge_times:
    hf_avg = float(np.mean(hf_times))
    bridge_avg = float(np.mean(bridge_times))
    print(f"hf_full avg={hf_avg:.4f}s")
    print(f"gemma_bridge avg={bridge_avg:.4f}s")
    print(f"speedup={hf_avg / bridge_avg:.2f}x")
    print(f"top5 average={float(np.mean(top5_hits)):.4f}")
print("Final text:", repr(tokenizer.decode(cur)))
