import os
import time
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = os.getenv("MODEL_NAME", "google/gemma-2b")
PROMPT = os.getenv("PROMPT", "In a surprising discovery, scientists found that")
GEN_STEPS = int(os.getenv("GEN_STEPS", "8"))

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
INTERMEDIATE = config.intermediate_size
VOCAB = config.vocab_size
ROPE_THETA = getattr(config, "rope_theta", 10000.0)
RMS_EPS = config.rms_norm_eps

print("Layers:", NUM_LAYERS)
print("Heads:", NUM_HEADS)
print("KV heads:", NUM_KV_HEADS)
print("Head dim:", HEAD_DIM)
print("D model:", D_MODEL)


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


def gelu_pytorch_tanh(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x * x * x)))


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return np.concatenate([-x2, x1], axis=-1)


def rope_cos_sin(position, dim):
    inv_freq = 1.0 / (ROPE_THETA ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
    freqs = np.outer(np.array([position], dtype=np.float32), inv_freq)
    emb = np.concatenate([freqs, freqs], axis=-1)
    return np.cos(emb).astype(np.float32), np.sin(emb).astype(np.float32)


def apply_rope(x, position):
    # x: (heads, head_dim)
    cos, sin = rope_cos_sin(position, x.shape[-1])
    return (x * cos) + (rotate_half(x) * sin)


def repeat_kv(x):
    # x: (seq, kv_heads, head_dim) -> (seq, heads, head_dim)
    if NUM_KV_HEADS == NUM_HEADS:
        return x
    reps = NUM_HEADS // NUM_KV_HEADS
    return np.repeat(x, reps, axis=1)


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def manual_full_gemma(tokens):
    T = len(tokens)
    x = embed[tokens].astype(np.float32) * np.sqrt(float(D_MODEL))

    for li, (ln1, q_w, k_w, v_w, o_w, ln2, gate_w, up_w, down_w) in enumerate(blocks):
        h = rms_norm(x, ln1)

        q = h @ q_w.T
        k = h @ k_w.T
        v = h @ v_w.T

        q = q.reshape(T, NUM_HEADS, HEAD_DIM)
        k = k.reshape(T, NUM_KV_HEADS, HEAD_DIM)
        v = v.reshape(T, NUM_KV_HEADS, HEAD_DIM)

        for t in range(T):
            q[t] = apply_rope(q[t], t)
            k[t] = apply_rope(k[t], t)

        k_rep = repeat_kv(k)
        v_rep = repeat_kv(v)

        scores = np.einsum("thd,shd->hts", q, k_rep) / np.sqrt(float(HEAD_DIM))
        causal = np.tril(np.ones((T, T), dtype=np.float32))
        scores = scores + (1.0 - causal)[None, :, :] * -1e9

        probs = softmax(scores, axis=-1)
        attn = np.einsum("hts,shd->thd", probs, v_rep).reshape(T, D_MODEL)

        x = x + (attn @ o_w.T)

        h2 = rms_norm(x, ln2)
        gate = h2 @ gate_w.T
        up = h2 @ up_w.T

        # Gemma uses GELU approximate/tanh-style gated MLP in HF Gemma v1.
        mlp = gelu_pytorch_tanh(gate) * up
        x = x + (mlp @ down_w.T)

    x = rms_norm(x, final_norm_w)
    return x @ lm_head


def manual_decode_one_token(token_id, position, k_caches, v_caches):
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

        k_full = np.concatenate([k_caches[li], k[None, :, :]], axis=0)
        v_full = np.concatenate([v_caches[li], v[None, :, :]], axis=0)

        k_rep = repeat_kv(k_full)
        v_rep = repeat_kv(v_full)

        scores = np.einsum("hd,shd->hs", q, k_rep) / np.sqrt(float(HEAD_DIM))
        probs = softmax(scores, axis=-1)
        attn = np.einsum("hs,shd->hd", probs, v_rep).reshape(D_MODEL)

        x = x + (attn @ o_w.T)

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
        f"{label}: max={diff.max():.5f} mean={diff.mean():.5f} "
        f"cos={cos:.6f} top5={overlap}/5"
    )
    print("  ref top:", ref_top[:5])
    print("  out top:", out_top[:5])


def sample_logits(logits, generated, temperature=0.8, top_k=50, repetition_penalty=1.2):
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


tokens = tokenizer.encode(PROMPT)
print("Prompt:", repr(PROMPT))
print("Tokens:", tokens)

print("\n=== Full manual vs HF check ===")
ref = hf_logits(tokens)
manual = manual_full_gemma(tokens)[-1]
compare("full", ref, manual)

print("\n=== KV decode check ===")
k_caches = [np.zeros((0, NUM_KV_HEADS, HEAD_DIM), dtype=np.float32) for _ in range(NUM_LAYERS)]
v_caches = [np.zeros((0, NUM_KV_HEADS, HEAD_DIM), dtype=np.float32) for _ in range(NUM_LAYERS)]

for pos, tok in enumerate(tokens):
    logits, ks, vs = manual_decode_one_token(tok, pos, k_caches, v_caches)
    for li in range(NUM_LAYERS):
        k_caches[li] = np.concatenate([k_caches[li], ks[li][None, :, :]], axis=0)
        v_caches[li] = np.concatenate([v_caches[li], vs[li][None, :, :]], axis=0)

compare("prefill last token", ref, logits)

cur = list(tokens)
last_logits = logits

print("\n=== Generate ===")
for step in range(GEN_STEPS):
    next_tok = sample_logits(last_logits, cur)
    cur.append(next_tok)

    t0 = time.perf_counter()
    ref = hf_logits(cur)
    hf_t = time.perf_counter() - t0

    t1 = time.perf_counter()
    last_logits, ks, vs = manual_decode_one_token(next_tok, len(cur) - 1, k_caches, v_caches)
    manual_t = time.perf_counter() - t1

    for li in range(NUM_LAYERS):
        k_caches[li] = np.concatenate([k_caches[li], ks[li][None, :, :]], axis=0)
        v_caches[li] = np.concatenate([v_caches[li], vs[li][None, :, :]], axis=0)

    compare(f"decode s={step}", ref, last_logits)
    print(f"  times: hf_full={hf_t:.4f}s manual_kv={manual_t:.4f}s")
    print("  text:", tokenizer.decode(cur))