import numpy as np
import jax
import jax.numpy as jnp
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from src.graph import Graph


PROMPT = "Hello world"
GEN_STEPS = 6
USE_KV_CACHE = True
LAYER_IDX = 0
NUM_HEADS = 12
KV_QUANT_GROUP_SIZE = 32


print("Loading GPT-2...")
hf_model = GPT2LMHeadModel.from_pretrained("gpt2")
hf_model.eval()
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
state = hf_model.state_dict()

wte = state["transformer.wte.weight"].cpu().numpy().astype(np.float32)
wpe = state["transformer.wpe.weight"].cpu().numpy().astype(np.float32)

ln1_g = state[f"transformer.h.{LAYER_IDX}.ln_1.weight"].cpu().numpy().astype(np.float32)
ln1_b = state[f"transformer.h.{LAYER_IDX}.ln_1.bias"].cpu().numpy().astype(np.float32)

c_attn = state[f"transformer.h.{LAYER_IDX}.attn.c_attn.weight"].cpu().numpy().astype(np.float32)
D_MODEL = c_attn.shape[0]
HEAD_DIM = D_MODEL // NUM_HEADS

wq = np.ascontiguousarray(c_attn[:, :D_MODEL])
wk = np.ascontiguousarray(c_attn[:, D_MODEL:2 * D_MODEL])
wv = np.ascontiguousarray(c_attn[:, 2 * D_MODEL:3 * D_MODEL])
wo = np.ascontiguousarray(
    state[f"transformer.h.{LAYER_IDX}.attn.c_proj.weight"].cpu().numpy().astype(np.float32)
)

ln2_g = state[f"transformer.h.{LAYER_IDX}.ln_2.weight"].cpu().numpy().astype(np.float32)
ln2_b = state[f"transformer.h.{LAYER_IDX}.ln_2.bias"].cpu().numpy().astype(np.float32)
w1 = np.ascontiguousarray(state[f"transformer.h.{LAYER_IDX}.mlp.c_fc.weight"].cpu().numpy().astype(np.float32))
b1 = state[f"transformer.h.{LAYER_IDX}.mlp.c_fc.bias"].cpu().numpy().astype(np.float32)
w2 = np.ascontiguousarray(state[f"transformer.h.{LAYER_IDX}.mlp.c_proj.weight"].cpu().numpy().astype(np.float32))
b2 = state[f"transformer.h.{LAYER_IDX}.mlp.c_proj.bias"].cpu().numpy().astype(np.float32)

ln_f_g = state["transformer.ln_f.weight"].cpu().numpy().astype(np.float32)
ln_f_b = state["transformer.ln_f.bias"].cpu().numpy().astype(np.float32)
lm_head = np.ascontiguousarray(wte.T.copy())


def gelu_np(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x * x * x)))


def layer_norm_np(x, g, b, eps=1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.mean((x - mean) ** 2, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * g + b


def one_layer_full_jax(tokens):
    tokens = jnp.asarray(tokens, dtype=jnp.int32)
    T = tokens.shape[0]
    pos = jnp.arange(T, dtype=jnp.int32)
    x = wte[tokens] + wpe[pos]

    h = layer_norm_np(np.array(x), ln1_g, ln1_b)
    h = jnp.asarray(h, dtype=jnp.float32)

    q = h @ jnp.asarray(wq)
    k = h @ jnp.asarray(wk)
    v = h @ jnp.asarray(wv)

    q4 = jnp.reshape(q, (T, NUM_HEADS, HEAD_DIM))
    k4 = jnp.reshape(k, (T, NUM_HEADS, HEAD_DIM))
    v4 = jnp.reshape(v, (T, NUM_HEADS, HEAD_DIM))

    scores = jnp.einsum("thd,shd->hts", q4, k4) / jnp.sqrt(jnp.array(float(HEAD_DIM), dtype=jnp.float32))
    causal = jnp.tril(jnp.ones((T, T), dtype=jnp.float32))
    scores = scores + (1.0 - causal)[None, :, :] * -1e4
    probs = jax.nn.softmax(scores, axis=-1)
    attn = jnp.einsum("hts,shd->thd", probs, v4).reshape(T, D_MODEL)

    x = jnp.asarray(np.array(x), dtype=jnp.float32) + attn @ jnp.asarray(wo)
    h2 = layer_norm_np(np.array(x), ln2_g, ln2_b)
    h2 = gelu_np(h2 @ w1 + b1)
    x = np.array(x) + (h2 @ w2 + b2)
    x = layer_norm_np(x, ln_f_g, ln_f_b)
    return x @ lm_head


def quantize_kv_fp32_to_int8(cache):
    cache = np.ascontiguousarray(cache, dtype=np.float32)
    if cache.shape[0] == 0:
        return np.zeros((0,), dtype=np.int8), np.zeros((0,), dtype=np.float32)

    seq_len, kv_heads, head_dim = cache.shape
    num_groups = (head_dim + KV_QUANT_GROUP_SIZE - 1) // KV_QUANT_GROUP_SIZE

    q = np.zeros_like(cache, dtype=np.int8)
    scales = np.zeros((seq_len, kv_heads, num_groups), dtype=np.float32)

    for t in range(seq_len):
        for h in range(kv_heads):
            for g in range(num_groups):
                s = g * KV_QUANT_GROUP_SIZE
                e = min(s + KV_QUANT_GROUP_SIZE, head_dim)
                block = cache[t, h, s:e]
                max_abs = float(np.max(np.abs(block))) if block.size else 0.0
                scale = max(max_abs / 127.0, 1e-10)
                scales[t, h, g] = scale
                q[t, h, s:e] = np.clip(np.round(block / scale), -128, 127).astype(np.int8)

    return np.ascontiguousarray(q.reshape(-1)), np.ascontiguousarray(scales.reshape(-1))


def cactus_attention_full(q4, k4, v4):
    g = Graph()
    tq = g.input(q4.shape, Graph.FP16)
    tk = g.input(k4.shape, Graph.FP16)
    tv = g.input(v4.shape, Graph.FP16)
    g.set_input(tq, np.ascontiguousarray(q4.astype(np.float16)))
    g.set_input(tk, np.ascontiguousarray(k4.astype(np.float16)))
    g.set_input(tv, np.ascontiguousarray(v4.astype(np.float16)))

    out = g.attention(tq, tk, tv, 1.0 / np.sqrt(HEAD_DIM), is_causal=True)
    g.execute()
    return out.numpy().astype(np.float32)


def cactus_attention_decode(q_new4, k_new4, v_new4, position, k_cache, v_cache):
    g = Graph()
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
        ck, ks = quantize_kv_fp32_to_int8(k_cache)
        cv, vs = quantize_kv_fp32_to_int8(v_cache)
        out = g.attention_int8_hybrid(
            tq, tk, tv,
            1.0 / np.sqrt(HEAD_DIM),
            position,
            ck, cv, ks, vs,
            cache_len, NUM_HEADS, HEAD_DIM,
        )
    g.execute()
    return out.numpy().astype(np.float32)


def one_layer_decode_token(token_id, position, k_cache, v_cache, use_kv_cache):
    x_new = wte[token_id] + wpe[position]
    h_new = layer_norm_np(x_new, ln1_g, ln1_b)

    q_new = h_new @ wq
    k_new = h_new @ wk
    v_new = h_new @ wv

    q_new4 = q_new.reshape(1, 1, NUM_HEADS, HEAD_DIM)
    k_new4 = k_new.reshape(1, 1, NUM_HEADS, HEAD_DIM)
    v_new4 = v_new.reshape(1, 1, NUM_HEADS, HEAD_DIM)

    if use_kv_cache:
        attn_new4 = cactus_attention_decode(q_new4, k_new4, v_new4, position, k_cache, v_cache)
    else:
        k_all4 = np.concatenate([k_cache, k_new.reshape(1, NUM_HEADS, HEAD_DIM)], axis=0).reshape(1, position + 1, NUM_HEADS, HEAD_DIM)
        v_all4 = np.concatenate([v_cache, v_new.reshape(1, NUM_HEADS, HEAD_DIM)], axis=0).reshape(1, position + 1, NUM_HEADS, HEAD_DIM)
        attn_new4 = cactus_attention_full(q_new4, k_all4, v_all4)

    attn_new = attn_new4.reshape(D_MODEL)

    x = x_new + attn_new @ wo
    h2 = layer_norm_np(x, ln2_g, ln2_b)
    h2 = gelu_np(h2 @ w1 + b1)
    x = x + (h2 @ w2 + b2)
    x = layer_norm_np(x, ln_f_g, ln_f_b)
    logits = x @ lm_head

    return logits.astype(np.float32), k_new.reshape(NUM_HEADS, HEAD_DIM), v_new.reshape(NUM_HEADS, HEAD_DIM)


def compare_logits(step_label, ref_logits, test_logits):
    ref = ref_logits.astype(np.float32)
    out = test_logits.astype(np.float32)
    diff = np.abs(ref - out)
    cos = float(np.dot(ref, out) / (np.linalg.norm(ref) * np.linalg.norm(out) + 1e-8))

    ref_top = np.argsort(ref)[-5:][::-1]
    out_top = np.argsort(out)[-5:][::-1]
    overlap = len(set(ref_top.tolist()) & set(out_top.tolist()))

    print(
        f"{step_label}: max={float(diff.max()):.5f} "
        f"mean={float(diff.mean()):.5f} cos={cos:.6f} top5={overlap}/5"
    )


def main():
    print("Mode:", "KV cache decode" if USE_KV_CACHE else "full attention (no cache)")
    prompt_tokens = tokenizer.encode(PROMPT)
    print("Prompt:", repr(PROMPT))
    print("Prompt tokens:", prompt_tokens)

    k_cache = np.zeros((0, NUM_HEADS, HEAD_DIM), dtype=np.float32)
    v_cache = np.zeros((0, NUM_HEADS, HEAD_DIM), dtype=np.float32)
    cur_tokens = []

    print("\nPrefill check (one-layer):")
    for tok in prompt_tokens:
        cur_tokens.append(int(tok))
        pos = len(cur_tokens) - 1

        ref_logits = np.array(one_layer_full_jax(np.array(cur_tokens, dtype=np.int32))[-1], dtype=np.float32)
        test_logits, k_new, v_new = one_layer_decode_token(
            int(tok), pos, k_cache, v_cache, use_kv_cache=USE_KV_CACHE
        )
        compare_logits(f"prefill t={pos}", ref_logits, test_logits)

        k_cache = np.concatenate([k_cache, k_new[None, ...]], axis=0)
        v_cache = np.concatenate([v_cache, v_new[None, ...]], axis=0)

    print("\nDecode check (generate using JAX full logits):")
    for step in range(GEN_STEPS):
        ref_next_logits = np.array(one_layer_full_jax(np.array(cur_tokens, dtype=np.int32))[-1], dtype=np.float32)
        next_token = int(np.argmax(ref_next_logits))
        cur_tokens.append(next_token)
        pos = len(cur_tokens) - 1

        ref_logits = np.array(one_layer_full_jax(np.array(cur_tokens, dtype=np.int32))[-1], dtype=np.float32)
        test_logits, k_new, v_new = one_layer_decode_token(
            next_token, pos, k_cache, v_cache, use_kv_cache=USE_KV_CACHE
        )
        compare_logits(f"decode s={step} pos={pos}", ref_logits, test_logits)

        k_cache = np.concatenate([k_cache, k_new[None, ...]], axis=0)
        v_cache = np.concatenate([v_cache, v_new[None, ...]], axis=0)

    text = tokenizer.decode(cur_tokens)
    print("\nFinal text:")
    print(repr(text))


if __name__ == "__main__":
    main()
