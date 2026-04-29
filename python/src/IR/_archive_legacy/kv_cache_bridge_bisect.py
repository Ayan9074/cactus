import re
import traceback
import os
import numpy as np
import jax
import jax.numpy as jnp
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from src.graph import Graph
from src.IR.stablehlo_ir import parse_stablehlo_ops
from src.IR.lower_to_cactus import lower_to_cactus, find_attention_blocks


PROMPT = "Hello world"
LAYER_STEPS = [1, 2, 4, 12]
NUM_HEADS = 12
KV_QUANT_GROUP_SIZE = 32
FORCE_DENSE_KV_ATTENTION = os.getenv("FORCE_DENSE_KV_ATTENTION", "1") == "1"


def extract_return_name(ir):
    for line in ir.splitlines():
        line = line.strip()
        if line.startswith("return "):
            m = re.search(r"return\s+(%[\w\d_]+)", line)
            if m:
                return m.group(1)
            raise RuntimeError(f"Bad return line: {line}")
    raise RuntimeError("No return found")


print("Loading GPT-2...")
hf_model = GPT2LMHeadModel.from_pretrained("gpt2")
hf_model.eval()
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
state = hf_model.state_dict()

wte = state["transformer.wte.weight"].cpu().numpy().astype(np.float32)
wpe = state["transformer.wpe.weight"].cpu().numpy().astype(np.float32)
D_MODEL = wte.shape[1]
HEAD_DIM = D_MODEL // NUM_HEADS

all_blocks = []
for i in range(12):
    ln1_g = state[f"transformer.h.{i}.ln_1.weight"].cpu().numpy().astype(np.float32)
    ln1_b = state[f"transformer.h.{i}.ln_1.bias"].cpu().numpy().astype(np.float32)
    c_attn = state[f"transformer.h.{i}.attn.c_attn.weight"].cpu().numpy().astype(np.float32)
    wq = np.ascontiguousarray(c_attn[:, :D_MODEL])
    wk = np.ascontiguousarray(c_attn[:, D_MODEL:2 * D_MODEL])
    wv = np.ascontiguousarray(c_attn[:, 2 * D_MODEL:3 * D_MODEL])
    wo = np.ascontiguousarray(state[f"transformer.h.{i}.attn.c_proj.weight"].cpu().numpy().astype(np.float32))
    ln2_g = state[f"transformer.h.{i}.ln_2.weight"].cpu().numpy().astype(np.float32)
    ln2_b = state[f"transformer.h.{i}.ln_2.bias"].cpu().numpy().astype(np.float32)
    w1 = np.ascontiguousarray(state[f"transformer.h.{i}.mlp.c_fc.weight"].cpu().numpy().astype(np.float32))
    b1 = state[f"transformer.h.{i}.mlp.c_fc.bias"].cpu().numpy().astype(np.float32)
    w2 = np.ascontiguousarray(state[f"transformer.h.{i}.mlp.c_proj.weight"].cpu().numpy().astype(np.float32))
    b2 = state[f"transformer.h.{i}.mlp.c_proj.bias"].cpu().numpy().astype(np.float32)
    all_blocks.append((ln1_g, ln1_b, wq, wk, wv, wo, ln2_g, ln2_b, w1, b1, w2, b2))

ln_f_g = state["transformer.ln_f.weight"].cpu().numpy().astype(np.float32)
ln_f_b = state["transformer.ln_f.bias"].cpu().numpy().astype(np.float32)
lm_head = np.ascontiguousarray(wte.T.copy())


def make_decode_fn(num_layers):
    def gelu(x):
        return 0.5 * x * (1.0 + jnp.tanh(jnp.sqrt(2.0 / jnp.pi) * (x + 0.044715 * x * x * x)))

    def layer_norm(x, g, b):
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.mean((x - mean) ** 2, axis=-1, keepdims=True)
        return (x - mean) / jnp.sqrt(var + 1e-5) * g + b

    def fn(token, pos, wte, wpe, *params):
        block_params = params[:-3]
        lnf_g, lnf_b, lm = params[-3:]

        one_hot = jax.nn.one_hot(token, wte.shape[0], dtype=wte.dtype)
        x = one_hot @ wte
        x = x + wpe[pos]

        for i in range(num_layers):
            o = i * 12
            ln1_g, ln1_b, wq, wk, wv, wo, ln2_g, ln2_b, w1, b1, w2, b2 = block_params[o:o + 12]
            h = layer_norm(x, ln1_g, ln1_b)
            q = h @ wq
            k = h @ wk
            v = h @ wv
            scores = (q @ k.T) / jnp.sqrt(h.shape[-1])
            probs = jax.nn.softmax(scores, axis=-1)
            attn_out = probs @ v
            x = x + attn_out @ wo
            h = layer_norm(x, ln2_g, ln2_b)
            h = gelu(h @ w1 + b1)
            x = x + (h @ w2 + b2)

        x = layer_norm(x, lnf_g, lnf_b)
        return x @ lm

    return fn


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


def compare_logits(ref, out):
    ref = ref.astype(np.float32)
    out = out.astype(np.float32)
    diff = np.abs(ref - out)
    cos = float(np.dot(ref, out) / (np.linalg.norm(ref) * np.linalg.norm(out) + 1e-8))
    ref_top = np.argsort(ref)[-5:][::-1]
    out_top = np.argsort(out)[-5:][::-1]
    overlap = len(set(ref_top.tolist()) & set(out_top.tolist()))
    return float(diff.max()), float(diff.mean()), cos, overlap


def full_logits_np(tokens, num_layers, blocks):
    t = tokens.shape[0]
    x = wte[tokens] + wpe[:t]
    for i in range(num_layers):
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
        p = h @ w1 + b1
        h = 0.5 * p * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (p + 0.044715 * p ** 3)))
        x = x + (h @ w2 + b2)

    x = (x - x.mean(-1, keepdims=True)) / np.sqrt(((x - x.mean(-1, keepdims=True)) ** 2).mean(-1, keepdims=True) + 1e-5)
    x = x * ln_f_g + ln_f_b
    return x @ lm_head


def run_level(num_layers):
    print(f"\n===== BISECT LEVEL: {num_layers} layer(s) =====")
    blocks = all_blocks[:num_layers]
    flat = []
    for b in blocks:
        flat.extend(b)

    fn = make_decode_fn(num_layers)
    jit_decode = jax.jit(fn)

    def build_inputs(token_id, pos):
        return (
            np.array([token_id], dtype=np.int32),
            np.array([pos], dtype=np.int32),
            wte, wpe,
            *flat,
            ln_f_g, ln_f_b, lm_head,
        )

    sample = build_inputs(0, 0)
    sample_jax = [jnp.array(x, dtype=jnp.int32) if x.dtype == np.int32 else jnp.array(x) for x in sample]
    ir = str(jit_decode.lower(*sample_jax).compiler_ir(dialect="stablehlo"))
    return_name = extract_return_name(ir)
    nodes = parse_stablehlo_ops(ir)
    attn_blocks = find_attention_blocks(nodes, verbose=False)
    attn_out_names = [nodes[b["out"]].name for b in attn_blocks]
    print("Detected attention blocks:", len(attn_out_names))
    if len(attn_out_names) != num_layers:
        raise RuntimeError(f"attention block count mismatch: expected {num_layers}, got {len(attn_out_names)}")

    k_caches = [np.zeros((0, NUM_HEADS, HEAD_DIM), dtype=np.float32) for _ in range(num_layers)]
    v_caches = [np.zeros((0, NUM_HEADS, HEAD_DIM), dtype=np.float32) for _ in range(num_layers)]

    prompt_tokens = tokenizer.encode(PROMPT)
    seq = prompt_tokens[:]

    # Run two bridge steps so second step has cache_len > 0.
    outs = []
    for step in range(2):
        token_id = seq[step] if step < len(seq) else int(seq[-1])
        pos = step
        inputs_np = build_inputs(token_id, pos)

        g = Graph()
        input_map = {}
        input_shapes = {}
        for i, arr in enumerate(inputs_np):
            if arr.dtype == np.int32:
                arr2 = arr.astype(np.float32)
                t = g.input(arr2.shape, Graph.FP32)
            else:
                arr2 = arr.astype(np.float16)
                t = g.input(arr2.shape, Graph.FP16)
            g.set_input(t, np.ascontiguousarray(arr2))
            input_map[f"%arg{i}"] = t
            input_shapes[f"%arg{i}"] = arr2.shape

        def cache_provider(layer_idx, _, __):
            ck, ks = quantize_kv(k_caches[layer_idx])
            cv, vs = quantize_kv(v_caches[layer_idx])
            return {
                "cached_keys": ck,
                "cached_values": cv,
                "k_scales": ks,
                "v_scales": vs,
                "cache_len": int(k_caches[layer_idx].shape[0]),
                "num_kv_heads": NUM_HEADS,
                "head_dim": HEAD_DIM,
            }

        env = lower_to_cactus(
            nodes,
            g,
            input_map,
            input_shapes,
            raw_inputs=inputs_np,
            enable_attention_fusion=True,
            use_kv_cache=True,
            kv_cache_provider=cache_provider,
            position_offset=pos,
            force_dense_kv_attention=FORCE_DENSE_KV_ATTENTION,
        )
        g.execute()
        out_step = env[return_name].numpy().astype(np.float32)[0]
        outs.append(out_step)
        print(f"step {step}: execute OK")

        for li, out_name in enumerate(attn_out_names):
            k_new4 = env[out_name + "_k"].numpy().astype(np.float32)
            v_new4 = env[out_name + "_v"].numpy().astype(np.float32)
            k_caches[li] = np.concatenate([k_caches[li], k_new4.reshape(1, NUM_HEADS, HEAD_DIM)], axis=0)
            v_caches[li] = np.concatenate([v_caches[li], v_new4.reshape(1, NUM_HEADS, HEAD_DIM)], axis=0)

    ref0 = full_logits_np(np.array(seq[:1], dtype=np.int32), num_layers, blocks)[-1]
    dmax0, dmean0, cos0, top50 = compare_logits(ref0, outs[0])
    print(
        f"quality ({num_layers}L @cache_len=0): "
        f"max={dmax0:.5f} mean={dmean0:.5f} cos={cos0:.6f} top5={top50}/5"
    )

    ref1 = full_logits_np(np.array(seq[:2], dtype=np.int32), num_layers, blocks)[-1]
    dmax, dmean, cos, top5 = compare_logits(ref1, outs[1])
    print(
        f"quality ({num_layers}L @cache_len=1): "
        f"max={dmax:.5f} mean={dmean:.5f} cos={cos:.6f} top5={top5}/5"
    )

    return True


def main():
    results = {}
    for n in LAYER_STEPS:
        try:
            run_level(n)
            results[n] = "PASS"
        except Exception as e:
            results[n] = f"FAIL: {type(e).__name__}: {e}"
            print(traceback.format_exc())
            break

    print("\n===== BISECT SUMMARY =====")
    for n in LAYER_STEPS:
        if n in results:
            print(f"{n:>2} layer(s): {results[n]}")
        else:
            print(f"{n:>2} layer(s): SKIPPED")


if __name__ == "__main__":
    main()
