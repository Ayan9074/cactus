import re
import time
import statistics
import jax
import jax.numpy as jnp
import numpy as np

from transformers import GPT2LMHeadModel, GPT2Tokenizer

from src.graph import Graph
from src.IR.stablehlo_ir import parse_stablehlo_ops
from src.IR.lower_to_cactus import lower_to_cactus


NUM_LAYERS = 12
PROMPT = "Hello world"
MAX_T = 16
BENCH_STEPS = 8
WARMUP = 2
USE_KV_CACHE = False


print("Loading GPT2...")
hf_model = GPT2LMHeadModel.from_pretrained("gpt2")
hf_model.eval()
state = hf_model.state_dict()
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")


wte = state["transformer.wte.weight"].cpu().numpy().astype(np.float16)
wpe = state["transformer.wpe.weight"].cpu().numpy().astype(np.float16)

blocks = []

for i in range(NUM_LAYERS):
    ln1_g = state[f"transformer.h.{i}.ln_1.weight"].cpu().numpy().astype(np.float16)
    ln1_b = state[f"transformer.h.{i}.ln_1.bias"].cpu().numpy().astype(np.float16)

    c_attn = state[f"transformer.h.{i}.attn.c_attn.weight"].cpu().numpy().astype(np.float16)
    D = c_attn.shape[0]

    wq = np.ascontiguousarray(c_attn[:, :D])
    wk = np.ascontiguousarray(c_attn[:, D:2 * D])
    wv = np.ascontiguousarray(c_attn[:, 2 * D:3 * D])

    wo = np.ascontiguousarray(
        state[f"transformer.h.{i}.attn.c_proj.weight"].cpu().numpy().astype(np.float16)
    )

    ln2_g = state[f"transformer.h.{i}.ln_2.weight"].cpu().numpy().astype(np.float16)
    ln2_b = state[f"transformer.h.{i}.ln_2.bias"].cpu().numpy().astype(np.float16)

    w1 = np.ascontiguousarray(
        state[f"transformer.h.{i}.mlp.c_fc.weight"].cpu().numpy().astype(np.float16)
    )
    b1 = state[f"transformer.h.{i}.mlp.c_fc.bias"].cpu().numpy().astype(np.float16)

    w2 = np.ascontiguousarray(
        state[f"transformer.h.{i}.mlp.c_proj.weight"].cpu().numpy().astype(np.float16)
    )
    b2 = state[f"transformer.h.{i}.mlp.c_proj.bias"].cpu().numpy().astype(np.float16)

    blocks.append((ln1_g, ln1_b, wq, wk, wv, wo, ln2_g, ln2_b, w1, b1, w2, b2))

ln_f_g = state["transformer.ln_f.weight"].cpu().numpy().astype(np.float16)
ln_f_b = state["transformer.ln_f.bias"].cpu().numpy().astype(np.float16)
lm_head = np.ascontiguousarray(wte.T.copy())


def gelu(x):
    return 0.5 * x * (
        1.0
        + jnp.tanh(jnp.sqrt(2.0 / jnp.pi) * (x + 0.044715 * x * x * x))
    )


def layer_norm(x, g, b):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean((x - mean) ** 2, axis=-1, keepdims=True)
    return (x - mean) / jnp.sqrt(var + 1e-5) * g + b


def causal_mask(T, dtype):
    mask = jnp.tril(jnp.ones((T, T), dtype=dtype))
    return (1.0 - mask) * jnp.array(-10000.0, dtype=dtype)


def gpt2_full_model(tokens, wte, wpe, *params):
    T = tokens.shape[0]

    block_params = params[:-3]
    ln_f_g, ln_f_b, lm_head = params[-3:]

    one_hot = jax.nn.one_hot(tokens, wte.shape[0], dtype=wte.dtype)
    x = one_hot @ wte
    x = x + wpe[:T]

    for i in range(NUM_LAYERS):
        offset = i * 12
        ln1_g, ln1_b, wq, wk, wv, wo, ln2_g, ln2_b, w1, b1, w2, b2 = block_params[offset:offset + 12]

        h = layer_norm(x, ln1_g, ln1_b)

        q = h @ wq
        k = h @ wk
        v = h @ wv

        scores = (q @ k.T) / jnp.sqrt(h.shape[-1])
        scores = scores + causal_mask(T, scores.dtype)

        probs = jax.nn.softmax(scores, axis=-1)
        attn_out = probs @ v

        x = x + attn_out @ wo

        h = layer_norm(x, ln2_g, ln2_b)
        h = gelu(h @ w1 + b1)
        x = x + (h @ w2 + b2)

    x = layer_norm(x, ln_f_g, ln_f_b)
    return x @ lm_head


jit_model = jax.jit(gpt2_full_model)


def build_inputs(tokens_np):
    flat = []
    for b in blocks:
        flat.extend(b)

    return (
        tokens_np,
        wte,
        wpe,
        *flat,
        ln_f_g,
        ln_f_b,
        lm_head,
    )


def pad_tokens(tokens):
    arr = np.zeros((MAX_T,), dtype=np.int32)
    arr[: min(len(tokens), MAX_T)] = tokens[:MAX_T]
    return arr


def jax_inputs(inputs_np):
    return [
        jnp.array(x, dtype=jnp.int32) if x.dtype == np.int32 else jnp.array(x)
        for x in inputs_np
    ]


def extract_return_name(ir):
    for line in ir.splitlines():
        line = line.strip()
        if line.startswith("return "):
            m = re.search(r"return\s+(%[\w\d_]+)", line)
            if m:
                return m.group(1)
            raise RuntimeError(f"Bad return line: {line}")
    raise RuntimeError("No return found")


def build_cactus_graph(tokens_np, enable_attention_fusion):
    inputs_np = build_inputs(tokens_np)
    inputs_jax = jax_inputs(inputs_np)

    t0 = time.perf_counter()
    lowered = jit_model.lower(*inputs_jax)
    ir = str(lowered.compiler_ir(dialect="stablehlo"))
    export_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    return_name = extract_return_name(ir)
    nodes = parse_stablehlo_ops(ir)
    parse_s = time.perf_counter() - t0

    g = Graph()
    input_tensors = []
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
        input_tensors.append(t)

    t0 = time.perf_counter()
    env = lower_to_cactus(
        nodes,
        g,
        input_map,
        input_shapes,
        inputs_np,
        enable_attention_fusion=enable_attention_fusion,
    )
    lower_s = time.perf_counter() - t0

    return {
        "g": g,
        "input_tensors": input_tensors,
        "out_tensor": env[return_name],
        "export_s": export_s,
        "parse_s": parse_s,
        "lower_s": lower_s,
    }


def update_cactus_inputs(graph_obj, inputs_np):
    for t, arr in zip(graph_obj["input_tensors"], inputs_np):
        if arr.dtype == np.int32:
            arr = arr.astype(np.float32)
        else:
            arr = arr.astype(np.float16)
        graph_obj["g"].set_input(t, np.ascontiguousarray(arr))


def run_cactus(graph_obj, inputs_np):
    update_cactus_inputs(graph_obj, inputs_np)

    t0 = time.perf_counter()
    graph_obj["g"].execute()
    out = graph_obj["out_tensor"].numpy().copy()
    execute_s = time.perf_counter() - t0

    return out, execute_s


def run_jax(inputs_np):
    inputs = jax_inputs(inputs_np)

    t0 = time.perf_counter()
    res = jit_model(*inputs)
    res.block_until_ready()
    out = np.array(res, dtype=np.float16).copy()
    execute_s = time.perf_counter() - t0

    return out, execute_s


def compare_outputs(name, ref, out):
    ref32 = ref.astype(np.float32)
    out32 = out.astype(np.float32)
    diff = np.abs(ref32 - out32)

    cos = np.dot(ref32.ravel(), out32.ravel()) / (
        np.linalg.norm(ref32.ravel()) * np.linalg.norm(out32.ravel()) + 1e-8
    )

    print(f"\n{name}")
    print("shape:", out.shape)
    print("max diff:", float(diff.max()))
    print("mean diff:", float(diff.mean()))
    print("cosine:", float(cos))

    ref_top = np.argsort(ref32[-1])[-5:][::-1]
    out_top = np.argsort(out32[-1])[-5:][::-1]
    print("top5 overlap:", len(set(ref_top.tolist()) & set(out_top.tolist())), "/ 5")


def summarize_times(label, times):
    avg = statistics.mean(times)
    mn = min(times)
    mx = max(times)
    print(f"{label:<24} avg={avg:.5f}s min={mn:.5f}s max={mx:.5f}s")


def benchmark():
    tokens = tokenizer.encode(PROMPT)
    tokens_np = pad_tokens(tokens)

    print("Prompt:", repr(PROMPT))
    print("Tokens:", tokens)
    print("MAX_T:", MAX_T)

    print("\nBuilding unfused Cactus graph...")
    unfused = build_cactus_graph(tokens_np, enable_attention_fusion=False)

    print("\nBuilding fused Cactus graph...")
    fused = build_cactus_graph(tokens_np, enable_attention_fusion=True)

    print("\nBuild costs:")
    print(f"unfused export: {unfused['export_s']:.5f}s parse: {unfused['parse_s']:.5f}s lower: {unfused['lower_s']:.5f}s")
    print(f"fused   export: {fused['export_s']:.5f}s parse: {fused['parse_s']:.5f}s lower: {fused['lower_s']:.5f}s")

    jax_times = []
    unfused_times = []
    fused_times = []

    print("\nWarming up...")
    inputs_np = build_inputs(tokens_np)
    run_jax(inputs_np)
    run_cactus(unfused, inputs_np)
    run_cactus(fused, inputs_np)

    print("\nRunning benchmark...")

    for i in range(BENCH_STEPS):
        inputs_np = build_inputs(tokens_np)

        # ---- JAX ----
        jax_out, jax_s = run_jax(inputs_np)

        # ---- Unfused ----
        unfused_out, unfused_s = run_cactus(unfused, inputs_np)

        # ---- Fused ----
        fused_out, fused_s = run_cactus(fused, inputs_np)

        if i >= WARMUP:
            jax_times.append(jax_s)
            unfused_times.append(unfused_s)
            fused_times.append(fused_s)

    # -------------------------
    # RESULTS
    # -------------------------
    print("\n=== OUTPUT CHECK ===")
    compare_outputs("Unfused vs JAX", jax_out, unfused_out)
    compare_outputs("Fused vs JAX", jax_out, fused_out)

    print("\n=== TIMING ===")
    summarize_times("jax", jax_times)
    summarize_times("cactus_unfused", unfused_times)
    summarize_times("cactus_fused", fused_times)

    # speedups
    jax_avg = statistics.mean(jax_times)
    unfused_avg = statistics.mean(unfused_times)
    fused_avg = statistics.mean(fused_times)

    print("\n=== SPEEDUPS ===")
    print(f"Unfused vs JAX: {jax_avg / unfused_avg:.2f}x")
    print(f"Fused vs JAX:   {jax_avg / fused_avg:.2f}x")
    print(f"Fused vs Unfused: {unfused_avg / fused_avg:.2f}x")



def sample_top_k(logits, k=40, temperature=0.8):
    logits = logits.astype(np.float32)

    # temperature
    logits = logits / max(temperature, 1e-5)

    top_k_idx = np.argpartition(logits, -k)[-k:]
    top_k_logits = logits[top_k_idx]

    probs = np.exp(top_k_logits - np.max(top_k_logits))
    probs = probs / probs.sum()

    return np.random.choice(top_k_idx, p=probs)

def decode_benchmark():
    print("\n================ DECODE BENCHMARK ================")
    print("Mode:", "kv-cache decode path" if USE_KV_CACHE else "full graph path")

    tokens = tokenizer.encode(PROMPT)
    print("Prompt:", PROMPT)
    print("Initial tokens:", tokens)

    # Build graphs ONCE
    tokens_np = pad_tokens(tokens)

    print("\nBuilding graphs...")
    cactus = build_cactus_graph(tokens_np, enable_attention_fusion=True)

    if USE_KV_CACHE:
        raise NotImplementedError(
            "KV-cache decode for full 12-layer IR graph is intentionally separated. "
            "Run python/src/IR/kv_cache_one_layer_test.py for the staged one-layer path."
        )

    # Warmup
    print("\nWarmup...")
    inputs_np = build_inputs(tokens_np)
    run_jax(inputs_np)
    run_cactus(cactus, inputs_np)

    jax_times = []
    cactus_times = []

    cur_tokens = tokens.copy()

    for step in range(20):
        tokens_np = pad_tokens(cur_tokens)

        inputs_np = build_inputs(tokens_np)

        # ---------------- JAX ----------------
        t0 = time.perf_counter()
        jax_out, _ = run_jax(inputs_np)
        jax_t = time.perf_counter() - t0

        # ---------------- Cactus ----------------
        t0 = time.perf_counter()
        cactus_out, _ = run_cactus(cactus, inputs_np)
        cactus_t = time.perf_counter() - t0

        jax_times.append(jax_t)
        cactus_times.append(cactus_t)

        # ---------------- SAMPLE ----------------
        next_token = sample_top_k(cactus_out[-1])
        cur_tokens.append(int(next_token))

        decoded = tokenizer.decode(cur_tokens)
        print(f"\nStep {step+1}")
        print(f"JAX time:    {jax_t:.4f}s")
        print(f"Cactus time: {cactus_t:.4f}s")
        print("Text:", repr(decoded))

    # ---------------- SUMMARY ----------------
    print("\n================ RESULTS ================")

    def summarize(name, times):
        print(f"{name}: avg={np.mean(times):.4f}s min={np.min(times):.4f}s max={np.max(times):.4f}s")

    summarize("JAX", jax_times)
    summarize("Cactus", cactus_times)

    jax_avg = np.mean(jax_times)
    cactus_avg = np.mean(cactus_times)

    print("\nTokens/sec:")
    print(f"JAX:    {1.0 / jax_avg:.2f}")
    print(f"Cactus: {1.0 / cactus_avg:.2f}")

    print("\nSpeedup (Cactus vs JAX):", jax_avg / cactus_avg)




    print("\n================ RESULTS ================")

    print("Per-step times:")
    for i, (jt, ct) in enumerate(zip(jax_times, cactus_times), start=1):
        print(f"step {i:02d}: jax={jt:.5f}s cactus={ct:.5f}s")

    print("\nAverages:")
    print(f"JAX:    avg={np.mean(jax_times):.5f}s min={np.min(jax_times):.5f}s max={np.max(jax_times):.5f}s")
    print(f"Cactus: avg={np.mean(cactus_times):.5f}s min={np.min(cactus_times):.5f}s max={np.max(cactus_times):.5f}s")

    print("\nTokens/sec:")
    print(f"JAX:    {1.0 / np.mean(jax_times):.2f}")
    print(f"Cactus: {1.0 / np.mean(cactus_times):.2f}")

    print("\nSpeedup Cactus vs JAX:", np.mean(jax_times) / np.mean(cactus_times))


# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":
    decode_benchmark()
