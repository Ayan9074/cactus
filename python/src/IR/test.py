import os
import sys
import numpy as np

# Run from repo root:
#   PYTHONPATH=python python python/src/IR/test_generic_lowerer_smoke.py
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
PYTHON_DIR = os.path.join(ROOT, "python")
if PYTHON_DIR not in sys.path:
    sys.path.insert(0, PYTHON_DIR)

import jax
import jax.numpy as jnp

from src.graph import Graph
from src.IR.stablehlo_ir import parse_stablehlo_ops
from src.IR.lower_to_cactus_rewrite import lower_to_cactus


def stablehlo_text(fn, *args):
    lowered = jax.jit(fn).lower(*args)
    ir = lowered.compiler_ir(dialect="stablehlo")
    if hasattr(ir, "as_text"):
        return ir.as_text()
    return str(ir)


def extract_return_names(ir_text):
    for line in ir_text.splitlines():
        line = line.strip()
        if line.startswith("return "):
            return [x.strip() for x in line.split("return", 1)[1].split(":")[0].split(",")]
    raise RuntimeError("No return line found")


def graph_dtype_for_np(x):
    # For now this lowerer is testing Cactus FP16 execution contract.
    if np.issubdtype(np.asarray(x).dtype, np.integer):
        return Graph.INT8
    return Graph.FP16


def run_cactus_from_jax(fn, inputs, dtype_policy="fp16", print_ir=False):
    inputs = [np.asarray(x) for x in inputs]

    jax_inputs = [jnp.asarray(x, dtype=x.dtype) for x in inputs]
    ir = stablehlo_text(fn, *jax_inputs)

    if print_ir:
        print(ir)

    nodes = parse_stablehlo_ops(ir)
    return_names = extract_return_names(ir)

    g = Graph()
    input_map = {}
    input_shapes = {}

    for i, arr in enumerate(inputs):
        name = f"%arg{i}"
        dtype = graph_dtype_for_np(arr)
        t = g.input(arr.shape, dtype)
        g.set_input(t, arr, dtype=dtype)
        input_map[name] = t
        input_shapes[name] = arr.shape

    env = lower_to_cactus(
        nodes,
        g,
        input_map,
        input_shapes=input_shapes,
        dtype_policy=dtype_policy,
    )

    g.execute()

    if len(return_names) != 1:
        raise NotImplementedError(f"Only single-output tests supported. Got {return_names}")

    out = env[return_names[0]].numpy().astype(np.float32)

    ref = np.asarray(fn(*jax_inputs)).astype(np.float32)
    jax_inputs_fp16 = [jnp.asarray(x, dtype=jnp.float16) for x in inputs]
    ref_fp16 = np.asarray(fn(*jax_inputs_fp16)).astype(np.float32)

    return ref, ref_fp16, out, ir


def compare(name, ref, out, atol=2e-1, rtol=2e-1):
    ref = np.asarray(ref, dtype=np.float32)
    out = np.asarray(out, dtype=np.float32)

    if ref.shape != out.shape:
        print(f"❌ {name}: shape mismatch ref={ref.shape} out={out.shape}")
        return False

    diff = np.abs(ref - out)
    max_err = float(diff.max()) if diff.size else 0.0
    mean_err = float(diff.mean()) if diff.size else 0.0

    ref_flat = ref.reshape(-1)
    out_flat = out.reshape(-1)
    cos = float(
        np.dot(ref_flat, out_flat)
        / (np.linalg.norm(ref_flat) * np.linalg.norm(out_flat) + 1e-8)
    )

    ok = np.allclose(ref, out, atol=atol, rtol=rtol)

    status = "✅" if ok else "❌"
    print(
        f"{status} {name}: shape={ref.shape} "
        f"max={max_err:.6f} mean={mean_err:.6f} cos={cos:.6f}"
    )

    if not ok:
        idx = np.unravel_index(np.argmax(diff), diff.shape)
        print("max diff idx =", idx)
        print("ref at idx =", ref[idx])
        print("out at idx =", out[idx])
        print("abs diff =", diff[idx])
        print("ref[:8] =", ref.reshape(-1)[:8])
        print("out[:8] =", out.reshape(-1)[:8])

    return ok


def test_add_broadcast():
    def fn(x, b):
        return x + b

    x = np.random.randn(2, 4).astype(np.float32)
    b = np.random.randn(4).astype(np.float32)
    return run_cactus_from_jax(fn, [x, b])


def run_jax_fp16_ref(fn, inputs):
    jax_inputs = [jnp.asarray(x, dtype=jnp.float16) for x in inputs]
    return np.asarray(fn(*jax_inputs)).astype(np.float32)

def test_matmul():
    def fn(x, w):
        return x @ w

    x = np.random.randn(3, 5).astype(np.float32)
    w = np.random.randn(5, 4).astype(np.float32)
    return run_cactus_from_jax(fn, [x, w])


def test_rmsnorm_decomposed():
    def fn(x, w):
        var = jnp.mean(x * x, axis=-1, keepdims=True)
        return x * jax.lax.rsqrt(var + 1e-6) * w

    x = np.random.randn(2, 8).astype(np.float32)
    w = np.random.randn(8).astype(np.float32)
    return run_cactus_from_jax(fn, [x, w])


def test_softmax_decomposed():
    def fn(x):
        # Avoid relying on stablehlo.custom_call/etc.
        m = jnp.max(x, axis=-1, keepdims=True)
        e = jnp.exp(x - m)
        return e / jnp.sum(e, axis=-1, keepdims=True)

    x = np.random.randn(2, 6).astype(np.float32)
    return run_cactus_from_jax(fn, [x])


def test_tiny_mlp():
    def fn(x, w1, b1, w2, b2):
        h = jnp.tanh((x @ w1) + b1)
        return (h @ w2) + b2

    x = np.random.randn(2, 5).astype(np.float32)
    w1 = np.random.randn(5, 7).astype(np.float32)
    b1 = np.random.randn(7).astype(np.float32)
    w2 = np.random.randn(7, 3).astype(np.float32)
    b2 = np.random.randn(3).astype(np.float32)
    return run_cactus_from_jax(fn, [x, w1, b1, w2, b2])


def test_attention_tiny_no_mask():
    def fn(q, k, v):
        # q, k, v: [heads, seq, dim]
        kt = jnp.swapaxes(k, -1, -2)
        scores = q @ kt
        m = jnp.max(scores, axis=-1, keepdims=True)
        e = jnp.exp(scores - m)
        probs = e / jnp.sum(e, axis=-1, keepdims=True)
        return probs @ v

    q = np.random.randn(2, 3, 4).astype(np.float32)
    k = np.random.randn(2, 3, 4).astype(np.float32)
    v = np.random.randn(2, 3, 5).astype(np.float32)
    return run_cactus_from_jax(fn, [q, k, v])



def test_tiny_transformer_block():
    def rms(x, w):
        var = jnp.mean(x * x, axis=-1, keepdims=True)
        return x * jax.lax.rsqrt(var + 1e-6) * w

    def softmax(x):
        m = jnp.max(x, axis=-1, keepdims=True)
        e = jnp.exp(x - m)
        return e / jnp.sum(e, axis=-1, keepdims=True)

    def fn(x, ln1, q_w, k_w, v_w, o_w, ln2, gate_w, up_w, down_w):
        # x: [T, D]
        T = x.shape[0]
        D = x.shape[1]
        H = 2
        HD = D // H

        h = rms(x, ln1)

        q = h @ q_w
        k = h @ k_w
        v = h @ v_w

        q = jnp.reshape(q, (T, H, HD))
        k = jnp.reshape(k, (T, H, HD))
        v = jnp.reshape(v, (T, H, HD))

        q = jnp.transpose(q, (1, 0, 2))  # [H, T, HD]
        k = jnp.transpose(k, (1, 0, 2))
        v = jnp.transpose(v, (1, 0, 2))

        scores = (q @ jnp.swapaxes(k, -1, -2)) / jnp.sqrt(jnp.array(float(HD), dtype=x.dtype))

        # causal mask
        causal = jnp.tril(jnp.ones((T, T), dtype=x.dtype))
        mask = (1.0 - causal) * -10000.0
        scores = scores + mask[None, :, :]

        probs = softmax(scores)
        attn = probs @ v

        attn = jnp.transpose(attn, (1, 0, 2))  # [T, H, HD]
        attn = jnp.reshape(attn, (T, D))
        attn = attn @ o_w

        x = x + attn

        h2 = rms(x, ln2)
        gate = jax.nn.gelu(h2 @ gate_w)
        up = h2 @ up_w
        mlp = (gate * up) @ down_w

        return x + mlp

    T = 4
    D = 8
    M = 16

    x = np.random.randn(T, D).astype(np.float32)
    ln1 = np.random.randn(D).astype(np.float32)
    q_w = np.random.randn(D, D).astype(np.float32)
    k_w = np.random.randn(D, D).astype(np.float32)
    v_w = np.random.randn(D, D).astype(np.float32)
    o_w = np.random.randn(D, D).astype(np.float32)
    ln2 = np.random.randn(D).astype(np.float32)
    gate_w = np.random.randn(D, M).astype(np.float32)
    up_w = np.random.randn(D, M).astype(np.float32)
    down_w = np.random.randn(M, D).astype(np.float32)

    return run_cactus_from_jax(
        fn,
        [x, ln1, q_w, k_w, v_w, o_w, ln2, gate_w, up_w, down_w],
    )


def test_two_tiny_transformer_blocks():
    def rms(x, w):
        var = jnp.mean(x * x, axis=-1, keepdims=True)
        return x * jax.lax.rsqrt(var + 1e-6) * w

    def softmax(x):
        m = jnp.max(x, axis=-1, keepdims=True)
        e = jnp.exp(x - m)
        return e / jnp.sum(e, axis=-1, keepdims=True)

    def block(x, ln1, q_w, k_w, v_w, o_w, ln2, gate_w, up_w, down_w):
        T = x.shape[0]
        D = x.shape[1]
        H = 4
        HD = D // H

        h = rms(x, ln1)

        q = h @ q_w
        k = h @ k_w
        v = h @ v_w

        q = jnp.reshape(q, (T, H, HD))
        k = jnp.reshape(k, (T, H, HD))
        v = jnp.reshape(v, (T, H, HD))

        q = jnp.transpose(q, (1, 0, 2))
        k = jnp.transpose(k, (1, 0, 2))
        v = jnp.transpose(v, (1, 0, 2))

        scores = (q @ jnp.swapaxes(k, -1, -2)) / jnp.sqrt(jnp.array(float(HD), dtype=x.dtype))

        causal = jnp.tril(jnp.ones((T, T), dtype=x.dtype))
        mask = (1.0 - causal) * -10000.0
        scores = scores + mask[None, :, :]

        probs = softmax(scores)
        attn = probs @ v

        attn = jnp.transpose(attn, (1, 0, 2))
        attn = jnp.reshape(attn, (T, D))
        attn = attn @ o_w

        x = x + attn

        h2 = rms(x, ln2)
        gate = jax.nn.gelu(h2 @ gate_w)
        up = h2 @ up_w
        mlp = (gate * up) @ down_w

        return x + mlp

    def fn(x, *params):
        p = list(params)
        x = block(x, *p[:9])
        x = block(x, *p[9:18])
        return x

    T = 8
    D = 16
    M = 32

    x = (np.random.randn(T, D) * 0.5).astype(np.float32)

    scale_attn = 1.0 / np.sqrt(D)
    scale_mlp = 1.0 / np.sqrt(M)

    params = []
    for _ in range(2):
        params.extend([
            np.ones(D, dtype=np.float32),                                  # ln1
            (np.random.randn(D, D) * scale_attn).astype(np.float32),        # q
            (np.random.randn(D, D) * scale_attn).astype(np.float32),        # k
            (np.random.randn(D, D) * scale_attn).astype(np.float32),        # v
            (np.random.randn(D, D) * scale_attn).astype(np.float32),        # o
            np.ones(D, dtype=np.float32),                                  # ln2
            (np.random.randn(D, M) * scale_attn).astype(np.float32),        # gate
            (np.random.randn(D, M) * scale_attn).astype(np.float32),        # up
            (np.random.randn(M, D) * scale_mlp).astype(np.float32),         # down
        ])

    return run_cactus_from_jax(fn, [x] + params)


def main():
    np.random.seed(0)

    tests = [
        ("add_broadcast", test_add_broadcast),
        ("matmul", test_matmul),
        ("rmsnorm_decomposed", test_rmsnorm_decomposed),
        ("softmax_decomposed", test_softmax_decomposed),
        ("tiny_mlp", test_tiny_mlp),
        ("attention_tiny_no_mask", test_attention_tiny_no_mask),
        ("tiny_transformer_block", test_tiny_transformer_block),
        ("two_tiny_transformer_blocks", test_two_tiny_transformer_blocks),
    ]

    passed = 0
    failed = 0

    for name, fn in tests:
        print(f"\n--- {name} ---")
        ok = False

        try:
            ref_fp32, ref_fp16, out, _ = fn()

            # Print FP32 comparison for information.
            # This is the ideal math target, but Cactus is mostly FP16 right now.
            compare(
                name + "_vs_jax_fp32",
                ref_fp32,
                out,
                atol=2e-1,
                rtol=2e-1,
            )

            # Use FP16 comparison as the actual pass/fail target.
            ok = compare(
                name + "_vs_jax_fp16",
                ref_fp16,
                out,
                atol=3e-1,
                rtol=3e-1,
            )

        except Exception as e:
            ok = False
            print(f"❌ {name}: exception: {type(e).__name__}: {e}")

        if ok:
            passed += 1
        else:
            failed += 1


if __name__ == "__main__":
    main()