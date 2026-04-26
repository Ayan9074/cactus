import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from src.graph import Graph
from src.IR.stablehlo_ir import parse_stablehlo_ops
from src.IR.lower_to_cactus import lower_to_cactus


def tiny_mlp(x, w1, b1, w2, b2):
    x = jax.nn.relu(x @ w1 + b1)
    x = jax.nn.relu(x @ w2 + b2)
    return x


def export_stablehlo_text(x, w1, b1, w2, b2):
    lowered = jax.jit(tiny_mlp).lower(x, w1, b1, w2, b2)
    return str(lowered.compiler_ir(dialect="stablehlo"))


def build_inputs(seed: int):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((1, 16)).astype(np.float16)
    w1 = rng.standard_normal((16, 32)).astype(np.float16)
    b1 = rng.standard_normal((32,)).astype(np.float16)
    w2 = rng.standard_normal((32, 16)).astype(np.float16)
    b2 = rng.standard_normal((16,)).astype(np.float16)
    return x, w1, b1, w2, b2


def build_ones_inputs():
    x = np.ones((1, 16), dtype=np.float16)
    w1 = np.ones((16, 32), dtype=np.float16)
    b1 = np.ones((32,), dtype=np.float16)
    w2 = np.ones((32, 16), dtype=np.float16)
    b2 = np.ones((16,), dtype=np.float16)
    return x, w1, b1, w2, b2


def run_cactus(nodes, x, w1, b1, w2, b2):
    g = Graph()

    t_x = g.input(x.shape, Graph.FP16)
    t_w1 = g.input(w1.shape, Graph.FP16)
    t_b1 = g.input(b1.shape, Graph.FP16)
    t_w2 = g.input(w2.shape, Graph.FP16)
    t_b2 = g.input(b2.shape, Graph.FP16)

    g.set_input(t_x, x)
    g.set_input(t_w1, w1)
    g.set_input(t_b1, b1)
    g.set_input(t_w2, w2)
    g.set_input(t_b2, b2)

    input_map = {
        "%arg0": t_x,
        "%arg1": t_w1,
        "%arg2": t_b1,
        "%arg3": t_w2,
        "%arg4": t_b2,
    }
    input_shapes = {
        "%arg0": x.shape,
        "%arg1": w1.shape,
        "%arg2": b1.shape,
        "%arg3": w2.shape,
        "%arg4": b2.shape,
    }

    env = lower_to_cactus(nodes, g, input_map, input_shapes)
    out_name = nodes[-1].name
    out_tensor = env[out_name]

    g.execute()
    return out_tensor.numpy()


def main():
    parser = argparse.ArgumentParser(description="End-to-end JAX -> StableHLO -> Cactus demo")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-mlir", type=Path, default=None)
    parser.add_argument("--ones", action="store_true", help="Use deterministic all-ones tensors")
    args = parser.parse_args()

    if args.ones:
        x_np, w1_np, b1_np, w2_np, b2_np = build_ones_inputs()
    else:
        x_np, w1_np, b1_np, w2_np, b2_np = build_inputs(args.seed)

    x = jnp.array(x_np)
    w1 = jnp.array(w1_np)
    b1 = jnp.array(b1_np)
    w2 = jnp.array(w2_np)
    b2 = jnp.array(b2_np)

    mlir_text = export_stablehlo_text(x, w1, b1, w2, b2)

    if args.save_mlir is not None:
        args.save_mlir.write_text(mlir_text)
        print(f"Saved StableHLO to: {args.save_mlir}")

    nodes = parse_stablehlo_ops(mlir_text)
    cactus_out = run_cactus(nodes, x_np, w1_np, b1_np, w2_np, b2_np)

    jax_out = np.array(jax.jit(tiny_mlp)(x, w1, b1, w2, b2), dtype=np.float16)

    abs_diff = np.abs(jax_out - cactus_out)
    print("JAX output shape:", jax_out.shape)
    print("Cactus output shape:", cactus_out.shape)
    print("Max abs diff:", float(abs_diff.max()))
    print("Mean abs diff:", float(abs_diff.mean()))
    print("Sample Cactus output:", cactus_out.reshape(-1)[:8])


if __name__ == "__main__":
    main()
