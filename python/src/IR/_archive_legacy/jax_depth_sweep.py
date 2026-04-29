import argparse

import jax
import jax.numpy as jnp
import numpy as np

from src.graph import Graph
from src.IR.stablehlo_ir import parse_stablehlo_ops
from src.IR.lower_to_cactus import lower_to_cactus


def deep_mlp(x, *params):
    # params = [w0, b0, w1, b1, ...]
    for i in range(0, len(params), 2):
        w = params[i]
        b = params[i + 1]
        x = jax.nn.relu(x @ w + b)
    return x


def export_stablehlo_text(x, params):
    lowered = jax.jit(deep_mlp).lower(x, *params)
    return str(lowered.compiler_ir(dialect="stablehlo"))


def build_params(depth: int, dim: int, value: float):
    x = np.full((1, dim), value, dtype=np.float16)
    params = []
    for _ in range(depth):
        w = np.full((dim, dim), value, dtype=np.float16)
        b = np.full((dim,), value, dtype=np.float16)
        params.extend([w, b])
    return x, params


def run_cactus(nodes, x_np, params_np):
    g = Graph()

    t_x = g.input(x_np.shape, Graph.FP16)
    g.set_input(t_x, x_np)

    input_map = {"%arg0": t_x}
    input_shapes = {"%arg0": x_np.shape}

    for i, p in enumerate(params_np, start=1):
        t = g.input(p.shape, Graph.FP16)
        g.set_input(t, p)
        input_map[f"%arg{i}"] = t
        input_shapes[f"%arg{i}"] = p.shape

    env = lower_to_cactus(nodes, g, input_map, input_shapes)
    out_name = nodes[-1].name
    out_tensor = env[out_name]

    g.execute()
    return out_tensor.numpy()


def run_depth(depth: int, dim: int, value: float):
    x_np, params_np = build_params(depth, dim, value)

    x = jnp.array(x_np)
    params = [jnp.array(p) for p in params_np]

    mlir_text = export_stablehlo_text(x, params)
    nodes = parse_stablehlo_ops(mlir_text)

    cactus_out = run_cactus(nodes, x_np, params_np)
    jax_out = np.array(jax.jit(deep_mlp)(x, *params), dtype=np.float16)

    abs_diff = np.abs(jax_out - cactus_out)
    return {
        "depth": depth,
        "num_nodes": len(nodes),
        "max_abs_diff": float(np.max(abs_diff)),
        "mean_abs_diff": float(np.mean(abs_diff)),
        "sample": cactus_out.reshape(-1)[:4].tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description="Depth sweep for clean JAX -> Cactus MLP")
    parser.add_argument("--depths", type=int, nargs="+", default=[2, 4, 8, 16])
    parser.add_argument("--dim", type=int, default=16)
    parser.add_argument("--value", type=float, default=0.01)
    args = parser.parse_args()

    print("Running depth sweep:", args.depths)
    print(f"dim={args.dim}, value={args.value}")

    for d in args.depths:
        result = run_depth(d, args.dim, args.value)
        print(
            f"depth={result['depth']:>2} | nodes={result['num_nodes']:>4} | "
            f"max_diff={result['max_abs_diff']:.8f} | "
            f"mean_diff={result['mean_abs_diff']:.8f} | "
            f"sample={result['sample']}"
        )


if __name__ == "__main__":
    main()
