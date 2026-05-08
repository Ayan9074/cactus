"""
jax_to_mlir.py
==============
Step 1: Lower a JAX function to StableHLO and print/save the MLIR text.

Usage
-----
    python jax_to_mlir.py
"""

import jax
import jax.numpy as jnp


def jax_to_mlir(fn, example_inputs: tuple) -> str:
    """
    JIT-compile a JAX function and lower it to StableHLO MLIR text.

    Parameters
    ----------
    fn : callable
        Any JAX-traceable function.
    example_inputs : tuple
        Concrete JAX arrays to use for tracing shapes/dtypes.

    Returns
    -------
    str
        The StableHLO MLIR module as text.
    """
    lowered = jax.jit(fn).lower(*example_inputs)
    with open("out.mlir", 'w') as f:
        f.write(lowered.as_text())
    return lowered.as_text()


# ---------------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    def simple_model(x, w):
        h = x @ w          # matmul
        return jax.nn.relu(h)

    x = jnp.ones((4, 8),  dtype=jnp.float32)
    w = jnp.ones((8, 16), dtype=jnp.float32)

    mlir_text = jax_to_mlir(simple_model, (x, w))
    with open("out.mlir", 'w') as f:
        f.write(mlir_text)