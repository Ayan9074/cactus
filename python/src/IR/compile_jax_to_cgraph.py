"""Compatibility entrypoint for JAX callable -> cgraph compile."""

from src.transpile.stablehlo.compile_jax_to_cgraph import main


if __name__ == "__main__":
    main()
