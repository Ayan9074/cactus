"""Compatibility entrypoint for generic JAX StableHLO compile."""

from src.transpile.stablehlo.compile_mlir_to_cgraph import main


if __name__ == "__main__":
    main()
