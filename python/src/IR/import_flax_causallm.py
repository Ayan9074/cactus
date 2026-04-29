"""Compatibility entrypoint for FlaxAutoModelForCausalLM import path."""

from src.transpile.stablehlo.import_flax_causallm import main


if __name__ == "__main__":
    main()
