# Cactus IR Core Paths

This folder is intentionally slimmed down. The two core implementation files are:

- `stablehlo_ir.py` (StableHLO parser)
- `lower_to_cactus.py` (StableHLO -> Cactus lowering)

## Unified Entry Point

Use:

`python -m src.IR.cactus_pipeline <subcommand> -- <args>`

Subcommands:

- `path1-python`: Python/JAX callable -> StableHLO -> Cactus
- `path2-jax-graph`: existing StableHLO (`.mlir`) -> Cactus
- `path3-flax-auto`: `FlaxAutoModelForCausalLM` -> StableHLO -> Cactus
- `path4-gemma`: Gemma-specific optimized bridge path
- `run`: run compiled Cactus artifact
- `profile`: generic JAX path profiling

## Essential IR Wrappers

- `compile_jax_to_cgraph.py`
- `compile_mlir_to_cgraph.py`
- `import_flax_causallm.py`
- `run_cgraph_generic.py`
- `profile_jax_generic.py`
- `cactus_pipeline.py`

## Model-Specific (Path 4)

- `gemma_kv_cache_compiler_bridge.py`
- `gemma_cactus_bridge.py`

## Archived Legacy Files

Older experiments and one-off scripts were moved to:

- `IR/_archive_legacy/`
- `transpile/stablehlo/_archive_legacy/`
