# StableHLO -> Cactus (JAX-only)

This is the JAX-only path:

1. Export a JAX function to StableHLO (`.mlir`)
2. Provide sample input tensors (`arg0..argN`)
3. Compile to `.cgraph`
4. Run with new runtime inputs

## Compile any JAX StableHLO graph

```bash
PYTHONPATH=/Users/ayan/cactus/python python3 -m src.IR.compile_mlir_to_cgraph \
  --stablehlo /abs/path/model.stablehlo.mlir \
  --inputs /abs/path/inputs.npz \
  --out /abs/path/model.cgraph \
  --meta /abs/path/model.cgraph.meta.json
```

`inputs.npz` must contain all model args as one of:
- `arg0`, `arg1`, ...
- `%arg0`, `%arg1`, ...
- `0`, `1`, ...

## Run compiled graph

```bash
PYTHONPATH=/Users/ayan/cactus/python python3 -m src.IR.run_cgraph_generic \
  --cgraph /abs/path/model.cgraph \
  --meta /abs/path/model.cgraph.meta.json \
  --inputs /abs/path/runtime_inputs.npz \
  --out /abs/path/output.npy
```

## Current behavior / constraints

- JAX-only frontend (no PyTorch/ONNX in this pipeline).
- Best-effort lowering through existing `lower_to_cactus` implementation.
- Your graph must use StableHLO ops that the current lowerer supports.
- Compile metadata stores exact input/output node ids so loaded `.cgraph` can be rebound safely.

## Architecture direction

Target staged flow:
- export -> import -> normalize -> optimize -> lower -> bind -> run

Migration strategy:
- Keep `python/src/IR/*` compatible while moving pieces into `python/src/transpile/stablehlo/*`.

## One-command JAX callable -> cgraph

```bash
PYTHONPATH=/Users/ayan/cactus/python python3 -m src.IR.compile_jax_to_cgraph \
  --callable your_pkg.your_module:your_jax_fn \
  --inputs /abs/path/inputs.npz \
  --out /abs/path/model.cgraph \
  --meta /abs/path/model.cgraph.meta.json \
  --save-mlir /abs/path/model_exported.stablehlo.mlir
```

Notes:
- `--inputs` NPZ is used both to lower the JAX function and to build sample graph inputs.
- Preferred NPZ keys: `arg0`, `arg1`, ... (also supports `%arg0`, `0`, etc.).
- Callable should be pure JAX function with positional args matching the NPZ order.

## Artifact details

The compiler now emits:
- `model.cgraph`
- `model.cgraph.meta.json`
- `model.cgraph.literal_inputs/`

`literal_inputs/` stores non-runtime graph inputs (e.g. lowered constants), which are rebound automatically by `run_cgraph_generic`.

## FP32 matmul note

Some JAX StableHLO exports insert `convert -> f32` around `dot_general`. The current Cactus matmul path is FP16-only in this flow, so dot operands are cast back to FP16 during lowering to keep execution valid.
