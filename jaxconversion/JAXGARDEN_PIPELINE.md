# JAXgarden Pipeline (Minimal)

This wrapper keeps the flow split and explicit:

1. Export StableHLO MLIR from a JAX/JAXgarden model.
2. Convert/load HF weights separately into NPZ args.
3. Lower MLIR to Cactus graph and run.

## Gemma2 (manual HF mapping path)

```bash
source /Users/ayan/Downloads/cactus/venv/bin/activate
python /Users/ayan/Downloads/cactus/jaxconversion/jaxgarden_pipeline.py \
  --model gemma2 \
  --out-prefix /private/tmp/gemma2_pipeline \
  --seq 16 \
  --real-hf-dir /Users/ayan/.jaxgarden/hf_models/google/gemma-2-2b-it \
  --prompt "The meaning of life is" \
  --tokenizer-model-id /Users/ayan/.jaxgarden/hf_models/google/gemma-2-2b-it \
  --show-text \
  --test
```

Outputs:
- `/private/tmp/gemma2_pipeline.stablehlo.mlir`
- `/private/tmp/gemma2_pipeline.inputs_weights.npz`
- `/private/tmp/gemma2_pipeline.cactus`
- `/private/tmp/gemma2_pipeline.convert_report.json`
- `/private/tmp/gemma2_pipeline.test.json`

## Llama (real HF->JAXgarden conversion path)

```bash
source /Users/ayan/Downloads/cactus/venv/bin/activate
python /Users/ayan/Downloads/cactus/jaxconversion/jaxgarden_pipeline.py \
  --model llama \
  --seq 8 \
  --hf-model-id TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --tokenizer-model-id /Users/ayan/.jaxgarden/hf_models/TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --prompt "The capital of France is" \
  --out-prefix /private/tmp/tinyllama_pipeline
```

Notes:
- In this repo, `Gemma2ForCausalLM` in local `jaxgarden` does not implement HF->JAX conversion yet.
- So Gemma2 currently uses the explicit/manual mapping path in `export_gemma_bundle.py`.
- Llama path uses `text_jaxgarden_llama_cactus.py` and checks practical fp16 parity plus next-token agreement.
