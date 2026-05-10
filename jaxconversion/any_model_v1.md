# Any-Model V1 Architecture

## Goal
Accept a broad set of user-provided models through one public contract and return:
- a Cactus graph when supported
- a precise compatibility report when not fully supported

## Public Contract (V1)
Input artifact:
1. StableHLO MLIR (`.mlir`) with static shapes.
2. Optional weights bundle (external files + manifest).

Output artifact:
1. Cactus graph file (`.cactus`) when lowering succeeds.
2. Compatibility report (`json`) always.
3. Bundle manifest (`manifest.json`) for reproducible deployment.

## Pipeline
1. Parse StableHLO text into `IRGraph` (`parse.py`).
2. Run capability analysis (`compatibility.py`):
   - supported ops
   - unsupported ops
   - dynamic-shape risk
   - estimated tier (A/B/C)
3. Select profile (`auto` for now):
   - `generic` (default patterns only)
   - future: `llama_family`, `qwen_family`, `gemma_family`, `bert_family`
4. Lower with `lower_to_cactus`.
5. Save graph + emit bundle manifest.

## Tiers
- Tier A (`ready`): all live ops supported by primitive or patterns.
- Tier B (`needs_rewrites`): mostly supported, but fallback/extra patterns needed.
- Tier C (`blocked`): hard blockers present.

## What “Any Model” Means in V1
“Any model that can export StableHLO with static shapes and whose used ops are currently supported by Cactus lowering (or future fallback path).”

## Immediate Next Milestones
1. Add family pattern packs (`llama_family`, `qwen_family`, `gemma_family`).
2. Add segmented fallback execution for unsupported islands.
3. Add hosted conversion endpoint with artifact cache by model hash.
