# JAX-only Refactor Plan (No behavior change first)

## Phase 0: Baseline
- Branch checkpoint: `backup/jax-only-baseline-2026-04-28`
- Current production path: `python/src/IR/*`

## Phase 1: Structure only
- Create `python/src/transpile/stablehlo/` package
- Add config + pattern registry + architecture docs
- Keep all runtime code unchanged

## Phase 2: Safe extraction
- Move pure helpers first:
  - StableHLO parsing utils
  - model adapter tables
  - artifact metadata helpers
- Keep wrappers in `python/src/IR/` importing new locations

## Phase 3: Pass manager
- Introduce explicit pass stages:
  - import
  - normalize
  - optimize
  - lower
  - bind
- Add per-pass logging + op-count deltas

## Phase 4: Capability-based lowering
- Infer capabilities from graph + metadata:
  - rope flavor
  - kv cache topology
  - gqa/mqa
  - quantizable rhs matmuls
- Select model pattern packs by capability + optional override

## Phase 5: Generic graph onboarding
- Public API for user-supplied JAX StableHLO artifacts
- Strict ABI validation for runtime bindings
- Fast failure with actionable diagnostics
