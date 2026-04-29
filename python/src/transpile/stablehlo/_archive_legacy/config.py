"""Central configuration scaffold for JAX-only StableHLO pipeline."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class StableHLOPipelineConfig:
    model_name: str
    pattern_pack: str = "generic"
    weight_precision: str = "fp16"
    enable_silu_fusion: bool = False
    validate_each_step: bool = True
