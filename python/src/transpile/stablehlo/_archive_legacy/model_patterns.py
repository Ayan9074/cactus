"""Model-family pattern registry for StableHLO/JAX lowering.

This is a scaffold for capability-driven rewrites.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(frozen=True)
class PatternPack:
    name: str
    capabilities: Dict[str, bool] = field(default_factory=dict)
    notes: str = ""


_PATTERN_PACKS: Dict[str, PatternPack] = {
    "generic": PatternPack(
        name="generic",
        capabilities={
            "rope": False,
            "kv_cache": False,
            "gqa": False,
        },
        notes="Fallback pack for unknown StableHLO graphs.",
    ),
    "gemma": PatternPack(
        name="gemma",
        capabilities={
            "rope": True,
            "kv_cache": True,
            "gqa": True,
        },
        notes="Gemma-family defaults. Keep behavior synced with IR pipeline.",
    ),
}


def get_pattern_pack(name: str) -> PatternPack:
    return _PATTERN_PACKS.get(name, _PATTERN_PACKS["generic"])


def list_pattern_packs() -> List[str]:
    return sorted(_PATTERN_PACKS)
