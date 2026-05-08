"""
pattern_registry.py
===================
Layer 2: pattern matching and fusion.

A Pattern inspects a window of IRNodes and, if it recognises a high-level
subgraph (relu, softmax, rms_norm, …), emits a single fused Cactus call
and tells the lowering loop how many nodes to consume.

Adding support for a new model = write one patterns_<name>.py file.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from parse_mlir import IRGraph, IRNode


@dataclass
class LoweringCtx:
    graph:     Any                # cactus Graph object
    env:       dict[str, Any]     # ssa_name -> cactus Tensor
    constants: dict[str, Any]     # ssa_name -> IRConstant
    ir:        Any                # IRGraph (for lookahead)


# (ctx, nodes, idx) -> (out_tensors, nodes_consumed) | None
PatternHandler = Callable[["LoweringCtx", list, int], Optional[tuple[list[Any], int]]]


@dataclass
class Pattern:
    name:        str
    handler:     PatternHandler
    trigger_ops: Optional[set[str]] = None   # only try if leading op is one of these


class PatternRegistry:
    def __init__(self):
        self._patterns: list[Pattern] = []

    def register(self, pattern: Pattern) -> None:
        self._patterns.append(pattern)

    def match(self, ctx: LoweringCtx, nodes: list, idx: int) -> Optional[tuple[list[Any], int]]:
        current_op = nodes[idx].op
        for p in self._patterns:
            if p.trigger_ops and current_op not in p.trigger_ops:
                continue
            result = p.handler(ctx, nodes, idx)
            if result is not None:
                return result
        return None


# ---------------------------------------------------------------------------
# Shared helpers used by pattern handlers
# ---------------------------------------------------------------------------

def resolve(ctx: LoweringCtx, ssa: str) -> Any:
    if ssa not in ctx.env:
        raise KeyError(f"SSA value {ssa!r} not yet emitted")
    return ctx.env[ssa]


def is_const(ctx: LoweringCtx, ssa: str) -> bool:
    return ssa in ctx.constants


def const_value(ctx: LoweringCtx, ssa: str) -> Any:
    return ctx.constants[ssa].value