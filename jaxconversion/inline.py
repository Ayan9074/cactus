"""
inline_calls.py
===============
Step 3: Inline all `call @foo` ops into @main, producing a flat
        single list of StableHLO ops with no function calls remaining.

This mirrors what production compilers (IREE, torch-mlir) do as their
first pass before any lowering: get everything into one flat scope so
every SSA value is visible at once.
"""

import re
from copy import deepcopy
from parse import Op, Func


def _rename_ssa(name: str, suffix: str) -> str:
    """Rename %foo -> %foo__<suffix> to avoid collisions when inlining."""
    return name + "__" + suffix


def inline_calls(funcs: dict[str, Func]) -> list[Op]:
    """
    Starting from @main, recursively inline every `call @foo` op.

    Returns
    -------
    list[Op]
        A flat, ordered list of Ops with no `call` ops remaining.
        SSA names are unique across all inlined scopes.
    """
    main = funcs.get("main")
    if main is None:
        raise ValueError("No @main function found in module.")

    call_count = [0]  # mutable counter for unique suffixes

    def inline_func(ops: list[Op], caller_args: list[str], callee: Func) -> list[Op]:
        """
        Inline callee into a call site.

        caller_args : the SSA values passed at the call site (bound to callee's inputs)
        Returns     : the callee's ops with SSA values renamed to avoid collisions,
                      and callee inputs substituted with caller args.
        """
        call_count[0] += 1
        suffix = f"{callee.name}_{call_count[0]}"

        # Map callee's arg names -> caller's SSA values
        arg_map: dict[str, str] = {}
        for (callee_arg, _type), caller_val in zip(callee.inputs, caller_args):
            arg_map[callee_arg] = caller_val

        # Build a renaming map for all SSA values defined inside the callee
        # so they don't collide with anything in the outer scope.
        callee_defined = set()
        for op in callee.ops:
            for out in op.outputs:
                callee_defined.add(out)

        rename: dict[str, str] = {}
        for ssa in callee_defined:
            rename[ssa] = _rename_ssa(ssa, suffix)

        def remap(ssa: str) -> str:
            if ssa in arg_map:
                return arg_map[ssa]   # substitute callee arg with caller value
            if ssa in rename:
                return rename[ssa]    # rename callee-local value
            return ssa                # pass through (shouldn't happen)

        inlined: list[Op] = []
        for op in callee.ops:
            new_op = Op(
                opcode=op.opcode,
                inputs=[remap(s) for s in op.inputs],
                outputs=[rename.get(s, s) for s in op.outputs],
                attributes=op.attributes,
                result_types=list(op.result_types),
                input_types=list(op.input_types),
            )
            inlined.append(new_op)

        return inlined

    def resolve(ops: list[Op]) -> list[Op]:
        """Walk ops; expand any `call` op by recursively inlining the callee."""
        result: list[Op] = []
        for op in ops:
            if op.opcode != "call":
                result.append(op)
                continue

            # Extract callee name from attributes: '@relu()'
            callee_name_m = re.search(r"@(\w+)", op.attributes)
            if callee_name_m is None:
                raise ValueError(f"Could not parse callee name from: {op.attributes!r}")
            callee_name = callee_name_m.group(1)

            if callee_name not in funcs:
                raise ValueError(f"Call to unknown function @{callee_name}")

            callee = funcs[callee_name]

            # Recursively inline the callee's body first (handles nested calls)
            callee_flat_ops = resolve(callee.ops)

            # Build a temporary Func with already-resolved ops for inlining
            callee_resolved = Func(
                name=callee.name,
                inputs=callee.inputs,
                outputs=callee.outputs,
                ops=callee_flat_ops,
                return_vals=callee.return_vals,
            )

            inlined_ops = inline_func(result, op.inputs, callee_resolved)

            # The callee's return vals, after renaming, become this call's outputs.
            # We need to add an identity mapping so downstream ops can find them.
            call_count_val = call_count[0]
            suffix = f"{callee_name}_{call_count_val}"
            renamed_returns = [_rename_ssa(r, suffix) for r in callee.return_vals]

            for inlined_op in inlined_ops:
                result.append(inlined_op)

            # If the call had output SSA names (e.g. %1 = call @relu(%0)),
            # map them to the callee's renamed return values via alias ops.
            for call_out, ret_val in zip(op.outputs, renamed_returns):
                alias = Op(
                    opcode="_alias",   # virtual op: call_out = ret_val
                    inputs=[ret_val],
                    outputs=[call_out],
                    attributes="",
                    result_types=op.result_types,
                    input_types=op.result_types,
                )
                result.append(alias)

        return result

    return resolve(main.ops)


# ---------------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')

    import jax, jax.numpy as jnp
    from test import jax_to_mlir
    from parse import parse_mlir

    def simple_model(x, w):
        h = x @ w
        return jax.nn.relu(h)

    x = jnp.ones((4, 8),  dtype=jnp.float32)
    w = jnp.ones((8, 16), dtype=jnp.float32)

    mlir_text = jax_to_mlir(simple_model, (x, w))
    funcs = parse_mlir(mlir_text)

    print("=== Before inlining (@main only) ===")
    for op in funcs["main"].ops:
        print(f"  [{op.opcode}] {op.outputs} <- {op.inputs}")

    flat_ops = inline_calls(funcs)

    print("\n=== After inlining (flat) ===")
    for op in flat_ops:
        print(f"  [{op.opcode}] {op.outputs} <- {op.inputs}  attrs:{op.attributes!r}")