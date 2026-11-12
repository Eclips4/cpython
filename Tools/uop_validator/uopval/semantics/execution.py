from collections.abc import Sequence
from dataclasses import replace

import z3

from ..domain.trace import Uop
from ..symbolic.state import ExitKind, FactKind, State, add_exit, with_fact
from .opaque import opaque
from .step import Handler, Step, UnsupportedUop
from .uops import core, frames, globals_builtins, typed, values

LOOP_BACK = "_JUMP_TO_TOP"
TERMINAL_EXITS = {
    "_EXIT_TRACE": ExitKind.SIDE,
    "_DEOPT": ExitKind.DEOPT,
    "_DYNAMIC_EXIT": ExitKind.DYNAMIC,
}

HANDLERS: dict[str, Handler] = (
    core.HANDLERS | values.HANDLERS | typed.HANDLERS | frames.HANDLERS | globals_builtins.HANDLERS
)


def modelled(op: str) -> bool:
    return op in HANDLERS or op == LOOP_BACK or op in TERMINAL_EXITS


def apply(uop: Uop, state: State, allow_opaque: bool) -> State:
    state = replace(state, popped=())
    if uop.op in HANDLERS:
        return HANDLERS[uop.op](uop, state)
    if allow_opaque:
        return opaque(uop, state)
    raise UnsupportedUop(f"{uop.op} at {uop.i} has no semantics in the model")


def execute(
    uops: Sequence[Uop],
    state: State,
    positions: Sequence[int] | None = None,
    allow_opaque: bool = False,
) -> list[Step]:
    steps: list[Step] = []
    for k, uop in enumerate(uops):
        if positions is not None:
            uop = replace(uop, i=positions[k])
        if uop.op == LOOP_BACK:
            state = with_fact(
                state,
                uop.i,
                FactKind.TERMINATOR,
                uop.op,
                None,
                "loop back: the state here is compared at the back-edge (R4)",
            )
            steps.append(Step(uop, state))
            break
        if uop.op in TERMINAL_EXITS:
            state = add_exit(
                state, uop.i, uop.op, TERMINAL_EXITS[uop.op], uop.target, z3.BoolVal(True)
            )
            state = with_fact(state, uop.i, FactKind.TERMINATOR, uop.op, None, "unconditional exit")
            steps.append(Step(uop, state))
            break
        state = apply(uop, state, allow_opaque)
        steps.append(Step(uop, state))
    return steps


def describe(state: State) -> list[str]:
    lines = [
        f"ip={z3.simplify(state.ip)} depth={len(state.stack)} valid={z3.simplify(state.valid)}"
    ]
    for k, s in enumerate(state.stack):
        lines.append(f"  stack[{k}] = {z3.simplify(s)}")
    for j, s in enumerate(state.locals):
        lines.append(f"  local[{j}] = {z3.simplify(s)}")
    return lines
