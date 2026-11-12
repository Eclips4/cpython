from collections.abc import Callable
from dataclasses import dataclass, replace

import z3

from ..domain.trace import Constant, Uop
from ..symbolic.ledger import incref_on_dup, ledger_add
from ..symbolic.objects import is_immortal_id
from ..symbolic.sorts import Slot
from ..symbolic.state import ExitKind, FactKind, State, add_exit, allocation_exit, push, with_fact
from ..symbolic.summaries import constant_facts


class UnsupportedUop(Exception):
    pass


@dataclass(frozen=True)
class Step:
    uop: Uop
    state: State


Handler = Callable[[Uop, State], State]


def nos(state: State, uop: Uop) -> z3.DatatypeRef:
    if len(state.stack) < 2:
        raise UnsupportedUop(f"{uop.op} at {uop.i}: stack too shallow")
    return state.stack[-2]


def tos(state: State, uop: Uop) -> z3.DatatypeRef:
    if not state.stack:
        raise UnsupportedUop(f"{uop.op} at {uop.i}: stack empty")
    return state.stack[-1]


def at_depth(state: State, uop: Uop, depth: int) -> z3.DatatypeRef:
    if len(state.stack) < depth:
        raise UnsupportedUop(f"{uop.op} at {uop.i}: stack too shallow")
    return state.stack[-depth]


def pop_n(state: State, uop: Uop, n: int) -> tuple[list[z3.DatatypeRef], State]:
    if len(state.stack) < n:
        raise UnsupportedUop(f"{uop.op} at {uop.i}: stack too shallow for {n} inputs")
    slots = list(state.stack[len(state.stack) - n :]) if n else []
    return slots, replace(
        state, stack=state.stack[: len(state.stack) - n], popped=state.popped + tuple(slots)
    )


def local_slot(uop: Uop, state: State) -> z3.DatatypeRef:
    if uop.oparg >= len(state.locals):
        raise UnsupportedUop(f"{uop.op} at {uop.i}: local {uop.oparg} outside the frame")
    return state.locals[uop.oparg]


def precondition(uop: Uop, state: State, expr: z3.BoolRef, note: str) -> State:
    return with_fact(state, uop.i, FactKind.PRECONDITION, uop.op, expr, note)


def exit_if(
    uop: Uop,
    state: State,
    kind: ExitKind,
    condition: z3.BoolRef,
    handover: State | None = None,
) -> State:
    return add_exit(state, uop.i, uop.op, kind, uop.target, condition, handover)


def allocation_exit_if(
    uop: Uop,
    state: State,
    kind: ExitKind,
    failure: z3.BoolRef,
    handover: State | None = None,
) -> State:
    return allocation_exit(state, uop.i, uop.op, kind, uop.target, failure, handover)


def push_constant(uop: Uop, state: State, c: Constant, borrowed: bool) -> State:
    ident, state = constant_facts(state, c)
    tag = borrowed or c.immortal
    slot = Slot.Ref(ident, z3.BoolVal(tag))
    if not tag:
        state = replace(state, ledger=incref_on_dup(state, slot))
    return push(state, slot)


def new_ref(state: State, ident: z3.ArithRef) -> tuple[z3.DatatypeRef, State]:
    borrowed = is_immortal_id(state.consts, ident)
    slot = Slot.Ref(ident, borrowed)
    return slot, replace(state, ledger=ledger_add(state.ledger, z3.Not(borrowed), ident, -1))
