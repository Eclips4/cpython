from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from enum import StrEnum

import z3

from ..domain.build import IntConstants
from ..domain.trace import CodeSummary
from .sorts import HeapSort, Slot


class ExitKind(StrEnum):
    SIDE = "side"
    DEOPT = "deopt"
    ERROR = "error"
    DYNAMIC = "dynamic"


GUARD_EXITS = frozenset({ExitKind.SIDE, ExitKind.DEOPT})


class FactKind(StrEnum):
    PRECONDITION = "precondition"
    ESCAPE = "escape"
    ALLOCATION = "allocation"
    TERMINATOR = "terminator"
    OPAQUE = "opaque"
    FRAME = "frame"


@dataclass(frozen=True)
class Fact:
    position: int
    kind: FactKind
    uop: str
    expr: z3.BoolRef | None
    note: str


LedgerEntry = tuple[z3.BoolRef, z3.ArithRef, int]


@dataclass(frozen=True)
class State:
    consts: IntConstants
    locals: tuple[z3.DatatypeRef, ...]
    stack: tuple[z3.DatatypeRef, ...]
    ip: z3.ArithRef
    valid: z3.BoolRef
    heap: z3.ArrayRef
    ledger: tuple[LedgerEntry, ...]
    ids: tuple[z3.ArithRef, ...]
    fresh_count: int
    facts: tuple[Fact, ...]
    constraints: tuple[z3.BoolRef, ...]
    code: z3.ArithRef
    stack_space: z3.ArithRef
    recursion_remaining: z3.ArithRef
    globals_version: z3.ArithRef
    builtins_version: z3.ArithRef
    path: tuple[z3.BoolRef, ...] = ()
    exits: tuple[Exit, ...] = ()
    oracle: tuple[z3.ExprRef, ...] = ()
    escape_count: int = 0
    alloc_count: int = 0
    fresh_ids: tuple[z3.ArithRef, ...] = ()
    exposed: tuple[tuple[z3.DatatypeRef, z3.BoolRef], ...] = ()
    epoch: z3.ArithRef = z3.IntVal(0)
    callers: tuple[Frame, ...] = ()
    return_offset: z3.ArithRef | None = None
    pending_frame: Frame | None = None
    frame_count: int = 0
    frame_id: int = 0
    codes: Mapping[int, CodeSummary] = field(default_factory=dict)
    func: z3.DatatypeRef = Slot.Null
    c_recursion_checked: bool = False
    popped: tuple[z3.DatatypeRef, ...] = ()


@dataclass(frozen=True)
class Frame:
    code: z3.ArithRef
    locals: tuple[z3.DatatypeRef, ...]
    stack: tuple[z3.DatatypeRef, ...]
    ip: z3.ArithRef
    return_offset: z3.ArithRef | None
    func: z3.DatatypeRef
    c_recursion_checked: bool = False
    globals_version: z3.ArithRef | None = None
    frame_id: int = 0


@dataclass(frozen=True)
class Exit:
    position: int
    uop: str
    kind: ExitKind
    target: int | None
    condition: z3.BoolRef
    state: State
    path: tuple[z3.BoolRef, ...]
    environmental: bool = False


def entry_state(
    c: IntConstants,
    nlocals: int,
    depth: int,
    ip0: int,
    code: int | None = None,
    codes: Mapping[int, CodeSummary] | None = None,
) -> State:
    locals_ = tuple(z3.Const(f"local{j}", Slot) for j in range(nlocals))
    stack = tuple(z3.Const(f"stack{k}", Slot) for k in range(depth))
    ids = tuple(Slot.id(s) for s in locals_ + stack)
    return State(
        consts=c,
        locals=locals_,
        stack=stack,
        ip=z3.IntVal(ip0),
        valid=z3.BoolVal(True),
        heap=z3.Const("heap0", HeapSort),
        ledger=(),
        ids=ids,
        fresh_count=0,
        facts=(),
        constraints=(),
        code=z3.IntVal(code) if code is not None else z3.Int("root_code"),
        stack_space=z3.Int("stack_space0"),
        recursion_remaining=z3.Int("recursion_remaining0"),
        globals_version=z3.Int("globals_version_f0"),
        builtins_version=z3.Int("builtins_version0"),
        codes=dict(codes) if codes is not None else {},
        func=z3.Const("root_func", Slot),
    )


def with_fact(
    state: State, position: int, kind: FactKind, uop: str, expr: z3.BoolRef | None, note: str
) -> State:
    return replace(state, facts=state.facts + (Fact(position, kind, uop, expr, note),))


def with_constraint(state: State, *constraints: z3.BoolRef) -> State:
    return replace(state, constraints=state.constraints + tuple(constraints))


def set_local(state: State, index: int, slot: z3.DatatypeRef) -> State:
    locals_ = list(state.locals)
    locals_[index] = slot
    return replace(state, locals=tuple(locals_))


def set_stack(state: State, depth: int, slot: z3.DatatypeRef) -> State:
    stack = list(state.stack)
    stack[-depth] = slot
    return replace(state, stack=tuple(stack))


def push(state: State, slot: z3.DatatypeRef) -> State:
    return replace(state, stack=state.stack + (slot,))


def pop(state: State) -> tuple[z3.DatatypeRef, State]:
    if not state.stack:
        raise ValueError("pop from an empty symbolic stack")
    return state.stack[-1], replace(
        state, stack=state.stack[:-1], popped=state.popped + (state.stack[-1],)
    )


def all_slots(state: State) -> list[z3.DatatypeRef]:
    slots = list(state.locals) + list(state.stack)
    for frame in state.callers:
        slots += list(frame.locals) + list(frame.stack)
    if state.pending_frame is not None:
        slots += list(state.pending_frame.locals) + list(state.pending_frame.stack)
    return slots


def add_exit(
    state: State,
    position: int,
    uop: str,
    kind: ExitKind,
    target: int | None,
    condition: z3.BoolRef,
    handover: State | None = None,
    environmental: bool = False,
) -> State:
    exit_ = Exit(
        position,
        uop,
        kind,
        target,
        condition,
        handover if handover is not None else state,
        state.path,
        environmental,
    )
    return replace(state, exits=state.exits + (exit_,), path=state.path + (z3.Not(condition),))


def allocation_exit(
    state: State,
    position: int,
    uop: str,
    kind: ExitKind,
    target: int | None,
    failure: z3.BoolRef,
    handover: State | None = None,
) -> State:
    return add_exit(state, position, uop, kind, target, failure, handover, environmental=True)
