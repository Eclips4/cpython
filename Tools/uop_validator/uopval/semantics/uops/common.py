from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace

import z3

from ...domain.trace import Uop
from ...symbolic.escape import escape
from ...symbolic.heap import expose
from ...symbolic.ledger import ledger_add
from ...symbolic.objects import is_compact, is_immortal_id, slot_int_value, slot_is_exact_int
from ...symbolic.sorts import Slot
from ...symbolic.state import ExitKind, State, push
from ..step import exit_if, precondition
from ..uninterpreted import unknown_error

COMPARE_UNORDERED = 1
COMPARE_LT = 2
COMPARE_GT = 4
COMPARE_EQ = 8

SlotTest = Callable[[State, z3.DatatypeRef], z3.BoolRef]


@dataclass(frozen=True)
class CompactOperands:
    left: z3.DatatypeRef
    right: z3.DatatypeRef
    left_value: z3.ArithRef
    right_value: z3.ArithRef


def require_compact_ints(
    uop: Uop, state: State, left: z3.DatatypeRef, right: z3.DatatypeRef
) -> tuple[CompactOperands, State]:
    left_value, right_value = slot_int_value(state, left), slot_int_value(state, right)
    state = precondition(
        uop,
        state,
        z3.And(
            slot_is_exact_int(state, left),
            slot_is_exact_int(state, right),
            is_compact(state.consts, left_value),
            is_compact(state.consts, right_value),
        ),
        "both operands are compact exact ints",
    )
    return CompactOperands(left, right, left_value, right_value), state


def push_all(state: State, slots: Iterable[z3.DatatypeRef]) -> State:
    for slot in slots:
        state = push(state, slot)
    return state


def raising_escape(
    uop: Uop, state: State, exposed: list[z3.DatatypeRef], when: z3.BoolRef | None = None
) -> tuple[z3.BoolRef, State]:
    state = expose(state, exposed, when)
    raises, state = escape(state, uop.i, uop.op, z3.BoolVal(True) if when is None else when, True)
    assert raises is not None
    return raises, state


def escaping_call(
    uop: Uop,
    state: State,
    before: State,
    operands: Sequence[z3.DatatypeRef],
    exposed: list[z3.DatatypeRef] | None = None,
    when: z3.BoolRef | None = None,
) -> State:
    raises, state = raising_escape(uop, state, list(operands) if exposed is None else exposed, when)
    failure = unknown_error(state, uop, list(operands))
    error = z3.Or(raises, failure if when is None else z3.And(when, failure))
    return exit_if(uop, state, ExitKind.ERROR, error, handover=before)


def take_ownership(state: State, slot: z3.DatatypeRef) -> State:
    mortal_borrowed = z3.And(
        Slot.is_Ref(slot), Slot.borrowed(slot), z3.Not(is_immortal_id(state.consts, Slot.id(slot)))
    )
    return replace(state, ledger=ledger_add(state.ledger, mortal_borrowed, Slot.id(slot), -1))
