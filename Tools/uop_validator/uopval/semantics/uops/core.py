from collections.abc import Callable
from dataclasses import replace

import z3

from ...domain.trace import Constant, Uop
from ...symbolic.escape import escape
from ...symbolic.heap import new_int
from ...symbolic.identities import FALSE_ID, TRUE_ID
from ...symbolic.ledger import decref_on_pop, decref_on_pop_if, incref_on_dup, ledger_add
from ...symbolic.objects import (
    bool_of,
    bool_slot,
    is_compact,
    is_immortal_id,
    is_range_iter,
    object_of,
    slot_int_value,
    slot_is,
    slot_is_bool,
    slot_is_exact_int,
    slot_is_none,
    slot_kind_is,
)
from ...symbolic.protocols import harmless_to_close
from ...symbolic.sorts import Obj, Slot
from ...symbolic.state import ExitKind, State, pop, push, set_local
from ...symbolic.summaries import type_facts, type_tag_of_summary
from ...symbolic.types import type_tag_of
from ..step import (
    UnsupportedUop,
    allocation_exit_if,
    at_depth,
    exit_if,
    local_slot,
    nos,
    precondition,
    push_constant,
    tos,
)
from .common import (
    COMPARE_EQ,
    COMPARE_GT,
    COMPARE_LT,
    CompactOperands,
    push_all,
    require_compact_ints,
)


def no_effect(uop: Uop, state: State) -> State:
    return state


def set_ip(uop: Uop, state: State) -> State:
    return replace(state, ip=z3.IntVal(uop.operand0 // 2))


def deopt_if_invalid(uop: Uop, state: State) -> State:
    return exit_if(uop, state, ExitKind.DEOPT, z3.Not(state.valid))


def check_periodic(uop: Uop, state: State) -> State:
    raises, state = escape(state, uop.i, uop.op, z3.BoolVal(True), True)
    assert raises is not None
    return exit_if(uop, state, ExitKind.ERROR, raises)


def tier2_resume_check(uop: Uop, state: State) -> State:
    pending = z3.Bool(f"eval_breaker_p{uop.i}")
    state = replace(state, oracle=state.oracle + (pending,))
    return exit_if(uop, state, ExitKind.DEOPT, pending)


def iter_check_range(uop: Uop, state: State) -> State:
    iterator = nos(state, uop)
    state = precondition(uop, state, Slot.is_Ref(iterator), "NOS is an object (E2)")
    return exit_if(uop, state, ExitKind.SIDE, z3.Not(is_range_iter(state, Slot.id(iterator))))


def guard_not_exhausted_range(uop: Uop, state: State) -> State:
    iterator = nos(state, uop)
    state = precondition(
        uop,
        state,
        z3.And(Slot.is_Ref(iterator), is_range_iter(state, Slot.id(iterator))),
        "NOS is a range iterator",
    )
    remaining = Obj.len(object_of(state, Slot.id(iterator)))
    return exit_if(uop, state, ExitKind.SIDE, remaining <= 0)


def guard_int(slot: z3.DatatypeRef, uop: Uop, state: State) -> State:
    compact_int = z3.And(
        slot_is_exact_int(state, slot), is_compact(state.consts, slot_int_value(state, slot))
    )
    return exit_if(uop, state, ExitKind.SIDE, z3.Not(compact_int))


def guard_overflowed(slot: z3.DatatypeRef, uop: Uop, state: State) -> State:
    state = precondition(uop, state, slot_is_exact_int(state, slot), "slot is an exact int")
    compact = is_compact(state.consts, slot_int_value(state, slot))
    return exit_if(uop, state, ExitKind.SIDE, z3.Not(compact))


def guard_tos_int(uop: Uop, state: State) -> State:
    return guard_int(tos(state, uop), uop, state)


def guard_nos_int(uop: Uop, state: State) -> State:
    return guard_int(nos(state, uop), uop, state)


def guard_tos_overflowed(uop: Uop, state: State) -> State:
    return guard_overflowed(tos(state, uop), uop, state)


def guard_nos_overflowed(uop: Uop, state: State) -> State:
    return guard_overflowed(nos(state, uop), uop, state)


def guard_tos_exact_int(uop: Uop, state: State) -> State:
    return exit_if(uop, state, ExitKind.SIDE, z3.Not(slot_is_exact_int(state, tos(state, uop))))


def load_fast_borrow(uop: Uop, state: State) -> State:
    slot = local_slot(uop, state)
    state = precondition(uop, state, Slot.is_Ref(slot), f"local {uop.oparg} is bound (E3)")
    return push(state, Slot.Ref(Slot.id(slot), z3.BoolVal(True)))


def load_fast(uop: Uop, state: State) -> State:
    slot = local_slot(uop, state)
    state = precondition(uop, state, Slot.is_Ref(slot), f"local {uop.oparg} is bound (E3)")
    state = replace(state, ledger=incref_on_dup(state, slot))
    return push(state, slot)


def load_fast_check(uop: Uop, state: State) -> State:
    slot = local_slot(uop, state)
    state = exit_if(uop, state, ExitKind.ERROR, Slot.is_Null(slot))
    state = replace(state, ledger=incref_on_dup(state, slot))
    return push(state, slot)


def load_fast_and_clear(uop: Uop, state: State) -> State:
    slot = local_slot(uop, state)
    return push(set_local(state, uop.oparg, Slot.Null), slot)


def recorded_constant(uop: Uop) -> Constant:
    if uop.constant is None:
        raise UnsupportedUop(f"{uop.op} at {uop.i}: no constant recorded")
    return uop.constant


def load_const_inline_borrow(uop: Uop, state: State) -> State:
    consts = state.consts
    if uop.op == "_LOAD_SMALL_INT":
        if not consts.small_min <= uop.oparg <= consts.small_max:
            raise UnsupportedUop(f"{uop.op} at {uop.i}: {uop.oparg} is outside the small-int cache")
        return push(state, Slot.Ref(z3.IntVal(uop.oparg), z3.BoolVal(True)))
    return push_constant(uop, state, recorded_constant(uop), True)


def load_const_inline(uop: Uop, state: State) -> State:
    return push_constant(uop, state, recorded_constant(uop), False)


def load_common_constant(uop: Uop, state: State) -> State:
    return push_constant(uop, state, recorded_constant(uop), True)


def swap_fast(uop: Uop, state: State) -> State:
    top, state = pop(state)
    old = local_slot(uop, state)
    return push(set_local(state, uop.oparg, top), old)


def rrot_3(uop: Uop, state: State) -> State:
    if len(state.stack) < 3:
        raise UnsupportedUop(f"{uop.op} at {uop.i}: stack too shallow")
    top, state = pop(state)
    middle, state = pop(state)
    bottom, state = pop(state)
    return push(push(push(state, top), bottom), middle)


def copy(uop: Uop, state: State) -> State:
    if uop.oparg < 1:
        raise UnsupportedUop(f"{uop.op} at {uop.i}: oparg {uop.oparg}")
    bottom = at_depth(state, uop, uop.oparg)
    state = precondition(uop, state, z3.Not(Slot.is_Null(bottom)), "copied slot is not NULL")
    state = replace(state, ledger=incref_on_dup(state, bottom))
    return push(state, bottom)


def swap(uop: Uop, state: State) -> State:
    if uop.oparg < 2 or len(state.stack) < uop.oparg:
        raise UnsupportedUop(
            f"{uop.op} at {uop.i}: oparg {uop.oparg} with depth {len(state.stack)}"
        )
    stack = list(state.stack)
    stack[-1], stack[-uop.oparg] = stack[-uop.oparg], stack[-1]
    return replace(state, stack=tuple(stack))


def push_null(uop: Uop, state: State) -> State:
    return push(state, Slot.Null)


def push_tagged_zero(uop: Uop, state: State) -> State:
    return push(state, Slot.TaggedInt(z3.IntVal(0)))


def push_null_conditional(uop: Uop, state: State) -> State:
    return push(state, Slot.Null) if uop.oparg & 1 else state


def insert_null(uop: Uop, state: State) -> State:
    self_, state = pop(state)
    return push(push(state, Slot.Null), self_)


def guard_null_at(depth: int, uop: Uop, state: State) -> State:
    return exit_if(uop, state, ExitKind.SIDE, z3.Not(Slot.is_Null(at_depth(state, uop, depth))))


def guard_not_null_at(depth: int, uop: Uop, state: State) -> State:
    return exit_if(uop, state, ExitKind.SIDE, Slot.is_Null(at_depth(state, uop, depth)))


def guard_nos_null(uop: Uop, state: State) -> State:
    return guard_null_at(2, uop, state)


def guard_third_null(uop: Uop, state: State) -> State:
    return guard_null_at(3, uop, state)


def guard_nos_not_null(uop: Uop, state: State) -> State:
    return guard_not_null_at(2, uop, state)


def guard_tos_not_null(uop: Uop, state: State) -> State:
    return guard_not_null_at(1, uop, state)


def pop_top(uop: Uop, state: State) -> State:
    slot, state = pop(state)
    state = replace(state, ledger=decref_on_pop(state, slot))
    _, state = escape(state, uop.i, uop.op, z3.Not(harmless_to_close(state, slot)), False)
    return state


def pop_top_oparg(uop: Uop, state: State) -> State:
    for _ in range(uop.oparg):
        state = pop_top(uop, state)
    return state


def pop_top_kind(
    uop: Uop, state: State, test: Callable[[State, z3.DatatypeRef], z3.BoolRef], note: str
) -> State:
    slot, state = pop(state)
    state = precondition(uop, state, test(state, slot), note)
    return replace(state, ledger=decref_on_pop(state, slot))


def pop_top_int(uop: Uop, state: State) -> State:
    return pop_top_kind(uop, state, slot_is_exact_int, "popped slot is an exact int")


def pop_top_float(uop: Uop, state: State) -> State:
    return pop_top_kind(
        uop,
        state,
        lambda s, slot: slot_kind_is(s, slot, Obj.is_Float),
        "popped slot is an exact float",
    )


def pop_top_unicode(uop: Uop, state: State) -> State:
    return pop_top_kind(
        uop, state, lambda s, slot: slot_kind_is(s, slot, Obj.is_Str), "popped slot is an exact str"
    )


def pop_top_nop(uop: Uop, state: State) -> State:
    slot, state = pop(state)
    no_decref = z3.Or(
        Slot.is_Null(slot),
        z3.And(Slot.is_Ref(slot), Slot.borrowed(slot)),
        z3.And(Slot.is_Ref(slot), is_immortal_id(state.consts, Slot.id(slot))),
    )
    return precondition(
        uop, state, no_decref, "popped slot is null, borrowed, or immortal: no decref"
    )


def make_heap_safe(uop: Uop, state: State) -> State:
    slot, state = pop(state)
    ident = Slot.id(slot)
    mortal_borrowed = z3.And(
        Slot.is_Ref(slot), Slot.borrowed(slot), z3.Not(is_immortal_id(state.consts, ident))
    )
    owned = Slot.Ref(ident, z3.BoolVal(False))
    ledger = ledger_add(state.ledger, mortal_borrowed, ident, -1)
    return push(replace(state, ledger=ledger), z3.If(mortal_borrowed, owned, slot))


def iter_next_range(uop: Uop, state: State) -> State:
    iterator = nos(state, uop)
    ident = Slot.id(iterator)
    state = precondition(
        uop,
        state,
        z3.And(Slot.is_Ref(iterator), is_range_iter(state, ident)),
        "NOS is a range iterator",
    )
    range_iter = object_of(state, ident)
    start, step = Obj.start(range_iter), Obj.step(range_iter)
    advanced = Obj.RangeIter(start + step, step, Obj.len(range_iter) - 1)
    state = replace(state, heap=z3.Store(state.heap, ident, advanced))
    slot, failure, state = new_int(state, start, uop.i)
    state = allocation_exit_if(uop, state, ExitKind.ERROR, failure)
    return push(state, slot)


def pop_compact_ints(uop: Uop, state: State) -> tuple[CompactOperands, State]:
    right, state = pop(state)
    left, state = pop(state)
    return require_compact_ints(uop, state, left, right)


def binary_op_int(
    uop: Uop, state: State, combine: Callable[[z3.ArithRef, z3.ArithRef], z3.ArithRef]
) -> State:
    before = state
    operands, state = pop_compact_ints(uop, state)
    total = combine(operands.left_value, operands.right_value)
    result, failure, state = new_int(state, total, uop.i)
    overflows = z3.Not(is_compact(state.consts, total))
    state = exit_if(uop, state, ExitKind.SIDE, overflows, handover=before)
    state = allocation_exit_if(uop, state, ExitKind.SIDE, failure, handover=before)
    return push_all(state, [result, operands.left, operands.right])


def binary_op_add_int(uop: Uop, state: State) -> State:
    return binary_op_int(uop, state, lambda a, b: a + b)


def binary_op_subtract_int(uop: Uop, state: State) -> State:
    return binary_op_int(uop, state, lambda a, b: a - b)


def binary_op_multiply_int(uop: Uop, state: State) -> State:
    return binary_op_int(uop, state, lambda a, b: a * b)


def compare_truth(oparg: int, a: z3.ArithRef, b: z3.ArithRef) -> z3.BoolRef:
    parts: list[z3.BoolRef] = []
    if oparg & COMPARE_LT:
        parts.append(a < b)
    if oparg & COMPARE_GT:
        parts.append(a > b)
    if oparg & COMPARE_EQ:
        parts.append(a == b)
    if not parts:
        return z3.BoolVal(False)
    return z3.Or(*parts) if len(parts) > 1 else parts[0]


def compare_op_int(uop: Uop, state: State) -> State:
    operands, state = pop_compact_ints(uop, state)
    result = bool_of(compare_truth(uop.oparg, operands.left_value, operands.right_value))
    return push_all(state, [result, operands.left, operands.right])


def to_bool_bool(uop: Uop, state: State) -> State:
    return exit_if(uop, state, ExitKind.SIDE, z3.Not(slot_is_bool(tos(state, uop))))


def unary_not(uop: Uop, state: State) -> State:
    flag, state = pop(state)
    state = precondition(uop, state, slot_is_bool(flag), "operand is a bool")
    return push(state, bool_of(slot_is(flag, FALSE_ID)))


def replace_with_true(uop: Uop, state: State) -> State:
    value, state = pop(state)
    return push(push(state, bool_slot(True)), value)


def to_bool_int(uop: Uop, state: State) -> State:
    value, state = pop(state)
    state = precondition(uop, state, slot_is_exact_int(state, value), "operand is an exact int")
    return push(push(state, bool_of(slot_int_value(state, value) != 0)), value)


def is_op(uop: Uop, state: State) -> State:
    right, state = pop(state)
    left, state = pop(state)
    same = z3.Or(
        z3.And(Slot.is_Ref(left), Slot.is_Ref(right), Slot.id(left) == Slot.id(right)),
        z3.And(Slot.is_TaggedInt(left), Slot.is_TaggedInt(right), Slot.k(left) == Slot.k(right)),
    )
    truth = z3.Not(same) if uop.oparg else same
    state = push(state, bool_of(truth))
    state = push(state, left)
    return push(state, right)


def pop_and_close_unless_none(uop: Uop, state: State) -> tuple[z3.BoolRef, State]:
    value, state = pop(state)
    none = slot_is_none(value)
    state = replace(state, ledger=decref_on_pop_if(state, z3.Not(none), value))
    closes_object = z3.And(z3.Not(none), z3.Not(harmless_to_close(state, value)))
    _, state = escape(state, uop.i, uop.op, closes_object, False)
    return none, state


def is_none(uop: Uop, state: State) -> State:
    none, state = pop_and_close_unless_none(uop, state)
    return push(state, bool_of(none))


def guard_is_bool_pop(uop: Uop, state: State, expected: int) -> State:
    flag, state = pop(state)
    return exit_if(uop, state, ExitKind.SIDE, z3.Not(slot_is(flag, expected)))


def guard_is_true_pop(uop: Uop, state: State) -> State:
    return guard_is_bool_pop(uop, state, TRUE_ID)


def guard_is_false_pop(uop: Uop, state: State) -> State:
    return guard_is_bool_pop(uop, state, FALSE_ID)


def guard_is_none_pop(uop: Uop, state: State) -> State:
    none, state = pop_and_close_unless_none(uop, state)
    return exit_if(uop, state, ExitKind.SIDE, z3.Not(none))


def guard_is_not_none_pop(uop: Uop, state: State) -> State:
    none, state = pop_and_close_unless_none(uop, state)
    return exit_if(uop, state, ExitKind.SIDE, none)


def guard_bit_pop(uop: Uop, state: State, exit_when_set: bool) -> State:
    try:
        true_bit = state.consts.bool_bit(uop.oparg, True)
        false_bit = state.consts.bool_bit(uop.oparg, False)
    except ValueError as e:
        raise UnsupportedUop(f"{uop.op} at {uop.i}: {e}") from None
    flag, state = pop(state)
    state = precondition(
        uop, state, slot_is_bool(flag), "popped slot is a bool: the address bit test is defined"
    )
    exits_true = true_bit == exit_when_set
    exits_false = false_bit == exit_when_set
    condition = z3.Or(
        z3.And(slot_is(flag, TRUE_ID), z3.BoolVal(exits_true)),
        z3.And(slot_is(flag, FALSE_ID), z3.BoolVal(exits_false)),
    )
    return exit_if(uop, state, ExitKind.SIDE, condition)


def guard_bit_is_set_pop(uop: Uop, state: State) -> State:
    return guard_bit_pop(uop, state, False)


def guard_bit_is_unset_pop(uop: Uop, state: State) -> State:
    return guard_bit_pop(uop, state, True)


def guard_type(uop: Uop, state: State) -> State:
    expected = uop.operand0_object
    if expected is None or expected.type_name is None:
        raise UnsupportedUop(f"{uop.op} at {uop.i}: no type recorded for the operand")
    owner = tos(state, uop)
    state = precondition(uop, state, Slot.is_Ref(owner), "owner is an object")
    state = replace(state, constraints=state.constraints + tuple(type_facts(expected)))
    other_type = type_tag_of(state, Slot.id(owner)) != type_tag_of_summary(expected)
    return exit_if(uop, state, ExitKind.SIDE, other_type)


HANDLERS: dict[str, Callable[[Uop, State], State]] = {
    "_START_EXECUTOR": deopt_if_invalid,
    "_MAKE_WARM": no_effect,
    "_NOP": no_effect,
    "_SET_IP": set_ip,
    "_CHECK_VALIDITY": deopt_if_invalid,
    "_CHECK_PERIODIC": check_periodic,
    "_TIER2_RESUME_CHECK": tier2_resume_check,
    "_CHECK_PEP_523": deopt_if_invalid,
    "_ITER_CHECK_RANGE": iter_check_range,
    "_GUARD_NOT_EXHAUSTED_RANGE": guard_not_exhausted_range,
    "_GUARD_TOS_INT": guard_tos_int,
    "_GUARD_NOS_INT": guard_nos_int,
    "_GUARD_TOS_OVERFLOWED": guard_tos_overflowed,
    "_GUARD_NOS_OVERFLOWED": guard_nos_overflowed,
    "_GUARD_TOS_EXACT_INT": guard_tos_exact_int,
    "_LOAD_FAST_BORROW": load_fast_borrow,
    "_LOAD_FAST": load_fast,
    "_LOAD_FAST_CHECK": load_fast_check,
    "_LOAD_FAST_AND_CLEAR": load_fast_and_clear,
    "_LOAD_SMALL_INT": load_const_inline_borrow,
    "_LOAD_CONST_INLINE_BORROW": load_const_inline_borrow,
    "_LOAD_CONST_INLINE": load_const_inline,
    "_LOAD_CONST": load_const_inline_borrow,
    "_LOAD_COMMON_CONSTANT": load_common_constant,
    "_SWAP_FAST": swap_fast,
    "_RROT_3": rrot_3,
    "_COPY": copy,
    "_SWAP": swap,
    "_PUSH_NULL": push_null,
    "_PUSH_TAGGED_ZERO": push_tagged_zero,
    "_PUSH_NULL_CONDITIONAL": push_null_conditional,
    "_INSERT_NULL": insert_null,
    "_GUARD_NOS_NULL": guard_nos_null,
    "_GUARD_THIRD_NULL": guard_third_null,
    "_GUARD_NOS_NOT_NULL": guard_nos_not_null,
    "_GUARD_TOS_NOT_NULL": guard_tos_not_null,
    "_POP_TOP": pop_top,
    "_POP_TOP_OPARG": pop_top_oparg,
    "_POP_TOP_INT": pop_top_int,
    "_POP_TOP_FLOAT": pop_top_float,
    "_POP_TOP_UNICODE": pop_top_unicode,
    "_POP_TOP_NOP": pop_top_nop,
    "_MAKE_HEAP_SAFE": make_heap_safe,
    "_ITER_NEXT_RANGE": iter_next_range,
    "_BINARY_OP_ADD_INT": binary_op_add_int,
    "_BINARY_OP_SUBTRACT_INT": binary_op_subtract_int,
    "_BINARY_OP_MULTIPLY_INT": binary_op_multiply_int,
    "_COMPARE_OP_INT": compare_op_int,
    "_TO_BOOL_BOOL": to_bool_bool,
    "_TO_BOOL_INT": to_bool_int,
    "_UNARY_NOT": unary_not,
    "_REPLACE_WITH_TRUE": replace_with_true,
    "_IS_OP": is_op,
    "_IS_NONE": is_none,
    "_GUARD_IS_TRUE_POP": guard_is_true_pop,
    "_GUARD_IS_FALSE_POP": guard_is_false_pop,
    "_GUARD_IS_NONE_POP": guard_is_none_pop,
    "_GUARD_IS_NOT_NONE_POP": guard_is_not_none_pop,
    "_GUARD_BIT_IS_SET_POP": guard_bit_is_set_pop,
    "_GUARD_BIT_IS_UNSET_POP": guard_bit_is_unset_pop,
    "_GUARD_TYPE": guard_type,
}
