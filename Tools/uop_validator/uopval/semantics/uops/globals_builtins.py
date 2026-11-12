from collections.abc import Callable
from dataclasses import replace

import z3

from ...domain.trace import Uop
from ...symbolic.escape import escape
from ...symbolic.functions import (
    descr_d_type,
    descr_ml_flags,
    descr_ml_meth,
    frozen_member,
    frozen_members_known,
    frozen_value,
)
from ...symbolic.heap import expose, is_unique, new_int
from ...symbolic.identities import DICT_TAG, FROZENDICT_TAG, METHOD_DESCRIPTOR_TAG
from ...symbolic.ledger import decref_on_pop
from ...symbolic.objects import (
    borrowed_if_immortal,
    is_compact,
    is_immortal_id,
    is_small_value,
    plain_kind,
)
from ...symbolic.protocols import harmless_to_close
from ...symbolic.sorts import Slot
from ...symbolic.state import ExitKind, State, push, with_constraint
from ...symbolic.summaries import constant_facts
from ...symbolic.types import has_tag, type_tag_of
from ..step import (
    Handler,
    UnsupportedUop,
    allocation_exit_if,
    at_depth,
    exit_if,
    new_ref,
    pop_n,
    precondition,
)
from ..uninterpreted import unknown_error, unknown_result, value_projection
from .common import escaping_call, push_all, raising_escape, require_compact_ints

global_value = z3.Function("global_value", z3.IntSort(), z3.IntSort(), z3.IntSort(), Slot)
builtin_value = z3.Function("builtin_value", z3.IntSort(), z3.IntSort(), Slot)
METH_O = 0x8
METH_NOARGS = 0x4
METH_FASTCALL = 0x80
METH_KEYWORDS = 0x2


def guard_globals_version(uop: Uop, state: State) -> State:
    return exit_if(uop, state, ExitKind.DEOPT, state.globals_version != z3.IntVal(uop.operand0))


def captured_global_fact(state: State, uop: Uop, value: z3.DatatypeRef) -> State:
    captured = uop.global_value
    if captured is None:
        return state
    try:
        ident, state = constant_facts(state, captured)
    except ValueError:
        return state
    return with_constraint(
        state, z3.Implies(state.valid, value == Slot.Ref(ident, z3.BoolVal(True)))
    )


def load_global_module(uop: Uop, state: State) -> State:
    before = state
    version = z3.IntVal(uop.operand0)
    frame_key = z3.IntVal(state.frame_count)
    value = global_value(frame_key, version, z3.IntVal(uop.operand1))
    state = captured_global_fact(state, uop, value)
    if uop.global_value is not None:
        state = with_constraint(state, z3.Implies(state.valid, state.globals_version == version))
    condition = z3.Or(state.globals_version != version, Slot.is_Null(value))
    state = exit_if(uop, state, ExitKind.DEOPT, condition, handover=before)
    slot, state = new_ref(state, Slot.id(value))
    return push(state, slot)


def load_global_builtins(uop: Uop, state: State) -> State:
    before = state
    version = z3.IntVal(uop.operand0)
    value = builtin_value(version, z3.IntVal(uop.operand1))
    state = captured_global_fact(state, uop, value)
    if uop.global_value is not None:
        state = with_constraint(state, z3.Implies(state.valid, state.builtins_version == version))
    condition = z3.Or(state.builtins_version != version, Slot.is_Null(value))
    state = exit_if(uop, state, ExitKind.DEOPT, condition, handover=before)
    slot, state = new_ref(state, Slot.id(value))
    return push(state, slot)


def int_inplace(
    combine: Callable[[z3.ArithRef, z3.ArithRef], z3.ArithRef], target_is_left: bool
) -> Handler:
    def handler(uop: Uop, state: State) -> State:
        consts = state.consts
        before = state
        (left, right), state = pop_n(state, uop, 2)
        operands, state = require_compact_ints(uop, state, left, right)
        left_value, right_value = operands.left_value, operands.right_value
        target = left if target_is_left else right
        mortal = z3.Not(is_immortal_id(consts, Slot.id(target)))
        state = precondition(
            uop,
            state,
            z3.Implies(mortal, is_unique(state, target, [left, right])),
            "the mutated operand is uniquely referenced unless immortal",
        )
        total = combine(left_value, right_value)
        in_place = z3.And(mortal, z3.Not(is_small_value(consts, total)), is_compact(consts, total))
        result, failure, state = new_int(state, total, uop.i)
        overflows = z3.Not(is_compact(consts, total))
        state = exit_if(uop, state, ExitKind.SIDE, overflows, handover=before)
        allocation_failed = z3.And(z3.Not(in_place), failure)
        state = allocation_exit_if(uop, state, ExitKind.SIDE, allocation_failed, handover=before)
        return push_all(state, [result, left, right])

    return handler


def dict_subscript_generic(uop: Uop, state: State) -> State:
    before = state
    (mapping, key), state = pop_n(state, uop, 2)
    state = precondition(
        uop,
        state,
        z3.Or(has_tag(state, mapping, DICT_TAG), has_tag(state, mapping, FROZENDICT_TAG)),
        "NOS is a dict",
    )
    getitem = z3.Function("dict_getitem", z3.IntSort(), Slot, state.heap.sort(), z3.IntSort(), Slot)
    generic_value = getitem(Slot.id(mapping), value_projection(key), state.heap, state.epoch)
    frozen_case = z3.And(
        frozen_members_known(Slot.id(mapping)), Slot.is_Ref(key), plain_kind(state, key)
    )
    hit = frozen_member(Slot.id(mapping), Slot.id(key))
    frozen_slot = Slot.Ref(frozen_value(Slot.id(mapping), Slot.id(key)), z3.BoolVal(True))
    value = z3.If(frozen_case, z3.If(hit, frozen_slot, Slot.Null), generic_value)
    state = with_constraint(
        state,
        borrowed_if_immortal(state, value),
    )
    state = expose(state, [key], z3.Not(frozen_case))
    raises, state = escape(state, uop.i, uop.op, z3.Not(frozen_case), True)
    assert raises is not None
    error = z3.Or(
        raises,
        Slot.is_Null(value),
        z3.And(z3.Not(frozen_case), unknown_error(state, uop, [mapping, key])),
    )
    state = exit_if(uop, state, ExitKind.ERROR, error, handover=before)
    slot, state = new_ref(state, Slot.id(value))
    return push_all(state, [slot, mapping, key])


def store_subscr_dict_generic(uop: Uop, state: State) -> State:
    before = state
    (value, mapping, key), state = pop_n(state, uop, 3)
    state = precondition(
        uop, state, has_tag(state, mapping, DICT_TAG), "the container is an exact dict"
    )
    state = escaping_call(uop, state, before, [value, mapping, key], exposed=[value, key])
    return push(state, mapping)


def descriptor_facts(state: State, uop: Uop) -> State:
    for summary in (uop.operand0_object, uop.operand1_object, uop.recorded):
        if summary is not None and summary.type == "method_descriptor" and summary.addr is not None:
            try:
                _, state = constant_facts(state, summary)
            except ValueError:
                continue
    return state


def guard_callable_method_descriptor(flags: int, total: int | None) -> Handler:
    def handler(uop: Uop, state: State) -> State:
        callable_ = at_depth(state, uop, uop.oparg + 2)
        self_or_null = at_depth(state, uop, uop.oparg + 1)
        state = descriptor_facts(state, uop)
        ident = Slot.id(callable_)
        total_args = z3.IntVal(uop.oparg) + z3.If(
            Slot.is_Null(self_or_null), z3.IntVal(0), z3.IntVal(1)
        )
        first = z3.If(
            Slot.is_Null(self_or_null),
            at_depth(state, uop, uop.oparg) if uop.oparg >= 1 else self_or_null,
            self_or_null,
        )
        matches = z3.And(
            has_tag(state, callable_, METHOD_DESCRIPTOR_TAG),
            descr_ml_flags(ident) == z3.IntVal(flags),
            total_args == z3.IntVal(total) if total is not None else total_args != 0,
            Slot.is_Ref(first),
            type_tag_of(state, Slot.id(first)) == descr_d_type(ident),
        )
        return exit_if(uop, state, ExitKind.SIDE, z3.Not(matches))

    return handler


def cfunction_call(
    state: State, uop: Uop, meth: z3.ArithRef, args: list[z3.DatatypeRef]
) -> tuple[z3.DatatypeRef, z3.BoolRef, State]:
    sorts = [z3.IntSort()] + [Slot] * len(args) + [state.heap.sort(), z3.IntSort()]
    call = z3.Function(f"cfunc_call_{len(args)}", *sorts, Slot)
    fails = z3.Function(f"cfunc_error_{len(args)}", *sorts, z3.BoolSort())
    values: list[z3.ExprRef] = (
        [meth] + [value_projection(arg) for arg in args] + [state.heap, state.epoch]
    )
    result = call(*values)
    state = with_constraint(
        state,
        Slot.is_Ref(result),
        z3.Implies(is_immortal_id(state.consts, Slot.id(result)), Slot.borrowed(result)),
    )
    return result, fails(*values), state


def call_method_descriptor(kind: str, inline: bool) -> Handler:
    def handler(uop: Uop, state: State) -> State:
        before = state
        state = descriptor_facts(state, uop)
        if inline and kind in ("noargs", "o"):
            slots, state = pop_n(state, uop, uop.oparg + 1)
            callable_, call_args = slots[0], slots[1:]
            meth: z3.ArithRef = z3.IntVal(uop.operand0)
        elif inline:
            slots, state = pop_n(state, uop, uop.oparg + 2)
            callable_, self_st, args = slots[0], slots[1], slots[2:]
            meth = z3.IntVal(uop.operand0)
            call_args = [self_st] + args
        else:
            slots, state = pop_n(state, uop, uop.oparg + 2)
            callable_, self_or_null, args = slots[0], slots[1], slots[2:]
            meth = descr_ml_meth(Slot.id(callable_))
            if kind in ("noargs", "o"):
                state = precondition(
                    uop, state, z3.Not(Slot.is_Null(self_or_null)), "the method has a self argument"
                )
            call_args = [self_or_null] + args
        result, failed, state = cfunction_call(state, uop, meth, call_args)
        raises, state = raising_escape(uop, state, call_args)
        state = exit_if(uop, state, ExitKind.ERROR, z3.Or(raises, failed), handover=before)
        if kind in ("noargs", "o"):
            return push_all(state, [result, callable_, *call_args])
        state = replace(state, ledger=decref_on_pop(state, callable_))
        return push_all(state, [result, *call_args])

    return handler


def call_builtin_fast_like(uop: Uop, state: State) -> State:
    before = state
    slots, state = pop_n(state, uop, uop.oparg + 2)
    callable_, self_or_null, args = slots[0], slots[1], slots[2:]
    result, state = unknown_result(state, uop, [callable_, self_or_null] + args, 0)
    state = escaping_call(
        uop, state, before, [callable_, self_or_null] + args, exposed=[self_or_null] + args
    )
    state = replace(state, ledger=decref_on_pop(state, callable_))
    _, state = escape(state, uop.i, uop.op, z3.Not(harmless_to_close(state, callable_)), False)
    return push_all(state, [result, self_or_null, *args])


def call_builtin_o(uop: Uop, state: State) -> State:
    before = state
    slots, state = pop_n(state, uop, uop.oparg + 2)
    callable_, self_or_null, args = slots[0], slots[1], slots[2:]
    if uop.oparg > 1:
        raise UnsupportedUop(f"{uop.op} at {uop.i}: oparg {uop.oparg}")
    arg = z3.If(Slot.is_Null(self_or_null), args[0], self_or_null) if args else self_or_null
    result, state = unknown_result(state, uop, [callable_, arg], 0)
    state = escaping_call(uop, state, before, [callable_, arg], exposed=[arg])
    return push_all(state, [result, callable_, arg])


HANDLERS: dict[str, Handler] = {
    "_CALL_BUILTIN_O": call_builtin_o,
    "_GUARD_GLOBALS_VERSION": guard_globals_version,
    "_LOAD_GLOBAL_MODULE": load_global_module,
    "_LOAD_GLOBAL_BUILTINS": load_global_builtins,
    "_BINARY_OP_ADD_INT_INPLACE": int_inplace(lambda a, b: a + b, True),
    "_BINARY_OP_SUBTRACT_INT_INPLACE": int_inplace(lambda a, b: a - b, True),
    "_BINARY_OP_MULTIPLY_INT_INPLACE": int_inplace(lambda a, b: a * b, True),
    "_BINARY_OP_ADD_INT_INPLACE_RIGHT": int_inplace(lambda a, b: a + b, False),
    "_BINARY_OP_SUBTRACT_INT_INPLACE_RIGHT": int_inplace(lambda a, b: a - b, False),
    "_BINARY_OP_MULTIPLY_INT_INPLACE_RIGHT": int_inplace(lambda a, b: a * b, False),
    "_BINARY_OP_SUBSCR_DICT": dict_subscript_generic,
    "_BINARY_OP_SUBSCR_DICT_KNOWN_HASH": dict_subscript_generic,
    "_STORE_SUBSCR_DICT": store_subscr_dict_generic,
    "_STORE_SUBSCR_DICT_KNOWN_HASH": store_subscr_dict_generic,
    "_GUARD_CALLABLE_METHOD_DESCRIPTOR_NOARGS": guard_callable_method_descriptor(METH_NOARGS, 1),
    "_GUARD_CALLABLE_METHOD_DESCRIPTOR_O": guard_callable_method_descriptor(METH_O, 2),
    "_GUARD_CALLABLE_METHOD_DESCRIPTOR_FAST": guard_callable_method_descriptor(METH_FASTCALL, None),
    "_GUARD_CALLABLE_METHOD_DESCRIPTOR_FAST_WITH_KEYWORDS": guard_callable_method_descriptor(
        METH_FASTCALL | METH_KEYWORDS, None
    ),
    "_CALL_METHOD_DESCRIPTOR_NOARGS": call_method_descriptor("noargs", False),
    "_CALL_METHOD_DESCRIPTOR_O": call_method_descriptor("o", False),
    "_CALL_METHOD_DESCRIPTOR_FAST": call_method_descriptor("fast", False),
    "_CALL_METHOD_DESCRIPTOR_FAST_WITH_KEYWORDS": call_method_descriptor("fast", False),
    "_CALL_METHOD_DESCRIPTOR_NOARGS_INLINE": call_method_descriptor("noargs", True),
    "_CALL_METHOD_DESCRIPTOR_O_INLINE": call_method_descriptor("o", True),
    "_CALL_METHOD_DESCRIPTOR_FAST_INLINE": call_method_descriptor("fast", True),
    "_CALL_METHOD_DESCRIPTOR_FAST_WITH_KEYWORDS_INLINE": call_method_descriptor("fast", True),
    "_CALL_BUILTIN_FAST": call_builtin_fast_like,
    "_CALL_BUILTIN_FAST_WITH_KEYWORDS": call_builtin_fast_like,
    "_CALL_BUILTIN_CLASS": call_builtin_fast_like,
}
