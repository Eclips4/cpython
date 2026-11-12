from dataclasses import replace

import z3

from ...domain.trace import Uop
from ...symbolic.arithmetic import fp_arith
from ...symbolic.escape import escape
from ...symbolic.functions import (
    ITER_SELF,
    ITER_VIRTUAL,
    dict_store_subscript,
    dict_subscript,
    iter_kind,
    special_descr,
    super_descr,
    type_generic_getattr,
    type_object_id,
)
from ...symbolic.heap import expose, is_unique, new_int, new_object, new_str
from ...symbolic.identities import (
    DICT_TAG,
    FROZENDICT_TAG,
    FROZENSET_TAG,
    LIST_TAG,
    MODULE_TAG,
    SET_TAG,
    SLICE_TAG,
    TUPLE_TAG,
    type_tag,
)
from ...symbolic.ledger import decref_on_pop, decref_on_pop_if, ledger_add
from ...symbolic.objects import (
    bool_of,
    borrowed_if_immortal,
    float_value,
    is_compact,
    is_float,
    is_immortal_id,
    is_str,
    is_tuple,
    is_type_object,
    slot_int_value,
    slot_is_exact_int,
    slot_object,
    str_value,
)
from ...symbolic.protocols import harmless_to_close, length_known, length_of
from ...symbolic.sorts import HeapSort, Obj, Slot
from ...symbolic.state import ExitKind, State, push, set_local, set_stack, with_constraint
from ...symbolic.summaries import constant_facts, type_facts, type_tag_of_summary
from ...symbolic.types import has_method_descriptor_flag, has_tag, type_tag_of, version_check
from .. import uop_metadata
from ..step import (
    Handler,
    UnsupportedUop,
    allocation_exit_if,
    at_depth,
    exit_if,
    new_ref,
    nos,
    pop_n,
    precondition,
    push_constant,
    tos,
)
from ..uninterpreted import unknown_error, unknown_result
from .common import (
    COMPARE_EQ,
    COMPARE_GT,
    COMPARE_LT,
    COMPARE_UNORDERED,
    SlotTest,
    escaping_call,
    push_all,
    raising_escape,
    take_ownership,
)

list_item = z3.Function(
    "list_item", z3.IntSort(), z3.IntSort(), HeapSort, z3.IntSort(), z3.IntSort()
)


def iter_kind_is(kind: int) -> SlotTest:
    return lambda state, slot: iter_kind(type_tag_of(state, Slot.id(slot))) == kind


def has_any_tag(*tags: int) -> SlotTest:
    return lambda state, slot: z3.Or(*[has_tag(state, slot, tag) for tag in tags])


def has_one_tag(tag: int) -> SlotTest:
    return lambda state, slot: has_tag(state, slot, tag)


KIND_TESTS: dict[str, SlotTest] = {
    "float": is_float,
    "str": is_str,
    "tuple": is_tuple,
    "list": has_one_tag(LIST_TAG),
    "dict": has_one_tag(DICT_TAG),
    "frozendict": has_one_tag(FROZENDICT_TAG),
    "any_dict": has_any_tag(DICT_TAG, FROZENDICT_TAG),
    "set": has_one_tag(SET_TAG),
    "frozenset": has_one_tag(FROZENSET_TAG),
    "any_set": has_any_tag(SET_TAG, FROZENSET_TAG),
    "slice": has_one_tag(SLICE_TAG),
    "dict_subscript": lambda state, slot: dict_subscript(type_tag_of(state, Slot.id(slot))),
    "dict_store_subscript": lambda state, slot: dict_store_subscript(
        type_tag_of(state, Slot.id(slot))
    ),
    "iter_self": iter_kind_is(ITER_SELF),
    "iter_virtual": iter_kind_is(ITER_VIRTUAL),
}
DEOPTING_TESTS = frozenset({"dict_subscript", "dict_store_subscript"})


def guard_kind(test: str, depth: int) -> Handler:
    def handler(uop: Uop, state: State) -> State:
        slot = at_depth(state, uop, depth)
        state = precondition(uop, state, Slot.is_Ref(slot), "guarded slot is an object")
        if test not in KIND_TESTS:
            raise UnsupportedUop(f"{uop.op}: unknown kind test {test}")
        matches = KIND_TESTS[test](state, slot)
        kind = ExitKind.DEOPT if test in DEOPTING_TESTS else ExitKind.SIDE
        return exit_if(uop, state, kind, z3.Not(matches))

    return handler


def guard_type_version(uop: Uop, state: State) -> State:
    owner = tos(state, uop)
    state = precondition(uop, state, Slot.is_Ref(owner), "owner is an object")
    matches, state = version_check(state, type_tag_of(state, Slot.id(owner)), uop.operand0)
    return exit_if(uop, state, ExitKind.SIDE, z3.Not(matches))


def guard_nos_type_version(uop: Uop, state: State) -> State:
    owner = nos(state, uop)
    state = precondition(uop, state, Slot.is_Ref(owner), "owner is an object")
    matches, state = version_check(state, Obj.ttag(slot_object(state, owner)), uop.operand0)
    return exit_if(uop, state, ExitKind.SIDE, z3.Not(z3.And(is_type_object(state, owner), matches)))


def lookup_fact(state: State, uop: Uop, function: z3.FuncDeclRef, *keys: z3.ArithRef) -> State:
    found = uop.lookup
    if found is None or found.addr is None:
        return state
    ident, state = constant_facts(state, found)
    if found.type_name is not None:
        state = with_constraint(state, *type_facts(found))
    return with_constraint(state, function(*keys) == ident)


def recorded_type_tag(state: State, uop: Uop) -> tuple[z3.ArithRef | None, State]:
    summary = uop.lookup_type
    if summary is None or summary.type_name is None:
        return None, state
    state = with_constraint(state, *type_facts(summary))
    return z3.IntVal(type_tag_of_summary(summary)), state


def load_special(uop: Uop, state: State) -> State:
    before = state
    (slot_method, self_), state = pop_n(state, uop, 2)
    state = precondition(
        uop,
        state,
        z3.And(Slot.is_Null(slot_method), Slot.is_Ref(self_)),
        "the method slot is NULL and self an object",
    )
    tag = type_tag_of(state, Slot.id(self_))
    descr_id = special_descr(tag, z3.IntVal(uop.oparg))
    known_tag, state = recorded_type_tag(state, uop)
    if known_tag is not None:
        state = lookup_fact(state, uop, special_descr, known_tag, z3.IntVal(uop.oparg))
    is_method = has_method_descriptor_flag(state, descr_id)
    missing = descr_id == 0
    bound, state = unknown_result(state, uop, [self_], 0)
    raises, state = raising_escape(uop, state, [self_], z3.Not(is_method))
    error = z3.Or(missing, raises, z3.And(z3.Not(is_method), unknown_error(state, uop, [self_])))
    state = exit_if(uop, state, ExitKind.ERROR, error, handover=before)
    descr_ref, state = new_ref(state, descr_id)
    state = replace(state, ledger=decref_on_pop_if(state, z3.Not(is_method), self_))
    return push_all(state, [z3.If(is_method, descr_ref, bound), z3.If(is_method, self_, Slot.Null)])


def guard_load_super_attr_method(uop: Uop, state: State) -> State:
    global_super = at_depth(state, uop, 3)
    class_ = nos(state, uop)
    super_call = z3.And(
        Slot.is_Ref(global_super),
        Slot.id(global_super) == type_object_id(z3.IntVal(type_tag("super"))),
        is_type_object(state, class_),
    )
    return exit_if(uop, state, ExitKind.SIDE, z3.Not(super_call))


def load_super_attr_method(uop: Uop, state: State) -> State:
    before = state
    (global_super, class_, self_), state = pop_n(state, uop, 3)
    state = precondition(
        uop,
        state,
        z3.And(is_type_object(state, class_), Slot.is_Ref(self_)),
        "class is a type and self an object",
    )
    class_tag = Obj.ttag(slot_object(state, class_))
    self_tag = type_tag_of(state, Slot.id(self_))
    descr_id = super_descr(class_tag, self_tag, z3.IntVal(uop.oparg >> 2))
    known_tag, state = recorded_type_tag(state, uop)
    if known_tag is not None:
        state = lookup_fact(
            state, uop, super_descr, class_tag, known_tag, z3.IntVal(uop.oparg >> 2)
        )
    is_method = z3.And(type_generic_getattr(self_tag), has_method_descriptor_flag(state, descr_id))
    bound, state = unknown_result(state, uop, [class_, self_], 0)
    state = escaping_call(uop, state, before, [class_, self_], when=z3.Not(is_method))
    for slot in (global_super, class_):
        state = replace(state, ledger=decref_on_pop(state, slot))
        _, state = escape(state, uop.i, uop.op, z3.Not(harmless_to_close(state, slot)), False)
    state = replace(state, ledger=decref_on_pop_if(state, z3.Not(is_method), self_))
    descr_ref, state = new_ref(state, descr_id)
    return push_all(state, [z3.If(is_method, descr_ref, bound), z3.If(is_method, self_, Slot.Null)])


def check_attr_class(uop: Uop, state: State) -> State:
    owner = tos(state, uop)
    matches, state = version_check(state, Obj.ttag(slot_object(state, owner)), uop.operand0)
    return exit_if(uop, state, ExitKind.SIDE, z3.Not(z3.And(is_type_object(state, owner), matches)))


def guard_type_iter(uop: Uop, state: State) -> State:
    expected = uop.operand0_object
    if expected is None or expected.type_name is None:
        raise UnsupportedUop(f"{uop.op} at {uop.i}: no type recorded for the operand")
    iterator = nos(state, uop)
    state = with_constraint(state, *type_facts(expected))
    state = precondition(uop, state, Slot.is_Ref(iterator), "iterator is an object")
    other_type = type_tag_of(state, Slot.id(iterator)) != type_tag_of_summary(expected)
    return exit_if(uop, state, ExitKind.SIDE, other_type)


def record_type(uop: Uop, state: State) -> State:
    recorded = uop.recorded
    if recorded is not None and recorded.type_name is not None:
        state = with_constraint(state, *type_facts(recorded))
    return state


def record_value(uop: Uop, state: State) -> State:
    recorded = uop.recorded
    if recorded is not None:
        try:
            _, state = constant_facts(state, recorded, held=False)
        except ValueError:
            return state
    return state


def binary_op_float(operation: str) -> Handler:
    def handler(uop: Uop, state: State) -> State:
        before = state
        (left, right), state = pop_n(state, uop, 2)
        state = precondition(
            uop,
            state,
            z3.And(is_float(state, left), is_float(state, right)),
            "both operands are exact floats",
        )
        left_value, right_value = float_value(state, left), float_value(state, right)
        value = fp_arith(operation, left_value, right_value)
        slot, failure, state = new_object(state, Obj.Float(value), uop.i, "_f")
        if operation == "div":
            division_by_zero = z3.fpIsZero(right_value)
            state = exit_if(uop, state, ExitKind.ERROR, division_by_zero, handover=before)
        state = allocation_exit_if(uop, state, ExitKind.ERROR, failure, handover=before)
        return push_all(state, [slot, left, right])

    return handler


def binary_op_float_inplace(operation: str, target_is_left: bool) -> Handler:
    def handler(uop: Uop, state: State) -> State:
        before = state
        (left, right), state = pop_n(state, uop, 2)
        target = left if target_is_left else right
        state = precondition(
            uop,
            state,
            z3.And(is_float(state, left), is_float(state, right)),
            "both operands are exact floats",
        )
        state = precondition(
            uop,
            state,
            is_unique(state, target, [left, right]),
            "the mutated operand is uniquely referenced",
        )
        left_value, right_value = float_value(state, left), float_value(state, right)
        value = fp_arith(operation, left_value, right_value)
        if operation == "div":
            division_by_zero = z3.fpIsZero(right_value)
            state = exit_if(uop, state, ExitKind.ERROR, division_by_zero, handover=before)
        slot, failure, state = new_object(state, Obj.Float(value), uop.i, "_f")
        state = replace(state, ledger=decref_on_pop(state, target))
        if target_is_left:
            return push_all(state, [slot, Slot.Null, right])
        return push_all(state, [slot, left, Slot.Null])

    return handler


def unary_negative_float_inplace(uop: Uop, state: State) -> State:
    (value,), state = pop_n(state, uop, 1)
    state = precondition(uop, state, is_float(state, value), "operand is an exact float")
    state = precondition(
        uop, state, is_unique(state, value, [value]), "the mutated operand is uniquely referenced"
    )
    slot, failure, state = new_object(
        state, Obj.Float(z3.fpNeg(float_value(state, value))), uop.i, "_f"
    )
    state = replace(state, ledger=decref_on_pop(state, value))
    return push_all(state, [slot, Slot.Null])


def compare_op_float(uop: Uop, state: State) -> State:
    (left, right), state = pop_n(state, uop, 2)
    state = precondition(
        uop,
        state,
        z3.And(is_float(state, left), is_float(state, right)),
        "both operands are exact floats",
    )
    left_value, right_value = float_value(state, left), float_value(state, right)
    parts: list[z3.BoolRef] = []
    if uop.oparg & COMPARE_UNORDERED:
        parts.append(z3.Or(z3.fpIsNaN(left_value), z3.fpIsNaN(right_value)))
    if uop.oparg & COMPARE_LT:
        parts.append(z3.fpLT(left_value, right_value))
    if uop.oparg & COMPARE_GT:
        parts.append(z3.fpGT(left_value, right_value))
    if uop.oparg & COMPARE_EQ:
        parts.append(z3.fpEQ(left_value, right_value))
    truth = z3.Or(*parts) if parts else z3.BoolVal(False)
    return push_all(state, [bool_of(truth), left, right])


def binary_op_add_unicode(uop: Uop, state: State) -> State:
    before = state
    (left, right), state = pop_n(state, uop, 2)
    state = precondition(
        uop,
        state,
        z3.And(is_str(state, left), is_str(state, right)),
        "both operands are exact strs",
    )
    slot, failure, state = new_str(
        state, z3.Concat(str_value(state, left), str_value(state, right)), uop.i, "_s"
    )
    state = allocation_exit_if(uop, state, ExitKind.ERROR, failure, handover=before)
    return push_all(state, [slot, left, right])


def binary_op_inplace_add_unicode(uop: Uop, state: State) -> State:
    before = state
    (left, right), state = pop_n(state, uop, 2)
    index = uop.operand0 & 0xFFFF
    if index >= len(state.locals):
        raise UnsupportedUop(f"{uop.op} at {uop.i}: target local {index} outside the frame")
    target = state.locals[index]
    state = precondition(
        uop,
        state,
        z3.And(is_str(state, left), is_str(state, right)),
        "both operands are exact strs",
    )
    state = exit_if(
        uop,
        state,
        ExitKind.SIDE,
        z3.Not(z3.And(Slot.is_Ref(target), Slot.id(target) == Slot.id(left))),
        handover=before,
    )
    slot, failure, state = new_str(
        state, z3.Concat(str_value(state, left), str_value(state, right)), uop.i, "_s"
    )
    for slot_to_close in (left, right, target):
        state = replace(state, ledger=decref_on_pop(state, slot_to_close))
    state = set_local(state, index, Slot.Null)
    state = allocation_exit_if(uop, state, ExitKind.ERROR, failure)
    return push(state, slot)


def compare_op_str(uop: Uop, state: State) -> State:
    (left, right), state = pop_n(state, uop, 2)
    state = precondition(
        uop,
        state,
        z3.And(is_str(state, left), is_str(state, right)),
        "both operands are exact strs",
    )
    equal = str_value(state, left) == str_value(state, right)
    if_equal = z3.BoolVal(bool(uop.oparg & COMPARE_EQ))
    if_different = z3.BoolVal(bool(uop.oparg & (COMPARE_UNORDERED | COMPARE_LT | COMPARE_GT)))
    return push_all(state, [bool_of(z3.If(equal, if_equal, if_different)), left, right])


def iter_check_kind(tag: int) -> Handler:
    def handler(uop: Uop, state: State) -> State:
        iterator = nos(state, uop)
        state = precondition(uop, state, Slot.is_Ref(iterator), "iterator is an object")
        return exit_if(uop, state, ExitKind.SIDE, type_tag_of(state, Slot.id(iterator)) != tag)

    return handler


def guard_not_exhausted_tuple(uop: Uop, state: State) -> State:
    iterator, index = nos(state, uop), tos(state, uop)
    state = precondition(
        uop,
        state,
        z3.And(is_tuple(state, iterator), Slot.is_TaggedInt(index)),
        "NOS is an exact tuple and TOS a tagged index",
    )
    return exit_if(
        uop,
        state,
        ExitKind.SIDE,
        z3.Not(z3.And(Slot.k(index) >= 0, Slot.k(index) < Obj.tlen(slot_object(state, iterator)))),
    )


def guard_not_exhausted_list(uop: Uop, state: State) -> State:
    iterator, index = nos(state, uop), tos(state, uop)
    state = precondition(
        uop,
        state,
        z3.And(has_tag(state, iterator, LIST_TAG), Slot.is_TaggedInt(index)),
        "NOS is an exact list and TOS a tagged index",
    )
    return exit_if(
        uop,
        state,
        ExitKind.SIDE,
        z3.Not(z3.And(Slot.k(index) >= 0, Slot.k(index) < length_of(state, iterator))),
    )


def iter_next_tuple(uop: Uop, state: State) -> State:
    (iterator, index), state = pop_n(state, uop, 2)
    state = precondition(
        uop,
        state,
        z3.And(
            is_tuple(state, iterator),
            Slot.is_TaggedInt(index),
            Slot.k(index) >= 0,
            Slot.k(index) < Obj.tlen(slot_object(state, iterator)),
        ),
        "NOS is an exact tuple and the index is in range",
    )
    item = z3.Select(Obj.titems(slot_object(state, iterator)), Slot.k(index))
    state = precondition(uop, state, Slot.is_Ref(item), "the tuple item is an object")
    slot, state = new_ref(state, Slot.id(item))
    return push_all(state, [iterator, Slot.TaggedInt(Slot.k(index) + 1), slot])


def iter_next_list(uop: Uop, state: State) -> State:
    (iterator, index), state = pop_n(state, uop, 2)
    state = precondition(
        uop,
        state,
        z3.And(has_tag(state, iterator, LIST_TAG), Slot.is_TaggedInt(index)),
        "NOS is an exact list and TOS a tagged index",
    )
    slot, state = new_ref(
        state, list_item(Slot.id(iterator), Slot.k(index), state.heap, state.epoch)
    )
    return push_all(state, [iterator, Slot.TaggedInt(Slot.k(index) + 1), slot])


def get_iter(uop: Uop, state: State) -> State:
    before = state
    (iterable,), state = pop_n(state, uop, 1)
    state = precondition(uop, state, Slot.is_Ref(iterable), "iterable is an object")
    kind = iter_kind(type_tag_of(state, Slot.id(iterable)))
    virtual = kind == ITER_VIRTUAL
    produced, state = unknown_result(state, uop, [iterable], 0)
    state = replace(state, ledger=decref_on_pop_if(state, z3.Not(virtual), iterable))
    state = escaping_call(uop, state, before, [iterable], when=z3.Not(virtual))
    iterator = z3.If(virtual, iterable, produced)
    first_index = z3.If(virtual, Slot.TaggedInt(z3.IntVal(0)), Slot.Null)
    return push_all(state, [iterator, first_index])


def get_iter_trad(uop: Uop, state: State) -> State:
    before = state
    (iterable,), state = pop_n(state, uop, 1)
    state = precondition(uop, state, Slot.is_Ref(iterable), "iterable is an object")
    shared = replace(uop, op="_GET_ITER")
    produced, state = unknown_result(state, shared, [iterable], 0)
    state = replace(state, ledger=decref_on_pop(state, iterable))
    raises, state = raising_escape(uop, state, [iterable])
    error = z3.Or(raises, unknown_error(state, shared, [iterable]))
    state = exit_if(uop, state, ExitKind.ERROR, error, handover=before)
    return push_all(state, [produced, Slot.Null])


def iternext_result(
    state: State, uop: Uop, iterator: z3.DatatypeRef
) -> tuple[z3.DatatypeRef, State]:
    next_of = z3.Function(f"iternext_p{uop.i}", z3.IntSort(), state.heap.sort(), z3.IntSort(), Slot)
    value = next_of(Slot.id(iterator), state.heap, state.epoch)
    state = with_constraint(
        state,
        borrowed_if_immortal(state, value),
    )
    return value, state


def for_iter_tier_two(uop: Uop, state: State) -> State:
    before = state
    iterator, index = nos(state, uop), tos(state, uop)
    state = precondition(uop, state, Slot.is_Ref(iterator), "iterator is an object")
    virtual = iter_kind(type_tag_of(state, Slot.id(iterator))) == ITER_VIRTUAL
    state = precondition(
        uop,
        state,
        Slot.is_TaggedInt(index) == virtual,
        "a tagged index sits next to a virtual iterable (T1)",
    )
    item, state = iternext_result(state, uop, iterator)
    vitem_fn = z3.Function(
        f"vnext_item_p{uop.i}", z3.IntSort(), z3.IntSort(), state.heap.sort(), z3.IntSort(), Slot
    )
    vindex_fn = z3.Function(
        f"vnext_index_p{uop.i}",
        z3.IntSort(),
        z3.IntSort(),
        state.heap.sort(),
        z3.IntSort(),
        z3.IntSort(),
    )
    position = z3.If(Slot.is_TaggedInt(index), Slot.k(index), z3.IntVal(0))
    vitem = vitem_fn(Slot.id(iterator), position, state.heap, state.epoch)
    vindex = vindex_fn(Slot.id(iterator), position, state.heap, state.epoch)
    produced = z3.If(virtual, vitem, item)
    raises, state = raising_escape(uop, state, [iterator], z3.Not(virtual))
    shared = replace(uop, op="_FOR_ITER_TIER_TWO", oparg=0, operand0=0, operand1=0)
    error = z3.Or(raises, z3.And(z3.Not(virtual), unknown_error(state, shared, [iterator])))
    state = exit_if(uop, state, ExitKind.ERROR, error, handover=before)
    state = exit_if(uop, state, ExitKind.SIDE, Slot.is_Null(produced), handover=before)
    state = set_stack(state, 1, z3.If(virtual, Slot.TaggedInt(vindex), index))
    return push(state, produced)


def iter_next_inline(uop: Uop, state: State) -> State:
    before = state
    iterator = nos(state, uop)
    state = precondition(uop, state, Slot.is_Ref(iterator), "iterator is an object")
    item, state = iternext_result(state, uop, iterator)
    raises, state = raising_escape(uop, state, [iterator])
    shared = replace(uop, op="_FOR_ITER_TIER_TWO", oparg=0, operand0=0, operand1=0)
    error = z3.Or(raises, unknown_error(state, shared, [iterator]))
    state = exit_if(uop, state, ExitKind.ERROR, error, handover=before)
    state = exit_if(uop, state, ExitKind.SIDE, Slot.is_Null(item), handover=before)
    return push(state, item)


def build_tuple(uop: Uop, state: State) -> State:
    before = state
    values, state = pop_n(state, uop, uop.oparg)
    items: z3.ArrayRef = z3.K(z3.IntSort(), Slot.Null)
    for position, value in enumerate(values):
        borrowed_value = Slot.Ref(Slot.id(value), z3.BoolVal(True))
        items = z3.Store(items, z3.IntVal(position), borrowed_value)
        state = precondition(uop, state, Slot.is_Ref(value), "tuple items are objects")
        mortal_borrowed = z3.And(
            Slot.borrowed(value), z3.Not(is_immortal_id(state.consts, Slot.id(value)))
        )
        state = replace(state, ledger=ledger_add(state.ledger, mortal_borrowed, Slot.id(value), -1))
    slot, failure, state = new_object(state, Obj.Tuple(z3.IntVal(uop.oparg), items), uop.i, "_t")
    state = allocation_exit_if(uop, state, ExitKind.ERROR, failure, handover=before)
    return push(state, slot)


def guard_subscr_tuple_int_bounds(uop: Uop, state: State) -> State:
    tuple_, subscript = nos(state, uop), tos(state, uop)
    state = precondition(
        uop,
        state,
        z3.And(is_tuple(state, tuple_), slot_is_exact_int(state, subscript)),
        "NOS is a tuple and TOS an exact int",
    )
    value = slot_int_value(state, subscript)
    in_bounds = z3.And(
        value >= 0, is_compact(state.consts, value), value < Obj.tlen(slot_object(state, tuple_))
    )
    return exit_if(uop, state, ExitKind.SIDE, z3.Not(in_bounds))


def binary_op_subscr_tuple_int(uop: Uop, state: State) -> State:
    (tuple_, subscript), state = pop_n(state, uop, 2)
    value = slot_int_value(state, subscript)
    state = precondition(
        uop,
        state,
        z3.And(
            is_tuple(state, tuple_),
            slot_is_exact_int(state, subscript),
            value >= 0,
            value < Obj.tlen(slot_object(state, tuple_)),
        ),
        "NOS is a tuple and TOS an index in range",
    )
    item = z3.Select(Obj.titems(slot_object(state, tuple_)), value)
    state = precondition(uop, state, Slot.is_Ref(item), "the tuple item is an object")
    slot, state = new_ref(state, Slot.id(item))
    return push_all(state, [slot, tuple_, subscript])


def unpack_two_tuple(uop: Uop, state: State) -> State:
    before = state
    (sequence,), state = pop_n(state, uop, 1)
    state = precondition(uop, state, is_tuple(state, sequence), "operand is an exact tuple")
    state = exit_if(
        uop, state, ExitKind.SIDE, Obj.tlen(slot_object(state, sequence)) != 2, handover=before
    )
    items = Obj.titems(slot_object(state, sequence))
    first, state = new_ref(state, Slot.id(z3.Select(items, z3.IntVal(1))))
    second, state = new_ref(state, Slot.id(z3.Select(items, z3.IntVal(0))))
    state = replace(state, ledger=decref_on_pop(state, sequence))
    return push_all(state, [first, second])


def unpack_tuple(uop: Uop, state: State) -> State:
    before = state
    (sequence,), state = pop_n(state, uop, 1)
    state = precondition(uop, state, is_tuple(state, sequence), "operand is an exact tuple")
    state = exit_if(
        uop,
        state,
        ExitKind.SIDE,
        Obj.tlen(slot_object(state, sequence)) != uop.oparg,
        handover=before,
    )
    items = Obj.titems(slot_object(state, sequence))
    slots: list[z3.DatatypeRef] = []
    for k in range(uop.oparg - 1, -1, -1):
        slot, state = new_ref(state, Slot.id(z3.Select(items, z3.IntVal(k))))
        slots.append(slot)
    state = replace(state, ledger=decref_on_pop(state, sequence))
    return push_all(state, slots)


def unpack_unique(count: int | None) -> Handler:
    def handler(uop: Uop, state: State) -> State:
        length = count if count is not None else uop.oparg
        (sequence,), state = pop_n(state, uop, 1)
        state = precondition(
            uop,
            state,
            z3.And(is_tuple(state, sequence), Obj.tlen(slot_object(state, sequence)) == length),
            f"operand is an exact tuple of length {length}",
        )
        state = precondition(
            uop,
            state,
            is_unique(state, sequence, [sequence]),
            "the unpacked tuple is uniquely referenced",
        )
        items = Obj.titems(slot_object(state, sequence))
        slots: list[z3.DatatypeRef] = []
        for k in range(length - 1, -1, -1):
            slot, state = new_ref(state, Slot.id(z3.Select(items, z3.IntVal(k))))
            slots.append(slot)
        state = replace(state, ledger=decref_on_pop(state, sequence))
        return push_all(state, slots)

    return handler


def get_len(uop: Uop, state: State) -> State:
    before = state
    container = tos(state, uop)
    known = length_known(state, container)
    slot, failure, state = new_int(state, length_of(state, container), uop.i)
    state = with_constraint(state, length_of(state, container) >= 0)
    state = escaping_call(uop, state, before, [container], when=z3.Not(known))
    state = allocation_exit_if(uop, state, ExitKind.ERROR, failure, handover=before)
    return push(state, slot)


def load_attr_descr(with_self: bool, close_owner: bool) -> Handler:
    def handler(uop: Uop, state: State) -> State:
        descriptor = uop.operand0_object
        if descriptor is None:
            raise UnsupportedUop(f"{uop.op} at {uop.i}: no descriptor recorded for the operand")
        (owner,), state = pop_n(state, uop, 1)
        state = precondition(uop, state, Slot.is_Ref(owner), "owner is an object")
        if close_owner:
            state = replace(state, ledger=decref_on_pop(state, owner))
            _, state = escape(state, uop.i, uop.op, z3.Not(harmless_to_close(state, owner)), False)
        state = push_constant(uop, state, descriptor, False)
        if with_self:
            state = push(state, owner)
        return state

    return handler


def load_attr_instance_value(uop: Uop, state: State) -> State:
    before = state
    (owner,), state = pop_n(state, uop, 1)
    state = precondition(uop, state, Slot.is_Ref(owner), "owner is an object")
    instance_value = z3.Function(
        "instance_value", z3.IntSort(), z3.IntSort(), state.heap.sort(), z3.IntSort(), Slot
    )
    value = instance_value(Slot.id(owner), z3.IntVal(uop.operand0), state.heap, state.epoch)
    kind = ExitKind.DEOPT if uop.op == "_LOAD_ATTR_INSTANCE_VALUE" else ExitKind.SIDE
    state = exit_if(uop, state, kind, Slot.is_Null(value), handover=before)
    slot, state = new_ref(state, Slot.id(value))
    return push_all(state, [slot, owner])


def load_attr_module(uop: Uop, state: State) -> State:
    before = state
    (owner,), state = pop_n(state, uop, 1)
    state = precondition(uop, state, Slot.is_Ref(owner), "owner is an object")
    not_module = type_tag_of(state, Slot.id(owner)) != MODULE_TAG
    version_fn = z3.Function(
        "module_dict_version", z3.IntSort(), state.heap.sort(), z3.IntSort(), z3.IntSort()
    )
    attr_fn = z3.Function(
        "module_attr",
        z3.IntSort(),
        z3.IntSort(),
        z3.IntSort(),
        state.heap.sort(),
        z3.IntSort(),
        Slot,
    )
    value = attr_fn(
        Slot.id(owner), z3.IntVal(uop.operand0), z3.IntVal(uop.operand1), state.heap, state.epoch
    )
    if uop.global_value is not None and uop.global_value.addr is not None:
        ident, state = constant_facts(state, uop.global_value)
        state = with_constraint(
            state, z3.Implies(state.valid, value == Slot.Ref(ident, z3.BoolVal(True)))
        )
        state = with_constraint(
            state,
            z3.Implies(
                state.valid,
                version_fn(Slot.id(owner), state.heap, state.epoch) == z3.IntVal(uop.operand0),
            ),
        )
    condition = z3.Or(
        not_module,
        version_fn(Slot.id(owner), state.heap, state.epoch) != z3.IntVal(uop.operand0),
        Slot.is_Null(value),
    )
    state = exit_if(uop, state, ExitKind.SIDE, condition, handover=before)
    slot, state = new_ref(state, Slot.id(value))
    return push_all(state, [slot, owner])


def typed_build(tag: int, close_inputs: bool, result_kind: str = "other") -> Handler:
    def handler(uop: Uop, state: State) -> State:
        before = state
        info = uop_metadata.info(uop.op)
        if info is None:
            raise UnsupportedUop(f"{uop.op} at {uop.i}: no DSL metadata")
        count = info.popped(uop.oparg)
        values, state = pop_n(state, uop, count)
        for item in values:
            if close_inputs:
                state = replace(state, ledger=decref_on_pop(state, item))
            else:
                state = take_ownership(state, item)
        if result_kind == "str":
            text = z3.Function(
                f"built_str_p{uop.i}", state.heap.sort(), z3.IntSort(), z3.StringSort()
            )(state.heap, state.epoch)
            slot, failure, state = new_str(state, text, uop.i, "_b")
        else:
            slot, failure, state = new_object(state, Obj.Other(z3.IntVal(tag)), uop.i, "_b")
        if info.has("ESCAPES"):
            state = escaping_call(uop, state, before, values)
        state = allocation_exit_if(uop, state, ExitKind.ERROR, failure, handover=before)
        return push(state, slot)

    return handler


def container_mutation(escapes: bool) -> Handler:
    def handler(uop: Uop, state: State) -> State:
        before = state
        info = uop_metadata.info(uop.op)
        if info is None:
            raise UnsupportedUop(f"{uop.op} at {uop.i}: no DSL metadata")
        n_in, n_out = info.popped(uop.oparg), info.pushed(uop.oparg)
        values, state = pop_n(state, uop, n_in)
        kept, consumed = values[:n_out], values[n_out:]
        for item in consumed:
            state = take_ownership(state, item)
        error: z3.BoolRef = unknown_error(state, uop, values)
        if escapes:
            raises, state = raising_escape(uop, state, consumed)
            error = z3.Or(error, raises)
        state = exit_if(uop, state, ExitKind.ERROR, error, handover=before)
        return push_all(state, kept)

    return handler


def binary_op_subscr_list_int(uop: Uop, state: State) -> State:
    before = state
    (list_, subscript), state = pop_n(state, uop, 2)
    state = precondition(
        uop,
        state,
        z3.And(has_tag(state, list_, LIST_TAG), slot_is_exact_int(state, subscript)),
        "NOS is an exact list and TOS an exact int",
    )
    index = slot_int_value(state, subscript)
    length = length_of(state, list_)
    normalized = z3.If(index < 0, index + length, index)
    state = exit_if(
        uop,
        state,
        ExitKind.SIDE,
        z3.Not(z3.And(normalized >= 0, normalized < length)),
        handover=before,
    )
    slot, state = new_ref(state, list_item(Slot.id(list_), normalized, state.heap, state.epoch))
    return push_all(state, [slot, list_, subscript])


def binary_op_subscr_list_slice(uop: Uop, state: State) -> State:
    before = state
    (list_, subscript), state = pop_n(state, uop, 2)
    state = precondition(uop, state, has_tag(state, list_, LIST_TAG), "NOS is an exact list")
    slot, failure, state = new_object(state, Obj.Other(z3.IntVal(LIST_TAG)), uop.i, "_b")
    state = escaping_call(uop, state, before, [list_, subscript], exposed=[subscript])
    state = allocation_exit_if(uop, state, ExitKind.ERROR, failure, handover=before)
    return push_all(state, [slot, list_, subscript])


def store_subscr_list_int(uop: Uop, state: State) -> State:
    before = state
    (value, list_, subscript), state = pop_n(state, uop, 3)
    state = precondition(
        uop,
        state,
        z3.And(has_tag(state, list_, LIST_TAG), slot_is_exact_int(state, subscript)),
        "the container is an exact list and the index an exact int",
    )
    index = slot_int_value(state, subscript)
    length = length_of(state, list_)
    normalized = z3.If(index < 0, index + length, index)
    state = exit_if(
        uop,
        state,
        ExitKind.DEOPT,
        z3.Not(z3.And(normalized >= 0, normalized < length)),
        handover=before,
    )
    state = take_ownership(state, value)
    state = expose(state, [value])
    old_id = list_item(Slot.id(list_), normalized, state.heap, state.epoch)
    old = Slot.Ref(old_id, z3.BoolVal(False))
    state = replace(state, ledger=decref_on_pop(state, old))
    _, state = escape(state, uop.i, uop.op, z3.Not(harmless_to_close(state, old)), False)
    return push_all(state, [list_, subscript])


def make_function(uop: Uop, state: State) -> State:
    before = state
    (code,), state = pop_n(state, uop, 1)
    version = z3.Function(
        f"new_func_version_p{uop.i}", z3.IntSort(), state.heap.sort(), z3.IntSort(), z3.IntSort()
    )(Slot.id(code), state.heap, state.epoch)
    slot, failure, state = new_object(state, Obj.Func(version, version), uop.i, "_fn")
    raises, state = escape(state, uop.i, uop.op, z3.BoolVal(True), True)
    assert raises is not None
    state = exit_if(uop, state, ExitKind.ERROR, raises, handover=before)
    state = allocation_exit_if(uop, state, ExitKind.ERROR, failure, handover=before)
    return push_all(state, [slot, code])


HANDLERS: dict[str, Handler] = {
    "_GUARD_TOS_FLOAT": guard_kind("float", 1),
    "_GUARD_NOS_FLOAT": guard_kind("float", 2),
    "_GUARD_TOS_UNICODE": guard_kind("str", 1),
    "_GUARD_NOS_UNICODE": guard_kind("str", 2),
    "_GUARD_TOS_TUPLE": guard_kind("tuple", 1),
    "_GUARD_NOS_TUPLE": guard_kind("tuple", 2),
    "_GUARD_TOS_LIST": guard_kind("list", 1),
    "_GUARD_NOS_LIST": guard_kind("list", 2),
    "_GUARD_TOS_DICT": guard_kind("dict", 1),
    "_GUARD_TOS_FROZENDICT": guard_kind("frozendict", 1),
    "_GUARD_TOS_ANY_DICT": guard_kind("any_dict", 1),
    "_GUARD_TOS_SET": guard_kind("set", 1),
    "_GUARD_TOS_FROZENSET": guard_kind("frozenset", 1),
    "_GUARD_TOS_ANY_SET": guard_kind("any_set", 1),
    "_GUARD_TOS_SLICE": guard_kind("slice", 1),
    "_GUARD_NOS_DICT_SUBSCRIPT": guard_kind("dict_subscript", 2),
    "_GUARD_NOS_DICT_STORE_SUBSCRIPT": guard_kind("dict_store_subscript", 2),
    "_GUARD_ITERATOR": guard_kind("iter_self", 1),
    "_GUARD_ITER_VIRTUAL": guard_kind("iter_virtual", 1),
    "_GUARD_TYPE_VERSION": guard_type_version,
    "_GUARD_TYPE_VERSION_LOCKED": guard_type_version,
    "_GUARD_NOS_TYPE_VERSION": guard_nos_type_version,
    "_LOAD_SPECIAL": load_special,
    "_GUARD_LOAD_SUPER_ATTR_METHOD": guard_load_super_attr_method,
    "_LOAD_SUPER_ATTR_METHOD": load_super_attr_method,
    "_CHECK_ATTR_CLASS": check_attr_class,
    "_GUARD_TYPE_ITER": guard_type_iter,
    "_BINARY_OP_ADD_FLOAT": binary_op_float("add"),
    "_BINARY_OP_SUBTRACT_FLOAT": binary_op_float("sub"),
    "_BINARY_OP_MULTIPLY_FLOAT": binary_op_float("mul"),
    "_BINARY_OP_TRUEDIV_FLOAT": binary_op_float("div"),
    "_BINARY_OP_ADD_FLOAT_INPLACE": binary_op_float_inplace("add", True),
    "_BINARY_OP_SUBTRACT_FLOAT_INPLACE": binary_op_float_inplace("sub", True),
    "_BINARY_OP_MULTIPLY_FLOAT_INPLACE": binary_op_float_inplace("mul", True),
    "_BINARY_OP_TRUEDIV_FLOAT_INPLACE": binary_op_float_inplace("div", True),
    "_BINARY_OP_ADD_FLOAT_INPLACE_RIGHT": binary_op_float_inplace("add", False),
    "_BINARY_OP_SUBTRACT_FLOAT_INPLACE_RIGHT": binary_op_float_inplace("sub", False),
    "_BINARY_OP_MULTIPLY_FLOAT_INPLACE_RIGHT": binary_op_float_inplace("mul", False),
    "_BINARY_OP_TRUEDIV_FLOAT_INPLACE_RIGHT": binary_op_float_inplace("div", False),
    "_UNARY_NEGATIVE_FLOAT_INPLACE": unary_negative_float_inplace,
    "_COMPARE_OP_FLOAT": compare_op_float,
    "_BINARY_OP_ADD_UNICODE": binary_op_add_unicode,
    "_BINARY_OP_INPLACE_ADD_UNICODE": binary_op_inplace_add_unicode,
    "_COMPARE_OP_STR": compare_op_str,
    "_ITER_CHECK_LIST": iter_check_kind(LIST_TAG),
    "_ITER_CHECK_TUPLE": iter_check_kind(TUPLE_TAG),
    "_GUARD_NOT_EXHAUSTED_TUPLE": guard_not_exhausted_tuple,
    "_GUARD_NOT_EXHAUSTED_LIST": guard_not_exhausted_list,
    "_ITER_NEXT_TUPLE": iter_next_tuple,
    "_ITER_NEXT_LIST_TIER_TWO": iter_next_list,
    "_GET_ITER": get_iter,
    "_GET_ITER_TRAD": get_iter_trad,
    "_FOR_ITER_TIER_TWO": for_iter_tier_two,
    "_ITER_NEXT_INLINE": iter_next_inline,
    "_BUILD_TUPLE": build_tuple,
    "_GUARD_BINARY_OP_SUBSCR_TUPLE_INT_BOUNDS": guard_subscr_tuple_int_bounds,
    "_BINARY_OP_SUBSCR_TUPLE_INT": binary_op_subscr_tuple_int,
    "_UNPACK_SEQUENCE_TWO_TUPLE": unpack_two_tuple,
    "_UNPACK_SEQUENCE_TUPLE": unpack_tuple,
    "_UNPACK_SEQUENCE_UNIQUE_TWO_TUPLE": unpack_unique(2),
    "_UNPACK_SEQUENCE_UNIQUE_THREE_TUPLE": unpack_unique(3),
    "_UNPACK_SEQUENCE_UNIQUE_TUPLE": unpack_unique(None),
    "_GET_LEN": get_len,
    "_LOAD_ATTR_METHOD_NO_DICT": load_attr_descr(True, False),
    "_LOAD_ATTR_METHOD_WITH_VALUES": load_attr_descr(True, False),
    "_LOAD_ATTR_METHOD_LAZY_DICT": load_attr_descr(True, False),
    "_LOAD_ATTR_NONDESCRIPTOR_NO_DICT": load_attr_descr(False, True),
    "_LOAD_ATTR_NONDESCRIPTOR_WITH_VALUES": load_attr_descr(False, True),
    "_LOAD_ATTR_CLASS": load_attr_descr(False, True),
    "_LOAD_ATTR_INSTANCE_VALUE": load_attr_instance_value,
    "_LOAD_ATTR_SLOT": load_attr_instance_value,
    "_LOAD_ATTR_MODULE": load_attr_module,
    "_RECORD_TOS_TYPE": record_type,
    "_RECORD_NOS_TYPE": record_type,
    "_RECORD_TOS": record_value,
    "_RECORD_NOS": record_value,
    "_RECORD_4OS": record_value,
    "_RECORD_CALLABLE": record_value,
    "_RECORD_CALLABLE_KW": record_value,
    "_RECORD_BOUND_METHOD": record_value,
    "_RECORD_NOS_GEN_FUNC": record_value,
    "_RECORD_3OS_GEN_FUNC": record_value,
    "_BUILD_LIST": typed_build(LIST_TAG, False),
    "_BUILD_MAP": typed_build(DICT_TAG, False),
    "_BUILD_SET": typed_build(SET_TAG, False),
    "_BUILD_STRING": typed_build(0, True, "str"),
    "_BUILD_SLICE": typed_build(SLICE_TAG, False),
    "_LIST_APPEND": container_mutation(False),
    "_SET_ADD": container_mutation(True),
    "_MAP_ADD": container_mutation(True),
    "_LIST_EXTEND": container_mutation(True),
    "_SET_UPDATE": container_mutation(True),
    "_DICT_UPDATE": container_mutation(True),
    "_BINARY_OP_SUBSCR_LIST_INT": binary_op_subscr_list_int,
    "_BINARY_OP_SUBSCR_LIST_SLICE": binary_op_subscr_list_slice,
    "_STORE_SUBSCR_LIST_INT": store_subscr_list_int,
    "_MAKE_FUNCTION": make_function,
}
