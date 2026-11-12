from collections.abc import Callable
from dataclasses import dataclass, replace

import z3

from ...domain.trace import Constant, Uop
from ...symbolic.arithmetic import fp_arith, py_floordiv, py_mod
from ...symbolic.escape import escape
from ...symbolic.functions import (
    BUILTIN_ISINSTANCE,
    BUILTIN_LEN,
    BUILTIN_LIST_APPEND,
    bit_and,
    bit_or,
    bit_xor,
    builtin_object,
    cfunc_flags,
    frozen_member,
    frozen_members_known,
    int_as_float,
    ipow,
    pow2,
    subtype,
    type_object_id,
    type_vectorcall,
)
from ...symbolic.heap import expose, new_int, new_object, new_str
from ...symbolic.identities import (
    CFUNCTION_TAG,
    DICT_TAG,
    FLOAT_TAG,
    FROZENDICT_TAG,
    FROZENSET_TAG,
    INT_TAG,
    LIST_TAG,
    METHOD_TAG,
    SET_TAG,
    STR_TAG,
    TUPLE_TAG,
    type_tag,
)
from ...symbolic.ledger import decref_on_pop
from ...symbolic.objects import (
    bool_of,
    bool_slot,
    float_value,
    is_compact,
    is_float,
    is_func,
    is_str,
    is_tuple,
    is_type_object,
    none_slot,
    plain_kind,
    slot_int_value,
    slot_intlike_value,
    slot_is_bool,
    slot_is_exact_int,
    slot_is_intlike,
    slot_is_none,
    slot_object,
    str_value,
)
from ...symbolic.protocols import harmless_to_close, length_known, length_of, truth
from ...symbolic.sorts import Obj, Slot
from ...symbolic.state import ExitKind, State, push, set_local, with_constraint
from ...symbolic.summaries import type_tag_of_summary
from ...symbolic.types import has_tag, type_tag_of
from ..step import (
    Handler,
    UnsupportedUop,
    allocation_exit_if,
    at_depth,
    exit_if,
    local_slot,
    new_ref,
    nos,
    pop_n,
    precondition,
    tos,
)
from ..uninterpreted import unknown_error, unknown_result, value_projection
from .common import (
    COMPARE_EQ,
    COMPARE_GT,
    COMPARE_LT,
    COMPARE_UNORDERED,
    escaping_call,
    push_all,
    raising_escape,
)

(
    NB_ADD,
    NB_AND,
    NB_FLOOR_DIVIDE,
    NB_LSHIFT,
    NB_MATRIX_MULTIPLY,
    NB_MULTIPLY,
    NB_REMAINDER,
    NB_OR,
    NB_POWER,
    NB_RSHIFT,
    NB_SUBTRACT,
    NB_TRUE_DIVIDE,
    NB_XOR,
) = range(13)

NB_INPLACE_BASE = 13
NB_SUBSCR = 26
COMPARE_TO_BOOL = 16


def pow2_axioms(state: State) -> State:
    if any("pow2" in str(c) for c in state.constraints[-70:]):
        return state
    return with_constraint(state, *[pow2(z3.IntVal(k)) == z3.IntVal(1 << k) for k in range(0, 63)])


def int_binary(
    uop: Uop, state: State, op: int, left: z3.ArithRef, right: z3.ArithRef
) -> tuple[z3.ArithRef | None, z3.FPRef | None, z3.BoolRef, State]:
    error: z3.BoolRef = z3.BoolVal(False)
    value: z3.ArithRef | None = None
    fvalue: z3.FPRef | None = None
    if op == NB_ADD:
        value = left + right
    elif op == NB_SUBTRACT:
        value = left - right
    elif op == NB_MULTIPLY:
        value = left * right
    elif op == NB_FLOOR_DIVIDE:
        error = right == 0
        value = py_floordiv(left, right)
    elif op == NB_REMAINDER:
        error = right == 0
        value = py_mod(left, right)
    elif op == NB_LSHIFT:
        error = right < 0
        state = pow2_axioms(state)
        value = left * pow2(right)
    elif op == NB_RSHIFT:
        error = right < 0
        state = pow2_axioms(state)
        value = py_floordiv(left, pow2(right))
    elif op == NB_AND:
        value = bit_and(left, right)
    elif op == NB_OR:
        value = bit_or(left, right)
    elif op == NB_XOR:
        value = bit_xor(left, right)
    elif op == NB_POWER:
        state = with_constraint(
            state,
            ipow(left, z3.IntVal(0)) == 1,
            ipow(left, z3.IntVal(1)) == left,
            ipow(left, z3.IntVal(2)) == left * left,
            ipow(left, z3.IntVal(3)) == left * left * left,
        )
        value = ipow(left, right)
    elif op == NB_TRUE_DIVIDE:
        error = right == 0
        fvalue = fp_arith("div", int_as_float(left), int_as_float(right))
    elif op == NB_MATRIX_MULTIPLY:
        error = z3.BoolVal(True)
    return value, fvalue, error, state


def float_binary(op: int, left: z3.FPRef, right: z3.FPRef) -> tuple[z3.FPRef | None, z3.BoolRef]:
    if op == NB_ADD:
        return fp_arith("add", left, right), z3.BoolVal(False)
    if op == NB_SUBTRACT:
        return fp_arith("sub", left, right), z3.BoolVal(False)
    if op == NB_MULTIPLY:
        return fp_arith("mul", left, right), z3.BoolVal(False)
    if op == NB_TRUE_DIVIDE:
        return fp_arith("div", left, right), z3.fpIsZero(right)
    if op in (NB_FLOOR_DIVIDE, NB_REMAINDER):
        function = z3.Function(
            "float_floordiv" if op == NB_FLOOR_DIVIDE else "float_mod",
            left.sort(),
            right.sort(),
            left.sort(),
        )
        return function(left, right), z3.fpIsZero(right)
    if op == NB_POWER:
        function = z3.Function("float_pow", left.sort(), right.sort(), left.sort())
        return function(left, right), z3.BoolVal(False)
    return None, z3.BoolVal(True)


def binary_op(uop: Uop, state: State) -> State:
    op = uop.oparg
    if NB_INPLACE_BASE <= op < NB_SUBSCR:
        op -= NB_INPLACE_BASE
    if op == NB_SUBSCR:
        op = NB_MATRIX_MULTIPLY + 100
    before = state
    (lhs, rhs), state = pop_n(state, uop, 2)
    left_int, right_int = slot_is_intlike(state, lhs), slot_is_intlike(state, rhs)
    both_int = z3.And(left_int, right_int)
    if op in (NB_AND, NB_OR, NB_XOR):
        both_int = z3.And(both_int, z3.Not(z3.And(slot_is_bool(lhs), slot_is_bool(rhs))))
    left_value, right_value = slot_intlike_value(state, lhs), slot_intlike_value(state, rhs)
    int_result, int_float_result, int_error, state = int_binary(
        uop, state, op, left_value, right_value
    )
    left_float, right_float = is_float(state, lhs), is_float(state, rhs)
    float_case = z3.And(
        z3.Or(left_float, left_int), z3.Or(right_float, right_int), z3.Or(left_float, right_float)
    )
    left_as_float = z3.If(left_float, float_value(state, lhs), int_as_float(left_value))
    right_as_float = z3.If(right_float, float_value(state, rhs), int_as_float(right_value))
    float_result, float_error = float_binary(op, left_as_float, right_as_float)
    str_case = z3.And(is_str(state, lhs), is_str(state, rhs), z3.BoolVal(op == NB_ADD))
    generic = z3.Not(z3.Or(both_int, float_case, str_case))
    results: list[tuple[z3.BoolRef, z3.DatatypeRef]] = []
    error_parts: list[z3.BoolRef] = []
    alloc_parts: list[z3.BoolRef] = []
    if int_result is not None:
        slot, failure, state = new_int(state, int_result, uop.i, both_int)
        results.append((both_int, slot))
        error_parts.append(z3.And(both_int, int_error))
        alloc_parts.append(z3.And(both_int, failure))
    elif int_float_result is not None:
        slot, failure, state = new_object(
            state, Obj.Float(int_float_result), uop.i, "_if", both_int
        )
        results.append((both_int, slot))
        error_parts.append(z3.And(both_int, int_error))
        alloc_parts.append(z3.And(both_int, failure))
    else:
        error_parts.append(z3.And(both_int, int_error))
    if float_result is not None:
        slot, failure, state = new_object(state, Obj.Float(float_result), uop.i, "_f", float_case)
        results.append((float_case, slot))
        error_parts.append(z3.And(float_case, float_error))
        alloc_parts.append(z3.And(float_case, failure))
    else:
        error_parts.append(z3.And(float_case, float_error))
    if op == NB_ADD:
        slot, failure, state = new_str(
            state, z3.Concat(str_value(state, lhs), str_value(state, rhs)), uop.i, "_s", str_case
        )
        results.append((str_case, slot))
        alloc_parts.append(z3.And(str_case, failure))
    unknown, state = unknown_result(state, uop, [lhs, rhs], 0)
    error_parts.append(z3.And(generic, unknown_error(state, uop, [lhs, rhs])))
    raises, state = raising_escape(uop, state, [lhs, rhs], generic)
    error_parts.append(raises)
    state = exit_if(uop, state, ExitKind.ERROR, z3.Or(*error_parts), handover=before)
    state = allocation_exit_if(uop, state, ExitKind.ERROR, z3.Or(*alloc_parts), handover=before)
    result = unknown
    for condition, slot in reversed(results):
        result = z3.If(condition, slot, result)
    return push_all(state, [result, lhs, rhs])


def compare_truth_masked(
    oparg: int, lt: z3.BoolRef, gt: z3.BoolRef, eq: z3.BoolRef, unordered: z3.BoolRef
) -> z3.BoolRef:
    parts: list[z3.BoolRef] = []
    if oparg & COMPARE_UNORDERED:
        parts.append(unordered)
    if oparg & COMPARE_LT:
        parts.append(lt)
    if oparg & COMPARE_GT:
        parts.append(gt)
    if oparg & COMPARE_EQ:
        parts.append(eq)
    return z3.Or(*parts) if parts else z3.BoolVal(False)


def compare_op(uop: Uop, state: State) -> State:
    before = state
    (left, right), state = pop_n(state, uop, 2)
    left_int, right_int = slot_is_intlike(state, left), slot_is_intlike(state, right)
    left_float, right_float = is_float(state, left), is_float(state, right)
    left_value, right_value = slot_intlike_value(state, left), slot_intlike_value(state, right)
    both_int = z3.And(left_int, right_int)
    num_case = z3.And(
        z3.Or(left_int, left_float), z3.Or(right_int, right_float), z3.Or(left_float, right_float)
    )
    left_as_float = z3.If(left_float, float_value(state, left), int_as_float(left_value))
    right_as_float = z3.If(right_float, float_value(state, right), int_as_float(right_value))
    str_case = z3.And(is_str(state, left), is_str(state, right))
    equality = (uop.oparg & 0xF) in (COMPARE_EQ, COMPARE_UNORDERED | COMPARE_LT | COMPARE_GT)
    str_equality = z3.And(str_case, z3.BoolVal(equality))
    left_none, right_none = (
        Obj.is_NoneObj(slot_object(state, left)),
        Obj.is_NoneObj(slot_object(state, right)),
    )
    mixed = z3.And(
        z3.BoolVal(equality),
        z3.Or(
            z3.And(z3.Or(left_int, left_float), is_str(state, right)),
            z3.And(is_str(state, left), z3.Or(right_int, right_float)),
            z3.And(left_none, z3.Or(right_int, right_float, is_str(state, right))),
            z3.And(right_none, z3.Or(left_int, left_float, is_str(state, left))),
        ),
    )
    int_truth = compare_truth_masked(
        uop.oparg,
        left_value < right_value,
        left_value > right_value,
        left_value == right_value,
        z3.BoolVal(False),
    )
    float_truth = compare_truth_masked(
        uop.oparg,
        z3.fpLT(left_as_float, right_as_float),
        z3.fpGT(left_as_float, right_as_float),
        z3.fpEQ(left_as_float, right_as_float),
        z3.Or(z3.fpIsNaN(left_as_float), z3.fpIsNaN(right_as_float)),
    )
    same = str_value(state, left) == str_value(state, right)
    str_truth = compare_truth_masked(uop.oparg, z3.Not(same), z3.Not(same), same, z3.BoolVal(False))
    mixed_truth = compare_truth_masked(
        uop.oparg, z3.BoolVal(False), z3.BoolVal(False), z3.BoolVal(False), z3.BoolVal(True)
    )
    generic = z3.Not(z3.Or(both_int, num_case, str_equality, mixed))
    unknown_truth = z3.Function(
        f"compare_truth_p{uop.i}", Slot, Slot, state.heap.sort(), z3.IntSort(), z3.BoolSort()
    )(left, right, state.heap, state.epoch)
    truth_value = z3.If(
        both_int,
        int_truth,
        z3.If(
            num_case,
            float_truth,
            z3.If(str_equality, str_truth, z3.If(mixed, mixed_truth, unknown_truth)),
        ),
    )
    for slot in (left, right):
        state = replace(state, ledger=decref_on_pop(state, slot))
    state = escaping_call(uop, state, before, [left, right], when=generic)
    if uop.oparg & COMPARE_TO_BOOL:
        return push(state, bool_of(truth_value))
    unknown, state = unknown_result(state, uop, [left, right], 0)
    return push(state, z3.If(generic, unknown, bool_of(truth_value)))


def same_value(state: State, first: z3.DatatypeRef, second: z3.DatatypeRef) -> z3.BoolRef:
    ints = z3.And(
        slot_is_exact_int(state, first),
        slot_is_exact_int(state, second),
        slot_int_value(state, first) == slot_int_value(state, second),
    )
    strs = z3.And(
        is_str(state, first),
        is_str(state, second),
        str_value(state, first) == str_value(state, second),
    )
    same_id = z3.And(Slot.is_Ref(first), Slot.is_Ref(second), Slot.id(first) == Slot.id(second))
    return z3.Or(same_id, ints, strs)


def contains_op_generic(
    uop: Uop, state: State, container_tags: tuple[int, ...] = (), note: str = ""
) -> State:
    before = state
    (left, right), state = pop_n(state, uop, 2)
    if container_tags:
        container = z3.Or(*[has_tag(state, right, tag) for tag in container_tags])
        state = precondition(uop, state, container, note)
    unknown_member = z3.Function(
        f"contains_p{uop.i}", Slot, Slot, state.heap.sort(), z3.IntSort(), z3.BoolSort()
    )(left, right, state.heap, state.epoch)
    str_case = z3.And(is_str(state, left), is_str(state, right))
    tuple_case = z3.And(is_tuple(state, right), plain_kind(state, left))
    items = Obj.titems(slot_object(state, right))
    length = Obj.tlen(slot_object(state, right))
    found = z3.BoolVal(False)
    for k in range(7, -1, -1):
        found = z3.Or(
            z3.And(length > k, same_value(state, z3.Select(items, z3.IntVal(k)), left)), found
        )
    tuple_member = z3.If(length <= 8, found, unknown_member)
    frozen_case = z3.And(
        Slot.is_Ref(right),
        frozen_members_known(Slot.id(right)),
        Slot.is_Ref(left),
        plain_kind(state, left),
    )
    frozen_hit = frozen_member(Slot.id(right), Slot.id(left))
    member = z3.If(
        str_case,
        z3.Contains(str_value(state, right), str_value(state, left)),
        z3.If(tuple_case, tuple_member, z3.If(frozen_case, frozen_hit, unknown_member)),
    )
    exact = z3.Or(str_case, z3.And(tuple_case, length <= 8), frozen_case)
    generic = z3.Not(z3.And(plain_kind(state, left), exact))
    state = escaping_call(uop, state, before, [left, right], when=generic)
    result = bool_of(z3.Not(member) if uop.oparg else member)
    return push_all(state, [result, left, right])


def contains_op(uop: Uop, state: State) -> State:
    return contains_op_generic(uop, state)


def contains_op_set(uop: Uop, state: State) -> State:
    tags = (SET_TAG, FROZENSET_TAG)
    return contains_op_generic(uop, state, tags, "right operand is a set or frozenset")


def contains_op_dict(uop: Uop, state: State) -> State:
    tags = (DICT_TAG, FROZENDICT_TAG)
    return contains_op_generic(uop, state, tags, "right operand is a dict or frozendict")


def binary_slice(uop: Uop, state: State) -> State:
    before = state
    (container, start, stop), state = pop_n(state, uop, 3)
    tag = type_tag_of(state, Slot.id(container))
    sequence = z3.Or(
        is_str(state, container), is_tuple(state, container), has_tag(state, container, LIST_TAG)
    )
    result, state = unknown_result(state, uop, [container, start, stop], 0)
    state = with_constraint(
        state,
        z3.Implies(
            z3.And(sequence, Slot.is_Ref(result)), type_tag_of(state, Slot.id(result)) == tag
        ),
    )
    for slot in (container, start, stop):
        state = replace(state, ledger=decref_on_pop(state, slot))
    state = escaping_call(uop, state, before, [container, start, stop])
    return push(state, result)


def binary_op_subscr_str_int(uop: Uop, state: State) -> State:
    before = state
    (string, subscript), state = pop_n(state, uop, 2)
    state = precondition(
        uop,
        state,
        z3.And(is_str(state, string), slot_is_exact_int(state, subscript)),
        "NOS is an exact str and TOS an exact int",
    )
    index = slot_int_value(state, subscript)
    length = z3.Length(str_value(state, string))
    state = exit_if(
        uop, state, ExitKind.SIDE, z3.Not(z3.And(index >= 0, index < length)), handover=before
    )
    result, state = unknown_result(state, uop, [string, subscript], 0)
    state = with_constraint(
        state,
        Slot.is_Ref(result),
        is_str(state, result),
        str_value(state, result) == z3.SubString(str_value(state, string), index, z3.IntVal(1)),
    )
    return push_all(state, [result, string, subscript])


@dataclass(frozen=True)
class ExtendTypes:
    left: int | None
    right: int | None
    result: int | None
    result_unique: bool
    has_guard: bool


def summary_type_tag(summary: Constant | None) -> int | None:
    if summary is None or summary.type_name is None:
        return None
    return type_tag_of_summary(summary)


def extend_types(uop: Uop) -> ExtendTypes:
    descr = uop.binop_descr
    if descr is None:
        return ExtendTypes(None, None, None, False, False)
    return ExtendTypes(
        summary_type_tag(descr.lhs_type),
        summary_type_tag(descr.rhs_type),
        summary_type_tag(descr.result_type),
        descr.result_unique,
        descr.has_guard,
    )


def guard_binary_op_extend(which: str) -> Handler:
    def handler(uop: Uop, state: State) -> State:
        left, right = nos(state, uop), tos(state, uop)
        types = extend_types(uop)
        kind = extend_custom_kind(uop)
        if types.has_guard and which == "both" and kind is not None:
            guard = extend_custom_guard(uop, state, left, right, kind)
            return exit_if(uop, state, ExitKind.SIDE, z3.Not(guard))
        if types.has_guard or types.left is None or types.right is None:
            raise UnsupportedUop(
                f"{uop.op} at {uop.i}: the specialization descriptor has a custom guard or unknown "
                "types"
            )
        state = precondition(
            uop, state, z3.And(Slot.is_Ref(left), Slot.is_Ref(right)), "operands are objects"
        )
        left_matches = type_tag_of(state, Slot.id(left)) == types.left
        right_matches = type_tag_of(state, Slot.id(right)) == types.right
        if which == "both":
            matches = z3.And(left_matches, right_matches)
        else:
            matches = left_matches if which == "lhs" else right_matches
        return exit_if(uop, state, ExitKind.SIDE, z3.Not(matches))

    return handler


def extend_nb(uop: Uop) -> int:
    return uop.oparg - NB_INPLACE_BASE if NB_INPLACE_BASE <= uop.oparg < NB_SUBSCR else uop.oparg


def extend_custom_kind(uop: Uop) -> str | None:
    types = extend_types(uop)
    if not types.has_guard or types.result is None:
        return None
    nb = extend_nb(uop)
    if types.result == INT_TAG and nb in (NB_AND, NB_OR, NB_XOR):
        return "compact_ints"
    if types.result == FLOAT_TAG and nb in (
        NB_ADD,
        NB_SUBTRACT,
        NB_MULTIPLY,
        NB_TRUE_DIVIDE,
    ):
        return "float_and_compact_int"
    return None


def extend_float_on_left(uop: Uop) -> z3.BoolRef:
    return z3.Bool(f"extend_float_on_left_{uop.operand0:x}")


def extend_custom_guard(
    uop: Uop, state: State, left: z3.DatatypeRef, right: z3.DatatypeRef, kind: str
) -> z3.BoolRef:
    def compact(slot: z3.DatatypeRef) -> z3.BoolRef:
        return z3.And(
            slot_is_exact_int(state, slot), is_compact(state.consts, slot_int_value(state, slot))
        )

    def finite(slot: z3.DatatypeRef) -> z3.BoolRef:
        return z3.And(is_float(state, slot), z3.Not(z3.fpIsNaN(float_value(state, slot))))

    if kind == "compact_ints":
        return z3.And(compact(left), compact(right))
    float_left = z3.And(finite(left), compact(right))
    float_right = z3.And(compact(left), finite(right))
    if extend_nb(uop) == NB_TRUE_DIVIDE:
        float_left = z3.And(float_left, slot_int_value(state, right) != 0)
        float_right = z3.And(float_right, z3.Not(z3.fpIsZero(float_value(state, right))))
    return z3.If(extend_float_on_left(uop), float_left, float_right)


def binary_op_extend_custom(
    uop: Uop, state: State, before: State, left: z3.DatatypeRef, right: z3.DatatypeRef, kind: str
) -> State:
    nb = extend_nb(uop)
    if kind == "compact_ints":
        left_value, right_value = slot_int_value(state, left), slot_int_value(state, right)
        value = (
            bit_and(left_value, right_value)
            if nb == NB_AND
            else bit_or(left_value, right_value)
            if nb == NB_OR
            else bit_xor(left_value, right_value)
        )
        result, failure, state = new_int(state, value, uop.i)
    else:
        on_left = extend_float_on_left(uop)
        left_as_float = z3.If(
            on_left, float_value(state, left), int_as_float(slot_int_value(state, left))
        )
        right_as_float = z3.If(
            on_left, int_as_float(slot_int_value(state, right)), float_value(state, right)
        )
        name = {NB_ADD: "add", NB_SUBTRACT: "sub", NB_MULTIPLY: "mul", NB_TRUE_DIVIDE: "div"}[nb]
        result, failure, state = new_object(
            state, Obj.Float(fp_arith(name, left_as_float, right_as_float)), uop.i, "_x"
        )
    state = allocation_exit_if(uop, state, ExitKind.ERROR, failure, handover=before)
    return push_all(state, [result, left, right])


def binary_op_extend(uop: Uop, state: State) -> State:
    before = state
    (left, right), state = pop_n(state, uop, 2)
    kind = extend_custom_kind(uop)
    if kind is not None:
        return binary_op_extend_custom(uop, state, before, left, right, kind)
    types = extend_types(uop)
    if types.result_unique and types.result is not None:
        value_of = z3.Function(
            f"extend_value_p{uop.i}", Slot, Slot, state.heap.sort(), z3.IntSort(), Obj
        )
        value = value_of(value_projection(left), value_projection(right), state.heap, state.epoch)
        result, failure, state = new_object(state, value, uop.i, "_x")
        state = with_constraint(state, type_tag_of(state, Slot.id(result)) == types.result)
    else:
        result, state = unknown_result(state, uop, [left, right], 0)
        failure = z3.BoolVal(False)
        if types.result is not None:
            result_type = type_tag_of(state, Slot.id(result)) == types.result
            state = with_constraint(state, z3.Implies(Slot.is_Ref(result), result_type))
    state = escaping_call(uop, state, before, [left, right])
    state = allocation_exit_if(uop, state, ExitKind.ERROR, failure, handover=before)
    return push_all(state, [result, left, right])


def unary_invert(uop: Uop, state: State) -> State:
    before = state
    (value,), state = pop_n(state, uop, 1)
    is_int = z3.And(slot_is_exact_int(state, value), z3.Not(slot_is_bool(value)))
    operand_value = slot_int_value(state, value)
    slot, failure, state = new_int(state, -operand_value - 1, uop.i)
    unknown, state = unknown_result(state, uop, [value], 0)
    raises, state = raising_escape(uop, state, [value], z3.Not(is_int))
    error = z3.Or(z3.And(z3.Not(is_int), unknown_error(state, uop, [value])), raises)
    state = exit_if(uop, state, ExitKind.ERROR, error, handover=before)
    state = allocation_exit_if(uop, state, ExitKind.ERROR, z3.And(is_int, failure), handover=before)
    state = push(state, z3.If(is_int, slot, unknown))
    return push(state, value)


def unary_negative(uop: Uop, state: State) -> State:
    before = state
    (value,), state = pop_n(state, uop, 1)
    is_int = slot_is_intlike(state, value)
    is_float_value = is_float(state, value)
    operand_value = slot_intlike_value(state, value)
    int_slot, int_failure, state = new_int(state, -operand_value, uop.i, is_int)
    float_slot, float_failure, state = new_object(
        state, Obj.Float(z3.fpNeg(float_value(state, value))), uop.i, "_f", is_float_value
    )
    unknown, state = unknown_result(state, uop, [value], 0)
    generic = z3.Not(z3.Or(is_int, is_float_value))
    raises, state = raising_escape(uop, state, [value], generic)
    error = z3.Or(z3.And(generic, unknown_error(state, uop, [value])), raises)
    state = exit_if(uop, state, ExitKind.ERROR, error, handover=before)
    state = allocation_exit_if(
        uop,
        state,
        ExitKind.ERROR,
        z3.Or(z3.And(is_int, int_failure), z3.And(is_float_value, float_failure)),
        handover=before,
    )
    state = push(state, z3.If(is_int, int_slot, z3.If(is_float_value, float_slot, unknown)))
    return push(state, value)


def to_bool(uop: Uop, state: State) -> State:
    before = state
    (value,), state = pop_n(state, uop, 1)
    generic = z3.Not(z3.Or(plain_kind(state, value), is_tuple(state, value)))
    result = bool_of(truth(state, value, uop.i))
    state = replace(state, ledger=decref_on_pop(state, value))
    state = escaping_call(uop, state, before, [value], when=generic)
    return push(state, result)


def to_bool_none(uop: Uop, state: State) -> State:
    (value,), state = pop_n(state, uop, 1)
    state = exit_if(uop, state, ExitKind.SIDE, z3.Not(slot_is_none(value)))
    return push(state, bool_slot(False))


def to_bool_str(uop: Uop, state: State) -> State:
    (value,), state = pop_n(state, uop, 1)
    state = precondition(uop, state, is_str(state, value), "operand is an exact str")
    state = push(state, bool_of(z3.Length(str_value(state, value)) != 0))
    return push(state, value)


def to_bool_list(uop: Uop, state: State) -> State:
    (value,), state = pop_n(state, uop, 1)
    state = precondition(uop, state, has_tag(state, value, LIST_TAG), "operand is an exact list")
    state = push(state, bool_of(length_of(state, value) != 0))
    return push(state, value)


def call_len(uop: Uop, state: State) -> State:
    before = state
    (callable_, null, arg), state = pop_n(state, uop, 3)
    known = length_known(state, arg)
    length = length_of(state, arg)
    slot, failure, state = new_int(state, length, uop.i)
    state = with_constraint(state, length >= 0)
    state = escaping_call(uop, state, before, [arg], when=z3.Not(known))
    state = allocation_exit_if(uop, state, ExitKind.ERROR, failure, handover=before)
    return push_all(state, [slot, arg, callable_])


def call_isinstance(uop: Uop, state: State) -> State:
    before = state
    (callable_, null, instance, cls), state = pop_n(state, uop, 4)
    instance_tag = type_tag_of(state, Slot.id(instance))
    cls_is_type = is_type_object(state, cls)
    cls_tag = Obj.ttag(slot_object(state, cls))
    state = with_constraint(state, subtype(instance_tag, instance_tag), subtype(cls_tag, cls_tag))
    result_truth = z3.If(
        cls_is_type,
        subtype(instance_tag, cls_tag),
        z3.Function(
            f"isinstance_p{uop.i}", Slot, Slot, state.heap.sort(), z3.IntSort(), z3.BoolSort()
        )(instance, cls, state.heap, state.epoch),
    )
    for slot in (cls, instance, callable_):
        state = replace(state, ledger=decref_on_pop(state, slot))
    state = escaping_call(uop, state, before, [instance, cls], when=z3.Not(cls_is_type))
    return push(state, bool_of(result_truth))


def call_type_1(uop: Uop, state: State) -> State:
    (callable_, null, arg), state = pop_n(state, uop, 3)
    state = precondition(uop, state, Slot.is_Ref(arg), "argument is an object")
    ident = type_object_id(type_tag_of(state, Slot.id(arg)))
    slot, state = new_ref(state, ident)
    state = with_constraint(
        state,
        Obj.is_TypeObj(z3.Select(state.heap, ident)),
        Obj.ttag(z3.Select(state.heap, ident)) == type_tag_of(state, Slot.id(arg)),
    )
    state = push(state, slot)
    return push(state, arg)


def call_str_1(uop: Uop, state: State) -> State:
    before = state
    (callable_, null, arg), state = pop_n(state, uop, 3)
    exact = is_str(state, arg)
    same, state = new_ref(state, Slot.id(arg))
    unknown, state = unknown_result(state, uop, [arg], 0)
    state = with_constraint(
        state, subtype(type_tag_of(state, Slot.id(unknown)), z3.IntVal(STR_TAG))
    )
    state = escaping_call(uop, state, before, [arg], when=z3.Not(exact))
    return push_all(state, [z3.If(exact, same, unknown), arg])


def call_tuple_1(uop: Uop, state: State) -> State:
    before = state
    (callable_, null, arg), state = pop_n(state, uop, 3)
    exact = is_tuple(state, arg)
    same, state = new_ref(state, Slot.id(arg))
    unknown, state = unknown_result(state, uop, [arg], 0)
    state = with_constraint(
        state,
        has_tag(state, unknown, TUPLE_TAG),
        Obj.is_Tuple(z3.Select(state.heap, Slot.id(unknown))),
    )
    state = escaping_call(uop, state, before, [arg], when=z3.Not(exact))
    return push_all(state, [z3.If(exact, same, unknown), arg])


def call_list_append(uop: Uop, state: State) -> State:
    before = state
    (callable_, self_, arg), state = pop_n(state, uop, 3)
    state = precondition(uop, state, has_tag(state, self_, LIST_TAG), "self is an exact list")
    state = expose(state, [arg])
    state = exit_if(
        uop, state, ExitKind.ERROR, unknown_error(state, uop, [self_, arg]), handover=before
    )
    return push_all(state, [none_slot(), callable_, self_])


def guard_callable_cached(which: int, depth: int) -> Handler:
    def handler(uop: Uop, state: State) -> State:
        callable_ = at_depth(state, uop, depth)
        cached = z3.And(
            Slot.is_Ref(callable_), Slot.id(callable_) == builtin_object(z3.IntVal(which))
        )
        return exit_if(uop, state, ExitKind.SIDE, z3.Not(cached))

    return handler


def guard_callable_type_object(name: str, depth: int) -> Handler:
    tag = type_tag(name)

    def handler(uop: Uop, state: State) -> State:
        callable_ = at_depth(state, uop, depth)
        is_type = z3.And(
            Slot.is_Ref(callable_), Slot.id(callable_) == type_object_id(z3.IntVal(tag))
        )
        return exit_if(uop, state, ExitKind.SIDE, z3.Not(is_type))

    return handler


def guard_callable_builtin_class(uop: Uop, state: State) -> State:
    callable_ = at_depth(state, uop, uop.oparg + 2)
    vectorcall_class = z3.And(
        is_type_object(state, callable_), type_vectorcall(Obj.ttag(slot_object(state, callable_)))
    )
    return exit_if(uop, state, ExitKind.SIDE, z3.Not(vectorcall_class))


def guard_callable_builtin(flags: int, total: int | None) -> Handler:
    def handler(uop: Uop, state: State) -> State:
        callable_ = at_depth(state, uop, uop.oparg + 2)
        self_or_null = at_depth(state, uop, uop.oparg + 1)
        parts = [
            has_tag(state, callable_, CFUNCTION_TAG),
            cfunc_flags(Slot.id(callable_)) == z3.IntVal(flags),
        ]
        if total is not None:
            total_args = z3.IntVal(uop.oparg) + z3.If(
                Slot.is_Null(self_or_null), z3.IntVal(0), z3.IntVal(1)
            )
            parts.append(total_args == z3.IntVal(total))
        return exit_if(uop, state, ExitKind.SIDE, z3.Not(z3.And(*parts)))

    return handler


def check_is_not_py_callable_at(depth_of: Callable[[Uop], int], methods_too: bool) -> Handler:
    def handler(uop: Uop, state: State) -> State:
        callable_ = at_depth(state, uop, depth_of(uop))
        python_callable = is_func(state, callable_)
        if methods_too:
            python_callable = z3.Or(python_callable, has_tag(state, callable_, METHOD_TAG))
        return exit_if(uop, state, ExitKind.SIDE, python_callable)

    return handler


check_is_not_py_callable = check_is_not_py_callable_at(lambda uop: uop.oparg + 2, True)


def delete_fast(uop: Uop, state: State) -> State:
    before = state
    slot = local_slot(uop, state)
    state = exit_if(uop, state, ExitKind.ERROR, Slot.is_Null(slot), handover=before)
    state = replace(state, ledger=decref_on_pop(state, slot))
    _, state = escape(state, uop.i, uop.op, z3.Not(harmless_to_close(state, slot)), False)
    return set_local(state, uop.oparg, Slot.Null)


HANDLERS: dict[str, Handler] = {
    "_BINARY_OP": binary_op,
    "_COMPARE_OP": compare_op,
    "_CONTAINS_OP": contains_op,
    "_CONTAINS_OP_SET": contains_op_set,
    "_CONTAINS_OP_DICT": contains_op_dict,
    "_BINARY_SLICE": binary_slice,
    "_BINARY_OP_SUBSCR_STR_INT": binary_op_subscr_str_int,
    "_BINARY_OP_SUBSCR_USTR_INT": binary_op_subscr_str_int,
    "_GUARD_BINARY_OP_EXTEND": guard_binary_op_extend("both"),
    "_GUARD_BINARY_OP_EXTEND_LHS": guard_binary_op_extend("lhs"),
    "_GUARD_BINARY_OP_EXTEND_RHS": guard_binary_op_extend("rhs"),
    "_BINARY_OP_EXTEND": binary_op_extend,
    "_UNARY_INVERT": unary_invert,
    "_UNARY_NEGATIVE": unary_negative,
    "_TO_BOOL": to_bool,
    "_TO_BOOL_NONE": to_bool_none,
    "_TO_BOOL_STR": to_bool_str,
    "_TO_BOOL_LIST": to_bool_list,
    "_CALL_LEN": call_len,
    "_CALL_ISINSTANCE": call_isinstance,
    "_CALL_TYPE_1": call_type_1,
    "_CALL_STR_1": call_str_1,
    "_CALL_TUPLE_1": call_tuple_1,
    "_CALL_LIST_APPEND": call_list_append,
    "_GUARD_CALLABLE_LEN": guard_callable_cached(BUILTIN_LEN, 3),
    "_GUARD_CALLABLE_ISINSTANCE": guard_callable_cached(BUILTIN_ISINSTANCE, 4),
    "_GUARD_CALLABLE_LIST_APPEND": guard_callable_cached(BUILTIN_LIST_APPEND, 3),
    "_GUARD_CALLABLE_TYPE_1": guard_callable_type_object("type", 3),
    "_GUARD_CALLABLE_STR_1": guard_callable_type_object("str", 3),
    "_GUARD_CALLABLE_TUPLE_1": guard_callable_type_object("tuple", 3),
    "_GUARD_CALLABLE_BUILTIN_CLASS": guard_callable_builtin_class,
    "_GUARD_CALLABLE_BUILTIN_O": guard_callable_builtin(0x8, 1),
    "_GUARD_CALLABLE_BUILTIN_FAST": guard_callable_builtin(0x80, None),
    "_GUARD_CALLABLE_BUILTIN_FAST_WITH_KEYWORDS": guard_callable_builtin(0x80 | 0x2, None),
    "_CHECK_IS_NOT_PY_CALLABLE": check_is_not_py_callable,
    "_CHECK_IS_NOT_PY_CALLABLE_KW": check_is_not_py_callable_at(lambda uop: uop.oparg + 3, True),
    "_CHECK_IS_NOT_PY_CALLABLE_EX": check_is_not_py_callable_at(lambda uop: 4, False),
    "_DELETE_FAST": delete_fast,
}
