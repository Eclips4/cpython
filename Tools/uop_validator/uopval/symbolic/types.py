import z3

from .functions import metatype, type_of_version, type_version
from .identities import (
    BOOL_TAG,
    FLOAT_TAG,
    FUNCTION_TAG,
    INSTANCE_TAGS,
    INT_TAG,
    METHOD_DESCRIPTOR_FLAG_TAGS,
    MODELLED_TAGS,
    NONE_TAG,
    RANGE_ITER_TAG,
    STR_TAG,
    TAG_SHIFT,
    TUPLE_TAG,
    type_tag,
)
from .objects import canonical_str, is_small_id, is_small_value, object_of
from .sorts import Obj, Slot
from .state import State, with_constraint


def sanitized_tag(tag: z3.ArithRef) -> z3.ArithRef:
    clash = z3.Or(*[tag == t for t in MODELLED_TAGS])
    return z3.If(clash, tag + TAG_SHIFT, tag)


def sanitized_metatype(tag: z3.ArithRef) -> z3.ArithRef:
    clash = z3.Or(*[tag == t for t in MODELLED_TAGS + INSTANCE_TAGS])
    return z3.If(clash, tag + TAG_SHIFT, tag)


def type_tag_of(state: State, ident: z3.ArithRef) -> z3.ArithRef:
    obj = object_of(state, ident)
    cases: list[tuple[z3.BoolRef, z3.ArithRef]] = [
        (is_small_id(state.consts, ident), z3.IntVal(INT_TAG)),
        (
            z3.And(Obj.is_ExactInt(obj), z3.Not(is_small_value(state.consts, Obj.val(obj)))),
            z3.IntVal(INT_TAG),
        ),
        (Obj.is_Bool(obj), z3.IntVal(BOOL_TAG)),
        (Obj.is_NoneObj(obj), z3.IntVal(NONE_TAG)),
        (Obj.is_Float(obj), z3.IntVal(FLOAT_TAG)),
        (canonical_str(ident, obj), z3.IntVal(STR_TAG)),
        (Obj.is_Tuple(obj), z3.IntVal(TUPLE_TAG)),
        (Obj.is_RangeIter(obj), z3.IntVal(RANGE_ITER_TAG)),
        (Obj.is_Func(obj), z3.IntVal(FUNCTION_TAG)),
        (Obj.is_TypeObj(obj), sanitized_metatype(metatype(Obj.ttag(obj)))),
        (Obj.is_Cell(obj), z3.IntVal(type_tag("cell"))),
    ]
    result: z3.ArithRef = sanitized_tag(Obj.tag(obj))
    for condition, value in reversed(cases):
        result = z3.If(condition, value, result)
    return result


def has_tag(state: State, slot: z3.DatatypeRef, tag: int) -> z3.BoolRef:
    return z3.And(Slot.is_Ref(slot), type_tag_of(state, Slot.id(slot)) == tag)


def has_method_descriptor_flag(state: State, ident: z3.ArithRef) -> z3.BoolRef:
    tag = type_tag_of(state, ident)
    return z3.Or(
        Obj.is_Func(z3.Select(state.heap, ident)), *(tag == t for t in METHOD_DESCRIPTOR_FLAG_TAGS)
    )


def version_check(state: State, tag: z3.ArithRef, version: int) -> tuple[z3.BoolRef, State]:
    v = z3.IntVal(version)
    matches = type_version(tag) == v
    if version != 0:
        state = with_constraint(state, z3.Implies(matches, tag == type_of_version(v)))
    return matches, state
