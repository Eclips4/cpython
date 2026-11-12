import z3

from ..domain.build import IntConstants
from .functions import immortal_constant
from .identities import EMPTY_STR_ID, FALSE_ID, NONE_ID, TRUE_ID
from .sorts import Obj, Slot
from .state import State


def is_small_id(consts: IntConstants, ident: z3.ArithRef) -> z3.BoolRef:
    return z3.And(ident >= consts.small_min, ident <= consts.small_max)


def is_small_value(consts: IntConstants, value: z3.ArithRef) -> z3.BoolRef:
    return z3.And(value >= consts.small_min, value <= consts.small_max)


def is_compact(consts: IntConstants, value: z3.ArithRef) -> z3.BoolRef:
    return z3.And(value > -consts.compact_bound, value < consts.compact_bound)


def is_bool_id(ident: z3.ArithRef) -> z3.BoolRef:
    return z3.Or(ident == TRUE_ID, ident == FALSE_ID)


def is_none_id(ident: z3.ArithRef) -> z3.BoolRef:
    return ident == NONE_ID


def is_immortal_id(consts: IntConstants, ident: z3.ArithRef) -> z3.BoolRef:
    return z3.Or(
        is_small_id(consts, ident),
        is_bool_id(ident),
        is_none_id(ident),
        ident == EMPTY_STR_ID,
        immortal_constant(ident),
    )


def borrowed_if_immortal(state: State, slot: z3.DatatypeRef) -> z3.BoolRef:
    immortal = z3.And(Slot.is_Ref(slot), is_immortal_id(state.consts, Slot.id(slot)))
    return z3.Implies(immortal, Slot.borrowed(slot))


def object_of(state: State, ident: z3.ArithRef) -> z3.DatatypeRef:
    return z3.Select(state.heap, ident)


def is_exact_int(state: State, ident: z3.ArithRef) -> z3.BoolRef:
    heap_object = object_of(state, ident)
    return z3.Or(
        is_small_id(state.consts, ident),
        z3.And(
            Obj.is_ExactInt(heap_object), z3.Not(is_small_value(state.consts, Obj.val(heap_object)))
        ),
    )


def int_value(state: State, ident: z3.ArithRef) -> z3.ArithRef:
    return z3.If(is_small_id(state.consts, ident), ident, Obj.val(object_of(state, ident)))


def is_range_iter(state: State, ident: z3.ArithRef) -> z3.BoolRef:
    return z3.And(
        z3.Not(is_small_id(state.consts, ident)), Obj.is_RangeIter(object_of(state, ident))
    )


def slot_is_exact_int(state: State, slot: z3.DatatypeRef) -> z3.BoolRef:
    return z3.And(Slot.is_Ref(slot), is_exact_int(state, Slot.id(slot)))


def slot_is_intlike(state: State, slot: z3.DatatypeRef) -> z3.BoolRef:
    ident = Slot.id(slot)
    return z3.And(Slot.is_Ref(slot), z3.Or(is_exact_int(state, ident), is_bool_id(ident)))


def slot_intlike_value(state: State, slot: z3.DatatypeRef) -> z3.ArithRef:
    ident = Slot.id(slot)
    return z3.If(
        ident == TRUE_ID,
        z3.IntVal(1),
        z3.If(ident == FALSE_ID, z3.IntVal(0), int_value(state, ident)),
    )


def slot_int_value(state: State, slot: z3.DatatypeRef) -> z3.ArithRef:
    return int_value(state, Slot.id(slot))


def slot_is_bool(slot: z3.DatatypeRef) -> z3.BoolRef:
    return z3.And(Slot.is_Ref(slot), is_bool_id(Slot.id(slot)))


def slot_is_none(slot: z3.DatatypeRef) -> z3.BoolRef:
    return z3.And(Slot.is_Ref(slot), is_none_id(Slot.id(slot)))


def slot_is(slot: z3.DatatypeRef, ident: int) -> z3.BoolRef:
    return z3.And(Slot.is_Ref(slot), Slot.id(slot) == ident)


def slot_kind_is(state: State, slot: z3.DatatypeRef, recognizer: z3.FuncDeclRef) -> z3.BoolRef:
    return z3.And(
        Slot.is_Ref(slot),
        z3.Not(is_small_id(state.consts, Slot.id(slot))),
        recognizer(object_of(state, Slot.id(slot))),
    )


def slot_object(state: State, slot: z3.DatatypeRef) -> z3.DatatypeRef:
    return object_of(state, Slot.id(slot))


def bool_slot(value: bool) -> z3.DatatypeRef:
    return Slot.Ref(z3.IntVal(TRUE_ID if value else FALSE_ID), z3.BoolVal(True))


def none_slot() -> z3.DatatypeRef:
    return Slot.Ref(z3.IntVal(NONE_ID), z3.BoolVal(True))


def bool_of(truth: z3.BoolRef) -> z3.DatatypeRef:
    return z3.If(truth, bool_slot(True), bool_slot(False))


def singleton_facts(heap: z3.ArrayRef) -> list[z3.BoolRef]:
    return [
        z3.Select(heap, z3.IntVal(TRUE_ID)) == Obj.Bool(z3.BoolVal(True)),
        z3.Select(heap, z3.IntVal(FALSE_ID)) == Obj.Bool(z3.BoolVal(False)),
        z3.Select(heap, z3.IntVal(NONE_ID)) == Obj.NoneObj,
        z3.Select(heap, z3.IntVal(EMPTY_STR_ID)) == Obj.Str(z3.StringVal("")),
    ]


def canonical_str(ident: z3.ArithRef, obj: z3.DatatypeRef) -> z3.BoolRef:
    return z3.And(Obj.is_Str(obj), z3.Or(z3.Length(Obj.sval(obj)) > 0, ident == EMPTY_STR_ID))


def is_float(state: State, slot: z3.DatatypeRef) -> z3.BoolRef:
    return slot_kind_is(state, slot, Obj.is_Float)


def is_str(state: State, slot: z3.DatatypeRef) -> z3.BoolRef:
    return z3.And(
        slot_kind_is(state, slot, Obj.is_Str),
        canonical_str(Slot.id(slot), object_of(state, Slot.id(slot))),
    )


def is_tuple(state: State, slot: z3.DatatypeRef) -> z3.BoolRef:
    return slot_kind_is(state, slot, Obj.is_Tuple)


def is_func(state: State, slot: z3.DatatypeRef) -> z3.BoolRef:
    return slot_kind_is(state, slot, Obj.is_Func)


def is_type_object(state: State, slot: z3.DatatypeRef) -> z3.BoolRef:
    return slot_kind_is(state, slot, Obj.is_TypeObj)


def float_value(state: State, slot: z3.DatatypeRef) -> z3.FPRef:
    return Obj.fval(slot_object(state, slot))


def str_value(state: State, slot: z3.DatatypeRef) -> z3.SeqRef:
    return Obj.sval(slot_object(state, slot))


def plain_kind(state: State, slot: z3.DatatypeRef) -> z3.BoolRef:
    heap_object = slot_object(state, slot)
    return z3.Or(
        slot_is_exact_int(state, slot),
        slot_is_bool(slot),
        slot_is_none(slot),
        z3.And(Slot.is_Ref(slot), z3.Or(Obj.is_Float(heap_object), Obj.is_Str(heap_object))),
    )
