import z3

from .functions import has_known_length, known_length, pinned
from .objects import is_immortal_id, is_small_id, object_of, slot_kind_is, slot_object
from .sorts import Obj, Slot
from .state import State


def harmless_to_close(state: State, slot: z3.DatatypeRef) -> z3.BoolRef:
    ident = Slot.id(slot)
    heap_object = object_of(state, ident)
    plain = z3.Or(
        Obj.is_ExactInt(heap_object),
        Obj.is_Bool(heap_object),
        Obj.is_NoneObj(heap_object),
        Obj.is_Float(heap_object),
        Obj.is_Str(heap_object),
        Obj.is_RangeIter(heap_object),
    )
    return z3.Or(
        Slot.is_Null(slot),
        Slot.borrowed(slot),
        is_immortal_id(state.consts, ident),
        pinned(ident),
        plain,
    )


def truth(state: State, slot: z3.DatatypeRef, position: int) -> z3.BoolRef:
    ident = Slot.id(slot)
    heap_object = object_of(state, ident)
    unknown = z3.Function(
        "truth_other", z3.IntSort(), state.heap.sort(), z3.IntSort(), z3.BoolSort()
    )
    return z3.If(
        is_small_id(state.consts, ident),
        ident != 0,
        z3.If(
            Obj.is_ExactInt(heap_object),
            Obj.val(heap_object) != 0,
            z3.If(
                Obj.is_Bool(heap_object),
                Obj.b(heap_object),
                z3.If(
                    Obj.is_NoneObj(heap_object),
                    z3.BoolVal(False),
                    z3.If(
                        Obj.is_Float(heap_object),
                        z3.Not(z3.fpIsZero(Obj.fval(heap_object))),
                        z3.If(
                            Obj.is_Str(heap_object),
                            z3.Length(Obj.sval(heap_object)) != 0,
                            z3.If(
                                Obj.is_Tuple(heap_object),
                                Obj.tlen(heap_object) != 0,
                                unknown(ident, state.heap, state.epoch),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )


def length_of(state: State, slot: z3.DatatypeRef) -> z3.ArithRef:
    heap_object = slot_object(state, slot)
    ident = Slot.id(slot)
    unknown = z3.Function(
        "length_other", z3.IntSort(), state.heap.sort(), z3.IntSort(), z3.IntSort()
    )
    return z3.If(
        Obj.is_Str(heap_object),
        z3.Length(Obj.sval(heap_object)),
        z3.If(
            Obj.is_Tuple(heap_object),
            Obj.tlen(heap_object),
            z3.If(
                has_known_length(ident),
                known_length(ident),
                unknown(ident, state.heap, state.epoch),
            ),
        ),
    )


def length_known(state: State, slot: z3.DatatypeRef) -> z3.BoolRef:
    return z3.Or(
        slot_kind_is(state, slot, Obj.is_Str),
        slot_kind_is(state, slot, Obj.is_Tuple),
        z3.And(Slot.is_Ref(slot), has_known_length(Slot.id(slot))),
    )
