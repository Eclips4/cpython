from dataclasses import replace

import z3

from .functions import immortal_constant, pinned
from .identities import EMPTY_STR_ID
from .objects import is_immortal_id, is_small_value
from .sorts import Obj, Slot
from .state import FactKind, State, all_slots, with_fact


def fresh_distinctness(state: State, fresh: z3.ArithRef) -> list[z3.BoolRef]:
    consts = state.consts
    parts: list[z3.BoolRef] = [
        fresh > consts.small_max,
        z3.Not(immortal_constant(fresh)),
        z3.Not(pinned(fresh)),
    ]
    parts += [fresh != other for other in state.ids]
    parts += [fresh != Slot.id(other) for other, _ in state.exposed]
    for slot in all_slots(state) + list(state.popped):
        parts.append(z3.Implies(Slot.is_Ref(slot), fresh != Slot.id(slot)))
    return parts


def new_str(
    state: State, text: z3.SeqRef, position: int, suffix: str, when: z3.BoolRef | None = None
) -> tuple[z3.DatatypeRef, z3.BoolRef, State]:
    empty = z3.Length(text) == 0
    stored = empty if when is None else z3.Or(empty, z3.Not(when))
    slot, failure, new = new_object(state, Obj.Str(text), position, suffix, z3.Not(stored))
    result = z3.If(empty, Slot.Ref(z3.IntVal(EMPTY_STR_ID), z3.BoolVal(True)), slot)
    return result, z3.And(z3.Not(empty), failure), new


def new_int(
    state: State, value: z3.ArithRef, position: int, when: z3.BoolRef | None = None
) -> tuple[z3.DatatypeRef, z3.BoolRef, State]:
    consts = state.consts
    fresh = z3.Int(f"fresh_p{position}")
    ok = z3.Bool(f"alloc_p{position}")
    small = is_small_value(consts, value)
    slot = z3.If(small, Slot.Ref(value, z3.BoolVal(True)), Slot.Ref(fresh, z3.BoolVal(False)))
    heap = z3.If(small, state.heap, z3.Store(state.heap, fresh, Obj.ExactInt(value)))
    if when is not None:
        heap = z3.If(when, heap, state.heap)
    failure = z3.And(z3.Not(small), z3.Not(ok))
    distinct = fresh_distinctness(state, fresh)
    new = replace(
        state,
        heap=heap,
        ids=state.ids + (fresh,),
        fresh_ids=state.fresh_ids + (fresh,),
        fresh_count=state.fresh_count + 1,
        alloc_count=state.alloc_count + 1,
        oracle=state.oracle + (ok,),
        constraints=state.constraints + tuple(distinct),
    )
    new = with_fact(
        new,
        position,
        FactKind.ALLOCATION,
        "",
        None,
        f"fresh_p{position} allocated unless the value is small; alloc_p{position} is the oracle's "
        "success bit",
    )
    return slot, failure, new


def new_object(
    state: State,
    obj: z3.DatatypeRef,
    position: int,
    suffix: str = "",
    when: z3.BoolRef | None = None,
) -> tuple[z3.DatatypeRef, z3.BoolRef, State]:
    fresh = z3.Int(f"fresh_p{position}{suffix}")
    ok = z3.Bool(f"alloc_p{position}{suffix}")
    slot = Slot.Ref(fresh, z3.BoolVal(False))
    heap: z3.ArrayRef = z3.Store(state.heap, fresh, obj)
    if when is not None:
        heap = z3.If(when, heap, state.heap)
    failure = z3.Not(ok)
    distinct = fresh_distinctness(state, fresh)
    new = replace(
        state,
        heap=heap,
        ids=state.ids + (fresh,),
        fresh_ids=state.fresh_ids + (fresh,),
        fresh_count=state.fresh_count + 1,
        alloc_count=state.alloc_count + 1,
        oracle=state.oracle + (ok,),
        constraints=state.constraints + tuple(distinct),
    )
    new = with_fact(
        new,
        position,
        FactKind.ALLOCATION,
        "",
        None,
        f"fresh_p{position}{suffix} allocated; alloc_p{position}{suffix} is the oracle's success "
        "bit",
    )
    return slot, failure, new


def expose(state: State, slots: list[z3.DatatypeRef], when: z3.BoolRef | None = None) -> State:
    condition = z3.BoolVal(True) if when is None else when
    return replace(state, exposed=state.exposed + tuple((s, condition) for s in slots))


def is_unique(state: State, slot: z3.DatatypeRef, exclude: list[z3.DatatypeRef]) -> z3.BoolRef:
    ident = Slot.id(slot)
    parts: list[z3.BoolRef] = [
        Slot.is_Ref(slot),
        z3.Not(Slot.borrowed(slot)),
        z3.Not(is_immortal_id(state.consts, ident)),
    ]
    parts.append(
        z3.Or(*[ident == f for f in state.fresh_ids]) if state.fresh_ids else z3.BoolVal(False)
    )
    parts += [ident != Slot.id(e) for e, _ in state.exposed]
    excluded = {id(s) for s in exclude}
    for other in all_slots(state):
        if id(other) in excluded:
            continue
        parts.append(z3.Not(z3.And(Slot.is_Ref(other), Slot.id(other) == ident)))
    return z3.And(*parts)
