from collections.abc import Sequence

import z3

from .sorts import Slot
from .state import LedgerEntry, State


def ledger_at(ledger: Sequence[LedgerEntry], key: z3.ArithRef) -> z3.ArithRef:
    terms: list[z3.ArithRef] = []
    key_value = key.as_long() if z3.is_int_value(key) else None
    for when, k, delta in ledger:
        if k.eq(key):
            terms.append(z3.If(when, z3.IntVal(delta), z3.IntVal(0)))
        elif key_value is not None and z3.is_int_value(k):
            if k.as_long() == key_value:
                terms.append(z3.If(when, z3.IntVal(delta), z3.IntVal(0)))
        else:
            terms.append(z3.If(z3.And(when, k == key), z3.IntVal(delta), z3.IntVal(0)))
    if not terms:
        return z3.IntVal(0)
    return z3.Sum(terms) if len(terms) > 1 else terms[0]


def ownership(slot: z3.DatatypeRef, depth: int = 6) -> tuple[z3.BoolRef, z3.ArithRef]:
    if z3.is_app(slot) and slot.decl().eq(Slot.Ref):
        ident, borrowed = slot.arg(0), slot.arg(1)
        if z3.is_true(borrowed):
            return z3.BoolVal(False), ident
        if z3.is_false(borrowed):
            return z3.BoolVal(True), ident
        return z3.Not(borrowed), ident
    if depth > 0 and z3.is_app_of(slot, z3.Z3_OP_ITE):
        condition = slot.arg(0)
        then_owned, then_key = ownership(slot.arg(1), depth - 1)
        else_owned, else_key = ownership(slot.arg(2), depth - 1)
        owned = (
            then_owned if then_owned.eq(else_owned) else z3.If(condition, then_owned, else_owned)
        )
        key = then_key if then_key.eq(else_key) else z3.If(condition, then_key, else_key)
        return owned, key
    return z3.And(Slot.is_Ref(slot), z3.Not(Slot.borrowed(slot))), Slot.id(slot)


def conjunction(when: z3.BoolRef, other: z3.BoolRef) -> z3.BoolRef:
    if z3.is_true(when):
        return other
    if z3.is_true(other):
        return when
    if z3.is_false(when) or z3.is_false(other):
        return z3.BoolVal(False)
    return z3.And(when, other)


def ledger_add(
    ledger: tuple[LedgerEntry, ...], when: z3.BoolRef, key: z3.ArithRef, delta: int
) -> tuple[LedgerEntry, ...]:
    when = z3.simplify(when)
    if z3.is_false(when):
        return ledger
    return ledger + ((when, z3.simplify(key), delta),)


def decref_on_pop(state: State, slot: z3.DatatypeRef) -> tuple[LedgerEntry, ...]:
    closes, ident = ownership(slot)
    return ledger_add(state.ledger, closes, ident, 1)


def decref_on_pop_if(
    state: State, when: z3.BoolRef, slot: z3.DatatypeRef
) -> tuple[LedgerEntry, ...]:
    closes, ident = ownership(slot)
    return ledger_add(state.ledger, conjunction(when, closes), ident, 1)


def incref_on_dup(state: State, slot: z3.DatatypeRef) -> tuple[LedgerEntry, ...]:
    counts, ident = ownership(slot)
    return ledger_add(state.ledger, counts, ident, -1)
