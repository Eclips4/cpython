from collections.abc import Sequence

import z3

from ..symbolic.ledger import ledger_at, ownership
from ..symbolic.sorts import Slot
from ..symbolic.state import Frame, LedgerEntry, State


def same_slot(a: z3.DatatypeRef, b: z3.DatatypeRef) -> z3.BoolRef:
    if a.eq(b):
        return z3.BoolVal(True)
    return z3.Or(
        z3.And(Slot.is_Null(a), Slot.is_Null(b)),
        z3.And(Slot.is_TaggedInt(a), Slot.is_TaggedInt(b), Slot.k(a) == Slot.k(b)),
        z3.And(Slot.is_Ref(a), Slot.is_Ref(b), Slot.id(a) == Slot.id(b)),
    )


KEY_CACHE: dict[int, tuple[z3.ExprRef, ...]] = {}


def array_children(term: z3.ExprRef) -> list[z3.ExprRef]:
    if z3.is_store(term):
        return [term.arg(0)]
    if z3.is_app_of(term, z3.Z3_OP_ITE):
        return [term.arg(1), term.arg(2)]
    return []


def array_keys(term: z3.ExprRef) -> list[z3.ExprRef]:
    stack: list[tuple[z3.ExprRef, bool]] = [(term, False)]
    while stack:
        t, expanded = stack.pop()
        ident = t.get_id()
        if ident in KEY_CACHE:
            continue
        children = array_children(t)
        if not expanded:
            stack.append((t, True))
            for child in children:
                if child.get_id() not in KEY_CACHE:
                    stack.append((child, False))
            continue
        merged: dict[int, z3.ExprRef] = {}
        for child in children:
            for key in KEY_CACHE[child.get_id()]:
                merged.setdefault(key.get_id(), key)
        if z3.is_store(t):
            merged.setdefault(t.arg(1).get_id(), t.arg(1))
        KEY_CACHE[ident] = tuple(merged.values())
    return list(KEY_CACHE[term.get_id()])


def arrays_equal(
    a: z3.ArrayRef, b: z3.ArrayRef, extra_keys: Sequence[z3.ExprRef] = ()
) -> z3.BoolRef:
    if a.eq(b):
        return z3.BoolVal(True)
    keys = array_keys(a) + array_keys(b) + list(extra_keys)
    unique: dict[int, z3.ExprRef] = {}
    for k in keys:
        unique.setdefault(k.get_id(), k)
    parts = [z3.Select(a, k) == z3.Select(b, k) for k in unique.values()]
    return z3.And(*parts) if parts else z3.BoolVal(True)


def owned_slots(state: State) -> list[z3.DatatypeRef]:
    slots = list(state.locals) + list(state.stack) + [state.func]
    for frame in state.callers:
        slots += list(frame.locals) + list(frame.stack) + [frame.func]
    if state.pending_frame is not None:
        slots += (
            list(state.pending_frame.locals)
            + list(state.pending_frame.stack)
            + [state.pending_frame.func]
        )
    return slots


def contributions(state: State) -> list[LedgerEntry]:
    entries = list(state.ledger)
    for slot in owned_slots(state):
        owned, ident = ownership(slot)
        owned = z3.simplify(owned)
        if not z3.is_false(owned):
            entries.append((owned, z3.simplify(ident), 1))
    return entries


def grouped(
    entries: Sequence[LedgerEntry],
) -> dict[tuple[int, int], tuple[z3.BoolRef, z3.ArithRef, int]]:
    out: dict[tuple[int, int], tuple[z3.BoolRef, z3.ArithRef, int]] = {}
    for when, key, delta in entries:
        group = (when.get_id(), key.get_id())
        previous = out.get(group)
        out[group] = (when, key, delta + (previous[2] if previous is not None else 0))
    return out


def balance_differences(a: State, b: State) -> list[LedgerEntry]:
    ga, gb = grouped(contributions(a)), grouped(contributions(b))
    out: list[LedgerEntry] = []
    for group in list(ga) + [g for g in gb if g not in ga]:
        when, key, _ = ga[group] if group in ga else gb[group]
        delta = (ga[group][2] if group in ga else 0) - (gb[group][2] if group in gb else 0)
        if delta != 0:
            out.append((when, key, delta))
    return out


def balance_at(state: State, key: z3.ExprRef) -> z3.ArithRef:
    return ledger_at(contributions(state), key)


def balances_equal(a: State, b: State) -> z3.BoolRef:
    differences = balance_differences(a, b)
    if not differences:
        return z3.BoolVal(True)
    by_key: dict[int, tuple[z3.ArithRef, list[z3.ArithRef]]] = {}
    for when, key, delta in differences:
        by_key.setdefault(key.get_id(), (key, []))[1].append(
            z3.If(when, z3.IntVal(delta), z3.IntVal(0))
        )
    groups = [
        (key, z3.Sum(terms) if len(terms) > 1 else terms[0]) for key, terms in by_key.values()
    ]
    parts: list[z3.BoolRef] = []
    for key, own in groups:
        total: z3.ArithRef = own
        for other_key, other_sum in groups:
            if other_key.eq(key):
                continue
            if z3.is_int_value(key) and z3.is_int_value(other_key):
                continue
            total = total + z3.If(other_key == key, other_sum, z3.IntVal(0))
        parts.append(total == z3.IntVal(0))
    return z3.And(*parts)


def frames_equal(a: Frame, b: Frame, with_stack: bool) -> z3.BoolRef:
    if len(a.locals) != len(b.locals) or (with_stack and len(a.stack) != len(b.stack)):
        return z3.BoolVal(False)
    parts: list[z3.BoolRef] = [same_slot(x, y) for x, y in zip(a.locals, b.locals, strict=True)]
    if with_stack:
        parts += [same_slot(x, y) for x, y in zip(a.stack, b.stack, strict=True)]
        parts.append(same_slot(a.func, b.func))
        if a.return_offset is not None and b.return_offset is not None:
            parts.append(a.return_offset == b.return_offset)
        elif (a.return_offset is None) != (b.return_offset is None):
            return z3.BoolVal(False)
    parts += [a.code == b.code, a.ip == b.ip]
    return z3.And(*parts)


def states_equal(a: State, b: State, with_ip: bool) -> z3.BoolRef:
    if (
        len(a.stack) != len(b.stack)
        or len(a.locals) != len(b.locals)
        or len(a.callers) != len(b.callers)
    ):
        return z3.BoolVal(False)
    if (a.pending_frame is None) != (b.pending_frame is None):
        return z3.BoolVal(False)
    parts: list[z3.BoolRef] = [same_slot(x, y) for x, y in zip(a.locals, b.locals, strict=True)]
    parts += [same_slot(x, y) for x, y in zip(a.stack, b.stack, strict=True)]
    slot_ids = [Slot.id(s) for s in owned_slots(a) + owned_slots(b)]
    parts += [
        arrays_equal(a.heap, b.heap, slot_ids),
        balances_equal(a, b),
        a.valid == b.valid,
        a.epoch == b.epoch,
    ]
    parts += [frames_equal(x, y, True) for x, y in zip(a.callers, b.callers, strict=True)]
    if a.pending_frame is not None and b.pending_frame is not None:
        parts.append(frames_equal(a.pending_frame, b.pending_frame, True))
    parts += [
        a.code == b.code,
        a.stack_space == b.stack_space,
        a.recursion_remaining == b.recursion_remaining,
        a.globals_version == b.globals_version,
    ]
    if with_ip:
        parts.append(a.ip == b.ip)
    return z3.And(*parts)


def observable_equal(a: State, b: State) -> z3.BoolRef:
    if len(a.locals) != len(b.locals) or len(a.callers) != len(b.callers):
        return z3.BoolVal(False)
    parts: list[z3.BoolRef] = [same_slot(x, y) for x, y in zip(a.locals, b.locals, strict=True)]
    slot_ids = [Slot.id(s) for s in list(a.locals) + list(b.locals)]
    parts += [
        arrays_equal(a.heap, b.heap, slot_ids),
        a.valid == b.valid,
        a.ip == b.ip,
        a.epoch == b.epoch,
    ]
    parts += [frames_equal(x, y, False) for x, y in zip(a.callers, b.callers, strict=True)]
    parts.append(a.code == b.code)
    return z3.And(*parts)
