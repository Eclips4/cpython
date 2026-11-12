import z3

from ..domain.trace import Uop
from ..symbolic.objects import is_immortal_id
from ..symbolic.sorts import Slot
from ..symbolic.state import State, with_constraint


def uf_slot(name: str, args: list[z3.ExprRef]) -> z3.DatatypeRef:
    fn = z3.Function(name, *[a.sort() for a in args], Slot)
    return fn(*args)


def uf_bool(name: str, args: list[z3.ExprRef]) -> z3.BoolRef:
    fn = z3.Function(name, *[a.sort() for a in args], z3.BoolSort())
    return fn(*args)


def uf_int(name: str, args: list[z3.ExprRef]) -> z3.ArithRef:
    fn = z3.Function(name, *[a.sort() for a in args], z3.IntSort())
    return fn(*args)


def value_projection(slot: z3.DatatypeRef) -> z3.DatatypeRef:
    return z3.If(Slot.is_Ref(slot), Slot.Ref(Slot.id(slot), z3.BoolVal(False)), slot)


SHARED_SEMANTICS = {
    "_STORE_SUBSCR_DICT_KNOWN_HASH": "_STORE_SUBSCR_DICT",
    "_BINARY_OP_SUBSCR_DICT_KNOWN_HASH": "_BINARY_OP_SUBSCR_DICT",
}


def key_of(uop: Uop) -> str:
    if uop.op in SHARED_SEMANTICS or uop.op in SHARED_SEMANTICS.values():
        return f"{SHARED_SEMANTICS.get(uop.op, uop.op)}_p{uop.i}"
    return f"{uop.op}_p{uop.i}_{uop.oparg}_{uop.operand0:x}_{uop.operand1:x}"


def unknown_result(
    state: State, uop: Uop, inputs: list[z3.DatatypeRef], index: int
) -> tuple[z3.DatatypeRef, State]:
    args: list[z3.ExprRef] = [value_projection(s) for s in inputs] + [state.heap, state.epoch]
    value = uf_slot(f"out_{key_of(uop)}_{index}", args)
    state = with_constraint(
        state,
        Slot.is_Ref(value),
        z3.Implies(is_immortal_id(state.consts, Slot.id(value)), Slot.borrowed(value)),
    )
    return value, state


def unknown_error(state: State, uop: Uop, inputs: list[z3.DatatypeRef]) -> z3.BoolRef:
    args: list[z3.ExprRef] = [value_projection(s) for s in inputs] + [state.heap, state.epoch]
    return uf_bool(f"error_{key_of(uop)}", args)
