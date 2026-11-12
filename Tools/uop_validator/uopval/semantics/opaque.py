from dataclasses import replace

import z3

from ..domain.trace import Uop
from ..symbolic.escape import escape
from ..symbolic.heap import expose
from ..symbolic.ledger import decref_on_pop
from ..symbolic.objects import bool_slot, is_immortal_id, none_slot
from ..symbolic.sorts import Slot
from ..symbolic.state import ExitKind, FactKind, State, push, with_constraint, with_fact
from . import uop_metadata
from .step import UnsupportedUop, exit_if
from .uninterpreted import key_of, value_projection

FRAME_UOPS = frozenset(
    {
        "_PUSH_FRAME",
        "_RETURN_VALUE",
        "_RETURN_GENERATOR",
        "_YIELD_VALUE",
        "_FOR_ITER_GEN_FRAME",
        "_SEND_GEN_FRAME",
        "_CREATE_INIT_FRAME",
        "_EXIT_INIT_CHECK",
        "_LOAD_ATTR_PROPERTY_FRAME",
        "_LOAD_ATTR_GETATTRIBUTE_OVERRIDDEN_FRAME",
        "_INIT_CALL_PY_EXACT_ARGS",
        "_PY_FRAME_GENERAL",
        "_PY_FRAME_KW",
        "_PY_FRAME_EX",
        "_BINARY_OP_SUBSCR_INIT_CALL",
        "_GUARD_CODE_VERSION__PUSH_FRAME",
        "_GUARD_CODE_VERSION_RETURN_VALUE",
        "_GUARD_CODE_VERSION_YIELD_VALUE",
        "_GUARD_CODE_VERSION_RETURN_GENERATOR",
        "_GUARD_IP__PUSH_FRAME",
        "_GUARD_IP_RETURN_VALUE",
        "_GUARD_IP_YIELD_VALUE",
        "_GUARD_IP_RETURN_GENERATOR",
        "_SAVE_RETURN_OFFSET",
        "_COPY_FREE_VARS",
        "_LOAD_DEREF",
        "_STORE_DEREF",
        "_MAKE_CELL",
        "_DELETE_DEREF",
        "_LOAD_FROM_DICT_OR_DEREF",
        "_CHECK_STACK_SPACE",
        "_CHECK_STACK_SPACE_OPERAND",
        "_CHECK_FUNCTION_VERSION",
        "_CHECK_FUNCTION_EXACT_ARGS",
        "_CHECK_RECURSION_REMAINING",
        "_CHECK_RECURSION_LIMIT",
    }
)


def opaque(uop: Uop, state: State) -> State:
    info = uop_metadata.info(uop.op)
    if info is None:
        raise UnsupportedUop(
            f"{uop.op} at {uop.i} has no semantics in the model and no DSL metadata"
        )
    if not opaque_eligible(uop.op):
        raise UnsupportedUop(
            f"{uop.op} at {uop.i} has no semantics in the model and cannot be opaque"
        )
    n_in, n_out = info.popped(uop.oparg), info.pushed(uop.oparg)
    if len(state.stack) < n_in:
        raise UnsupportedUop(f"{uop.op} at {uop.i}: stack too shallow for {n_in} inputs")
    before = state
    inputs = state.stack[len(state.stack) - n_in :] if n_in else ()
    key = key_of(uop)
    args: list[z3.ExprRef] = [value_projection(s) for s in inputs] + [state.heap, state.epoch]
    sorts = [Slot] * len(inputs) + [state.heap.sort(), z3.IntSort()]
    if info.has("EXIT") or info.has("DEOPT"):
        exit_fn = z3.Function(f"exit_{key}", *sorts, z3.BoolSort())
        state = exit_if(
            uop,
            state,
            ExitKind.SIDE if info.has("EXIT") else ExitKind.DEOPT,
            exit_fn(*args),
            handover=before,
        )
    raises: z3.BoolRef | None = None
    if info.has("ESCAPES"):
        state = expose(state, [s for s in inputs])
        raises, state = escape(state, uop.i, uop.op, z3.BoolVal(True), info.has("ERROR"))
    elif info.has("ERROR"):
        error_fn = z3.Function(f"error_{key}", *sorts, z3.BoolSort())
        raises = error_fn(*args)
    if raises is not None:
        state = exit_if(uop, state, ExitKind.ERROR, raises, handover=before)
    names: list[str] = []
    for item in info.inputs:
        names.extend([item.name] * item.count(uop.oparg))
    passthrough_sources = set(info.passthrough.values())
    kept = state
    for name, slot in zip(names, inputs, strict=True):
        if name in passthrough_sources or name == "unused":
            continue
        kept = replace(kept, ledger=decref_on_pop(kept, slot))
    state = replace(kept, stack=state.stack[: len(state.stack) - n_in])
    by_name: dict[str, z3.DatatypeRef] = {}
    for name, slot in zip(names, inputs, strict=True):
        by_name.setdefault(name, slot)
    outputs: list[z3.DatatypeRef] = []
    for item in info.outputs:
        count = item.count(uop.oparg)
        for _ in range(count):
            index = len(outputs)
            if item.name in info.passthrough and count == 1:
                outputs.append(by_name[info.passthrough[item.name]])
            elif item.name in by_name and index < len(inputs) and names[index] == item.name:
                outputs.append(inputs[index])
            else:
                out_fn = z3.Function(f"out_{key}_{index}", *sorts, Slot)
                value = out_fn(*args)
                state = with_constraint(
                    state,
                    z3.Not(Slot.is_TaggedInt(value)),
                    z3.Implies(
                        z3.And(Slot.is_Ref(value), is_immortal_id(state.consts, Slot.id(value))),
                        Slot.borrowed(value),
                    ),
                )
                kind = info.constant_outputs.get(item.name)
                if kind == "bool":
                    state = with_constraint(
                        state, z3.Or(value == bool_slot(True), value == bool_slot(False))
                    )
                elif kind == "none":
                    state = with_constraint(state, value == none_slot())
                elif kind == "null":
                    state = with_constraint(state, value == Slot.Null)
                outputs.append(value)
    for value in outputs:
        state = push(state, value)
    return with_fact(
        state,
        uop.i,
        FactKind.OPAQUE,
        uop.op,
        None,
        f"{uop.op}: {n_in} popped, {n_out} pushed, uninterpreted",
    )


def opaque_eligible(op: str) -> bool:
    info = uop_metadata.info(op)
    if info is None or op in FRAME_UOPS:
        return False
    return not (info.has("LOCAL") or info.has("CONST") or info.has("FREE"))
