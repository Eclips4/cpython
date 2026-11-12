from dataclasses import replace

import z3

from .sorts import Slot
from .state import FactKind, State, with_fact


def escape(
    state: State, position: int, uop: str, happens: z3.BoolRef, can_raise: bool
) -> tuple[z3.BoolRef | None, State]:
    k = state.escape_count
    invalidate = z3.Bool(f"escape_p{position}_invalidate")
    havoc = z3.Bool(f"escape_p{position}_havoc")
    raises = z3.Bool(f"escape_p{position}_raise") if can_raise else None
    mutate = z3.Bool(f"escape_p{position}_mutate")
    replaced = tuple(
        z3.If(z3.And(happens, havoc), z3.Const(f"local{j}_after_escape_p{position}", Slot), old)
        for j, old in enumerate(state.locals)
    )
    callers = tuple(
        replace(
            frame,
            locals=tuple(
                z3.If(
                    z3.And(happens, havoc),
                    z3.Const(f"caller{d}_local{j}_after_escape_p{position}", Slot),
                    old,
                )
                for j, old in enumerate(frame.locals)
            ),
        )
        for d, frame in enumerate(state.callers)
    )
    valid = z3.And(state.valid, z3.Not(z3.And(happens, invalidate)), z3.Not(z3.And(happens, havoc)))
    oracle = state.oracle + (invalidate, havoc, mutate) + ((raises,) if raises is not None else ())
    changed = z3.And(happens, invalidate)
    globals_version = z3.If(
        changed, z3.Int(f"globals_version_after_p{position}"), state.globals_version
    )
    builtins_version = z3.If(
        changed, z3.Int(f"builtins_version_after_p{position}"), state.builtins_version
    )
    callers = tuple(
        replace(
            frame,
            globals_version=z3.If(
                changed,
                z3.Int(f"caller{d}_globals_version_after_p{position}"),
                frame.globals_version,
            )
            if frame.globals_version is not None
            else None,
        )
        for d, frame in enumerate(callers)
    )
    epoch = z3.If(z3.And(happens, mutate), z3.Int(f"epoch_p{position}"), state.epoch)
    new = replace(
        state,
        locals=replaced,
        callers=callers,
        valid=valid,
        oracle=oracle,
        escape_count=k + 1,
        globals_version=globals_version,
        builtins_version=builtins_version,
        epoch=epoch,
    )
    new = with_fact(
        new,
        position,
        FactKind.ESCAPE,
        uop,
        happens,
        f"escape {k} at position {position}: may invalidate, may overwrite locals (then "
        "invalidates), may mutate reachable objects (new epoch for every uninterpreted read), "
        f"{'may raise' if can_raise else 'cannot raise'}",
    )
    raise_condition = z3.And(happens, raises) if raises is not None else None
    return raise_condition, new
