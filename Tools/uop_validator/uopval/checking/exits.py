from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

import z3

from ..domain.build import IntConstants
from ..domain.trace import Unit
from ..semantics.execution import execute
from ..symbolic.state import Exit, FactKind, State, entry_state
from .entry import entry_constraints
from .solver import Feasibility, feasibility


@dataclass(frozen=True)
class Analysis:
    unit: Unit
    entry: State
    final: State
    assumptions: tuple[z3.BoolRef, ...]
    exits: tuple[tuple[Exit, Feasibility], ...]
    loop_back: Feasibility


def analyse(
    unit: Unit,
    consts: IntConstants,
    extra: Sequence[z3.BoolRef] = (),
    timeout_ms: int = 10000,
    allow_opaque: bool = False,
) -> Analysis:
    entry = entry_state(
        consts,
        unit.code.nlocalsplus,
        unit.entry.stack_depth,
        unit.input[0].operand0 // 2,
        unit.code.co_version,
    )
    final = execute(unit.input, entry, allow_opaque=allow_opaque)[-1].state
    assumptions = tuple(entry_constraints(unit, entry)) + final.constraints + tuple(extra)
    checked = tuple(
        (ex, feasibility(list(assumptions) + list(ex.path), ex.condition, timeout_ms))
        for ex in final.exits
    )
    loop_back = feasibility(list(assumptions) + list(final.path), None, timeout_ms)
    return Analysis(unit, entry, final, assumptions, checked, loop_back)


class Implication(StrEnum):
    IMPLIED = "implied"
    NOT_IMPLIED = "not implied"
    UNKNOWN = "unknown"


def preconditions_implied(
    analysis: Analysis, timeout_ms: int = 10000
) -> list[tuple[int, str, str, Implication]]:
    results: list[tuple[int, str, str, Implication]] = []
    for fact in analysis.final.facts:
        if fact.kind != FactKind.PRECONDITION or fact.expr is None:
            continue
        path = [z3.Not(ex.condition) for ex in analysis.final.exits if ex.position < fact.position]
        solver = z3.Solver()
        solver.set("timeout", timeout_ms)
        solver.add(*analysis.assumptions, *path, z3.Not(fact.expr))
        outcome = solver.check()
        if outcome == z3.unsat:
            status = Implication.IMPLIED
        elif outcome == z3.sat:
            status = Implication.NOT_IMPLIED
        else:
            status = Implication.UNKNOWN
        results.append((fact.position, fact.uop, fact.note, status))
    return results
