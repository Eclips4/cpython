from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

import z3


class Reachability(StrEnum):
    FEASIBLE = "feasible"
    UNREACHABLE = "unreachable"
    VACUOUS = "vacuous"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Feasibility:
    status: Reachability
    model: z3.ModelRef | None


def feasibility(
    assumptions: Sequence[z3.BoolRef], condition: z3.BoolRef | None, timeout_ms: int = 10000
) -> Feasibility:
    solver = z3.Solver()
    solver.set("timeout", timeout_ms)
    solver.add(*assumptions)
    base = solver.check()
    if base == z3.unsat:
        return Feasibility(Reachability.VACUOUS, None)
    if base == z3.unknown:
        return Feasibility(Reachability.UNKNOWN, None)
    if condition is None:
        return Feasibility(Reachability.FEASIBLE, solver.model())
    solver.add(condition)
    result = solver.check()
    if result == z3.sat:
        return Feasibility(Reachability.FEASIBLE, solver.model())
    if result == z3.unsat:
        return Feasibility(Reachability.UNREACHABLE, None)
    return Feasibility(Reachability.UNKNOWN, None)
