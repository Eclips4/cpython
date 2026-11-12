import time
from collections.abc import Sequence
from dataclasses import dataclass, replace
from enum import StrEnum

import z3

from .obligations import Obligation


class ProofStatus(StrEnum):
    PROVED = "proved"
    REFUTED = "refuted"
    UNKNOWN = "unknown"
    SYNTACTIC = "proved syntactically"
    UNCHECKED = "unchecked (budget exhausted)"


@dataclass(frozen=True)
class Result:
    obligation: Obligation
    status: ProofStatus
    seconds: float
    model: z3.ModelRef | None


def mentions_float(term: z3.ExprRef) -> bool:
    seen: set[int] = set()
    stack = [term]
    while stack:
        t = stack.pop()
        if t.get_id() in seen:
            continue
        seen.add(t.get_id())
        if z3.is_fp(t) or z3.is_fprm(t):
            return True
        if z3.is_app(t):
            stack.extend(t.children())
    return False


def slim(result: Result) -> Result:
    lightweight = replace(result.obligation, formula=z3.BoolVal(True), states=None)
    return Result(lightweight, result.status, result.seconds, None)


def trivially_true(formula: z3.BoolRef) -> bool:
    simplified = z3.simplify(formula)
    return bool(z3.is_true(simplified))


def check(obligation: Obligation, assumptions: Sequence[z3.BoolRef], timeout_ms: int) -> Result:
    if trivially_true(obligation.formula):
        return Result(obligation, ProofStatus.SYNTACTIC, 0.0, None)
    solver = z3.Solver()
    solver.set("timeout", timeout_ms)
    solver.add(*assumptions)
    solver.add(z3.Not(obligation.formula))
    started = time.perf_counter()
    outcome = solver.check()
    elapsed = time.perf_counter() - started
    if outcome == z3.unsat:
        return Result(obligation, ProofStatus.PROVED, elapsed, None)
    if outcome == z3.sat:
        return Result(obligation, ProofStatus.REFUTED, elapsed, solver.model())
    return Result(obligation, ProofStatus.UNKNOWN, elapsed, None)
