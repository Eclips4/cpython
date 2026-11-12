import time
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

import z3

from ..alignment.classify import Admitted, Unsupported, classify
from ..alignment.families import F2, Family, is_open
from ..domain.build import IntConstants
from ..domain.trace import Unit, Uop
from ..semantics.execution import execute
from ..semantics.frame_annotation import annotate_frames, carry_annotations
from ..semantics.step import Step, UnsupportedUop
from ..symbolic.state import State
from .entry import entry_constraints, entry_for
from .equality import KEY_CACHE
from .obligations import build_obligations
from .proof import ProofStatus, Result, check, mentions_float, slim
from .solver import Reachability, feasibility


class Outcome(StrEnum):
    VALIDATED = "validated"
    CONSERVATIVE = "validated with conservative exits"
    COUNTEREXAMPLE = "model counterexample"
    INCONCLUSIVE = "inconclusive"
    VACUOUS = "vacuous"
    UNSUPPORTED = "unsupported"
    TOOL_ERROR = "tool error"


class ExecutionFailed(Exception):
    def __init__(self, outcome: Outcome, detail: str) -> None:
        super().__init__(detail)
        self.outcome = outcome
        self.detail = detail


@dataclass(frozen=True)
class Executions:
    entry: State
    steps_in: list[Step]
    steps_out: list[Step]


@dataclass(frozen=True)
class Budget:
    started: float
    unit_s: float | None
    until: int | None

    def exhausted(self) -> bool:
        return self.unit_s is not None and time.perf_counter() - self.started > self.unit_s

    def elapsed(self) -> float:
        return time.perf_counter() - self.started


@dataclass(frozen=True)
class Verdict:
    unit: Unit
    outcome: Outcome
    detail: str
    results: tuple[Result, ...]
    seconds: float
    entry: State | None
    assumptions: tuple[z3.BoolRef, ...]


def aligned_steps(
    pairs: Sequence[tuple[Uop, tuple[Uop, ...]]], steps: Sequence[Step], entry: State
) -> list[Step]:
    out: list[Step] = []
    k = 0
    last = Step(pairs[0][0], entry)
    for _, segment in pairs:
        if segment and k + len(segment) <= len(steps):
            k += len(segment)
            last = steps[k - 1]
        out.append(last)
    return out


def rewritten_positions(unit: Admitted) -> set[int]:
    def same(a: Uop, b: Uop) -> bool:
        return (a.op, a.oparg, a.operand0, a.operand1, a.target) == (
            b.op,
            b.oparg,
            b.operand0,
            b.operand1,
            b.target,
        )

    return {
        i
        for i, (a, segment) in enumerate(unit.pairs)
        if len(segment) != 1 or not same(a, segment[0])
    }


def execute_both(unit: Unit, classified: Admitted, entry: State, generic: bool) -> Executions:
    positions = [a.i for a, segment in classified.pairs for _ in segment]
    output_uops = [u for _, segment in classified.pairs for u in segment]
    annotated_input = annotate_frames(unit.input)
    annotated_output = carry_annotations(output_uops, annotated_input, positions)
    try:
        steps_in = execute(annotated_input, entry, allow_opaque=generic)
        steps_out = aligned_steps(
            classified.pairs,
            execute(annotated_output, entry, positions, allow_opaque=generic),
            entry,
        )
    except UnsupportedUop as e:
        raise ExecutionFailed(Outcome.UNSUPPORTED, str(e)) from e
    except (ValueError, IndexError) as e:
        raise ExecutionFailed(Outcome.TOOL_ERROR, str(e)) from e
    if len(steps_in) != len(steps_out):
        raise ExecutionFailed(
            Outcome.TOOL_ERROR,
            f"executions have different lengths: {len(steps_in)} vs {len(steps_out)}",
        )
    return Executions(entry, steps_in, steps_out)


def unit_assumptions(
    unit: Unit, executions: Executions, generic: bool, extra: Sequence[z3.BoolRef]
) -> tuple[z3.BoolRef, ...]:
    return (
        tuple(entry_constraints(unit, executions.entry, generic))
        + executions.steps_in[-1].state.constraints
        + executions.steps_out[-1].state.constraints
        + tuple(extra)
    )


def check_domain(assumptions: Sequence[z3.BoolRef], timeout_ms: int) -> tuple[Reachability, str]:
    domain_timeout_ms = max(timeout_ms * 6, 30000)
    domain = feasibility(assumptions, None, domain_timeout_ms)
    if domain.status != Reachability.UNKNOWN:
        return domain.status, ""
    without_floats = [a for a in assumptions if not mentions_float(a)]
    domain = feasibility(without_floats, None, domain_timeout_ms)
    return domain.status, " (entry domain checked without float constraints)"


def check_obligations(
    classified: Admitted,
    executions: Executions,
    assumptions: tuple[z3.BoolRef, ...],
    timeout_ms: int,
    budget: Budget,
    changed: set[int] | None,
) -> list[Result]:
    KEY_CACHE.clear()
    checked: list[Result] = []
    lemmas: deque[z3.BoolRef] = deque(maxlen=8)
    passes = (True, False) if changed is not None else (None,)
    for wanted in passes:
        obligations = build_obligations(
            classified, executions.entry, executions.steps_in, executions.steps_out
        )
        for o in obligations:
            if wanted is not None and changed is not None and (o.position in changed) != wanted:
                continue
            if budget.until is not None and o.position > budget.until:
                continue
            if budget.exhausted():
                checked.append(slim(Result(o, ProofStatus.UNCHECKED, 0.0, None)))
                continue
            result = check(o, assumptions + tuple(lemmas), timeout_ms)
            if result.status == ProofStatus.PROVED:
                lemmas.append(o.formula)
            checked.append(result if result.status == ProofStatus.REFUTED else slim(result))
    KEY_CACHE.clear()
    return checked


def judge(results: tuple[Result, ...], changed: set[int] | None, note: str) -> tuple[Outcome, str]:
    refuted = [
        r
        for r in results
        if r.status == ProofStatus.REFUTED and not r.obligation.conservative_if_refuted
    ]
    conservative = [
        r
        for r in results
        if r.status == ProofStatus.REFUTED and r.obligation.conservative_if_refuted
    ]
    unknown = [r for r in results if r.status in (ProofStatus.UNKNOWN, ProofStatus.UNCHECKED)]
    if refuted:
        r = refuted[0]
        return (
            Outcome.COUNTEREXAMPLE,
            f"{r.obligation.rule} at position {r.obligation.position}: "
            f"{r.obligation.description}{note}",
        )
    if unknown:
        r = unknown[0]
        unchecked = sum(1 for x in results if x.status == ProofStatus.UNCHECKED)
        detail = f" ({unchecked} obligations unchecked after the unit budget)" if unchecked else ""
        if changed is not None and all(x.obligation.position not in changed for x in unknown):
            detail += " (every obligation at a rewritten position proved)"
        return (
            Outcome.INCONCLUSIVE,
            f"{r.obligation.rule} at position {r.obligation.position}: "
            f"{r.obligation.description}{detail}{note}",
        )
    if conservative:
        positions = sorted({r.obligation.position for r in conservative})
        return Outcome.CONSERVATIVE, f"extra exits at positions {positions}{note}"
    return Outcome.VALIDATED, f"{len(results)} obligations proved{note}"


def refine(
    unit: Unit,
    consts: IntConstants,
    timeout_ms: int = 10000,
    extra: Sequence[z3.BoolRef] = (),
    family: Family = F2,
    unit_budget_s: float | None = None,
    until: int | None = None,
    rewrites_first: bool = False,
) -> Verdict:
    budget = Budget(time.perf_counter(), unit_budget_s, until)
    classified = classify(unit, family)
    if isinstance(classified, Unsupported):
        detail = "; ".join(f"{r.kind}@{r.position}: {r.detail}" for r in classified.reasons)
        return Verdict(unit, Outcome.UNSUPPORTED, detail, (), budget.elapsed(), None, ())
    assert unit.output is not None
    generic = is_open(family)
    entry = entry_for(unit, consts)
    try:
        executions = execute_both(unit, classified, entry, generic)
    except ExecutionFailed as e:
        return Verdict(unit, e.outcome, e.detail, (), budget.elapsed(), entry, ())
    assumptions = unit_assumptions(unit, executions, generic, extra)
    domain, note = check_domain(assumptions, timeout_ms)
    if domain == Reachability.VACUOUS:
        detail = "the entry domain is unsatisfiable" + note
        return Verdict(unit, Outcome.VACUOUS, detail, (), budget.elapsed(), entry, assumptions)
    if domain == Reachability.UNKNOWN:
        detail = "the entry domain's satisfiability is unknown"
        return Verdict(unit, Outcome.INCONCLUSIVE, detail, (), budget.elapsed(), entry, assumptions)
    changed = rewritten_positions(classified) if rewrites_first else None
    results = tuple(
        check_obligations(classified, executions, assumptions, timeout_ms, budget, changed)
    )
    outcome, detail = judge(results, changed, note)
    return Verdict(unit, outcome, detail, results, budget.elapsed(), entry, assumptions)
