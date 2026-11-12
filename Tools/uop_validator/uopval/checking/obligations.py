from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import z3

from ..alignment.classify import Admitted
from ..domain.trace import Uop
from ..semantics.step import Step
from ..symbolic.state import GUARD_EXITS, Exit, FactKind, State
from .equality import observable_equal, states_equal


@dataclass(frozen=True)
class Obligation:
    position: int
    rule: str
    description: str
    formula: z3.BoolRef
    conservative_if_refuted: bool = False
    states: tuple[State, State] | None = None


@dataclass(frozen=True)
class Traces:
    entry: State
    steps_in: Sequence[Step]
    steps_out: Sequence[Step]

    @property
    def final_in(self) -> State:
        return self.steps_in[-1].state

    @property
    def final_out(self) -> State:
        return self.steps_out[-1].state

    def before_in(self, position: int) -> State:
        return state_before(self.steps_in, self.entry, position)

    def before_out(self, position: int) -> State:
        return state_before(self.steps_out, self.entry, position)


@dataclass(frozen=True)
class Position:
    index: int
    uop: Uop
    output_ops: str
    phi: z3.BoolRef
    ins: list[Exit]
    outs: list[Exit]


def state_before(steps: Sequence[Step], entry: State, position: int) -> State:
    return entry if position == 0 else steps[position - 1].state


def fires(exits: Sequence[Exit], k: int) -> z3.BoolRef:
    return z3.And(exits[k].condition, *[z3.Not(e.condition) for e in exits[:k]])


def exits_by_position(state: State) -> dict[int, list[Exit]]:
    out: dict[int, list[Exit]] = {}
    for ex in state.exits:
        out.setdefault(ex.position, []).append(ex)
    return out


def facts_at(state: State, kind: FactKind, position: int) -> list[z3.BoolRef]:
    return [
        f.expr
        for f in state.facts
        if f.kind == kind and f.position == position and f.expr is not None
    ]


def handover_reference(traces: Traces, position: int, out_exit: Exit) -> State:
    before = traces.before_in(position)
    after = traces.steps_in[position].state if position < len(traces.steps_in) else before
    depth = len(out_exit.state.stack)
    if depth != len(before.stack) and depth == len(after.stack):
        return after
    return before


def path_terms(
    exits_in: dict[int, list[Exit]],
    exits_out: dict[int, list[Exit]],
    final_in: State,
    position: int,
) -> list[z3.BoolRef]:
    terms = [z3.Not(ex.condition) for ex in exits_in.get(position, [])]
    terms += [z3.Not(ex.condition) for ex in exits_out.get(position, [])]
    terms += facts_at(final_in, FactKind.PRECONDITION, position)
    return terms


def handed_over_equal(
    p: Position, e_in: Exit, fires_in: z3.BoolRef, k: int, note: str
) -> Obligation:
    e_out = p.outs[k]
    return Obligation(
        p.index,
        "R2",
        f"{p.uop.op}: handed-over states equal when both exit ({e_out.uop}, {e_in.kind}{note})",
        z3.Implies(
            z3.And(p.phi, fires_in, fires(p.outs, k)),
            states_equal(e_in.state, e_out.state, with_ip=False),
        ),
        states=(e_in.state, e_out.state),
    )


def output_preconditions(p: Position, traces: Traces) -> Iterator[Obligation]:
    for fact in traces.final_out.facts:
        if fact.kind != FactKind.PRECONDITION or fact.position != p.index or fact.expr is None:
            continue
        guards_before = [
            z3.Not(e.condition) for e in p.outs if e.kind in GUARD_EXITS and e.uop != fact.uop
        ]
        yield Obligation(
            p.index,
            "R1",
            f"output {fact.uop} precondition: {fact.note}",
            z3.Implies(z3.And(p.phi, *guards_before), fact.expr),
        )


def input_exit(p: Position, traces: Traces, j: int, e_in: Exit) -> Iterator[Obligation]:
    fires_in = fires(p.ins, j)
    matching = [k for k, e in enumerate(p.outs) if e.kind == e_in.kind and e.target == e_in.target]
    conservative = [k for k, e in enumerate(p.outs) if k not in matching and e.kind in GUARD_EXITS]
    if e_in.environmental:
        for k in matching:
            yield handed_over_equal(p, e_in, fires_in, k, ", allocation")
        return
    if not p.outs:
        yield Obligation(
            p.index,
            "R2",
            f"{p.uop.op} removed ({p.output_ops}): its {e_in.kind} exit must be unreachable",
            z3.Implies(p.phi, z3.Not(fires_in)),
        )
        return
    if not matching and not conservative:
        kinds = ", ".join(f"{e.kind}->{e.target}" for e in p.outs)
        yield Obligation(
            p.index,
            "R2",
            f"{p.uop.op}: exit {e_in.kind}->{e_in.target} has no output exit of the same "
            f"kind and target ({kinds}); it must be unreachable",
            z3.Implies(p.phi, z3.Not(fires_in)),
        )
        return
    covered_out: list[z3.BoolRef] = [fires(p.outs, k) for k in matching]
    covered_out += [
        z3.And(
            fires(p.outs, k),
            states_equal(
                p.outs[k].state, handover_reference(traces, p.index, p.outs[k]), with_ip=False
            ),
        )
        for k in conservative
    ]
    yield Obligation(
        p.index,
        "R2",
        f"{p.uop.op}: input {e_in.kind} exit implies a matching or conservative output exit",
        z3.Implies(z3.And(p.phi, fires_in), z3.Or(*covered_out)),
    )
    for k in matching:
        yield handed_over_equal(p, e_in, fires_in, k, "")


def input_exits(p: Position, traces: Traces) -> Iterator[Obligation]:
    for j, e_in in enumerate(p.ins):
        yield from input_exit(p, traces, j, e_in)


def output_exits(p: Position, traces: Traces) -> Iterator[Obligation]:
    for k, e_out in enumerate(p.outs):
        covered_by = [
            j for j, e in enumerate(p.ins) if e.kind == e_out.kind and e.target == e_out.target
        ]
        covered = z3.Or(*[fires(p.ins, j) for j in covered_by]) if covered_by else z3.BoolVal(False)
        label = (
            f"{e_out.uop} added" if not p.ins else f"{e_out.uop}: output exit implies input exit"
        )
        yield Obligation(
            p.index,
            "R3",
            label,
            z3.Implies(z3.And(p.phi, fires(p.outs, k)), covered),
            conservative_if_refuted=True,
        )
        reference = handover_reference(traces, p.index, e_out)
        yield Obligation(
            p.index,
            "R3",
            f"{e_out.uop}: state at an extra exit equals the input's state here",
            z3.Implies(
                z3.And(p.phi, fires(p.outs, k), z3.Not(covered)),
                states_equal(e_out.state, reference, with_ip=False),
            ),
            states=(reference, e_out.state),
        )


def escapes(p: Position, traces: Traces) -> Iterator[Obligation]:
    escapes_in = facts_at(traces.final_in, FactKind.ESCAPE, p.index)
    escapes_out = facts_at(traces.final_out, FactKind.ESCAPE, p.index)
    if not escapes_in and not escapes_out:
        return
    guards_here = [z3.Not(e.condition) for e in p.ins + p.outs if e.kind in GUARD_EXITS]
    phi_escape = z3.And(p.phi, *guards_here)
    happens = z3.Or(*(escapes_in + escapes_out))
    before_in, before_out = traces.before_in(p.index), traces.before_out(p.index)
    yield Obligation(
        p.index,
        "R6",
        f"{p.uop.op}: observable states including ip equal before the escape",
        z3.Implies(z3.And(phi_escape, happens), observable_equal(before_in, before_out)),
        states=(before_in, before_out),
    )
    if escapes_in and not escapes_out:
        yield Obligation(
            p.index,
            "R6",
            f"{p.uop.op}: input escapes but output does not: the escape must not happen",
            z3.Implies(phi_escape, z3.Not(z3.Or(*escapes_in))),
        )
    if escapes_out and not escapes_in:
        yield Obligation(
            p.index,
            "R6",
            f"{p.output_ops}: output escapes but input does not: the escape must not happen",
            z3.Implies(phi_escape, z3.Not(z3.Or(*escapes_out))),
        )


def back_edge(unit: Admitted, traces: Traces, accumulated: z3.BoolRef) -> Iterator[Obligation]:
    last = len(unit.pairs)
    phi_end = z3.And(accumulated, *facts_at(traces.final_in, FactKind.PRECONDITION, last))
    if unit.pairs[-1][0].op == "_JUMP_TO_TOP":
        yield Obligation(
            last - 1,
            "R4",
            "states equal at the loop back-edge",
            z3.Implies(phi_end, states_equal(traces.final_in, traces.final_out, with_ip=False)),
            states=(traces.final_in, traces.final_out),
        )


def build_obligations(
    unit: Admitted, entry: State, steps_in: Sequence[Step], steps_out: Sequence[Step]
) -> Iterator[Obligation]:
    traces = Traces(entry, steps_in, steps_out)
    exits_in = exits_by_position(traces.final_in)
    exits_out = exits_by_position(traces.final_out)
    accumulated: z3.BoolRef = z3.BoolVal(True)
    for i, (a, segment) in enumerate(unit.pairs):
        output_ops = " + ".join(u.op for u in segment) if segment else "(nothing)"
        phi = z3.And(accumulated, *facts_at(traces.final_in, FactKind.PRECONDITION, i))
        accumulated = z3.And(accumulated, *path_terms(exits_in, exits_out, traces.final_in, i))
        p = Position(i, a, output_ops, phi, exits_in.get(i, []), exits_out.get(i, []))
        yield from output_preconditions(p, traces)
        yield from input_exits(p, traces)
        yield from output_exits(p, traces)
        yield from escapes(p, traces)
    yield from back_edge(unit, traces, accumulated)
