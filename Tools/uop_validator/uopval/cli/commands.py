import itertools
import textwrap
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import z3

from ..adapters import build_constants, capture
from ..adapters.capture import Rejected
from ..alignment.classify import Admitted, classify
from ..alignment.families import Family, is_open
from ..alignment.summary import summarize
from ..checking.entry import entry_for
from ..checking.exits import analyse, preconditions_implied
from ..checking.explain import witness
from ..checking.refine import refine
from ..checking.solver import Reachability
from ..domain.trace import Unit, Uop
from ..semantics.execution import describe, execute
from ..semantics.frame_annotation import annotate_frames
from .report import print_counterexample, unit_columns, verdict_line

ROOT = Path(__file__).resolve().parents[4]


def cmd_stats(paths: Sequence[str], family: Family) -> int:
    records = list(itertools.chain.from_iterable(capture.load(p) for p in paths))
    units = [r for r in records if isinstance(r, Unit)]
    rejected = Counter(r.reason for r in records if isinstance(r, Rejected))
    s = summarize(classify(u, family) for u in units)
    outside = "".join(f"\n  {n:5d} {name}" for name, n in s.uops_outside_family.most_common(40))
    print(
        textwrap.dedent(f"""\
        family: {family.name}
        attempted records: {len(records)}
        rejected by the parser: {rejected.total()} {dict(rejected)}
        parsed units: {len(units)}, of which without optimizer output: {s.no_output}
        admitted units: {s.admitted}
        unsupported units: {len(units) - s.admitted}
        first reason: {dict(s.unsupported_first_reason.most_common())}
        all reasons: {dict(s.unsupported_all_reasons.most_common())}
        uops outside {family.name} (records containing them), top 40:""")
        + outside
    )
    return 0


def cmd_show(path: str, seq: int, family: Family) -> int:
    for r in capture.load(path):
        if isinstance(r, Rejected):
            if r.provenance.seq == seq:
                print(f"rejected: {r.reason}: {r.detail}")
                return 1
            continue
        if r.provenance.seq != seq:
            continue
        c = classify(r, family)
        print(
            f"{r.code.qualname} ip0={r.entry.ip0} depth={r.entry.stack_depth} input={len(r.input)} "
            f"tail={len(r.input_tail)} output={None if r.output is None else len(r.output)}"
        )
        if isinstance(c, Admitted):
            print(f"classification: {family.name}")
            segments: dict[int, tuple[Uop, ...]] = {a.i: seg for a, seg in c.pairs}
        else:
            print(
                "classification: unsupported: "
                + "; ".join(f"{x.kind}@{x.position}: {x.detail}" for x in c.reasons)
            )
            segments = {
                a.i: (r.output[a.i],)
                for a in r.input
                if r.output is not None and a.i < len(r.output)
            }
        for a in r.input:
            seg = segments.get(a.i, ())
            right = (
                ""
                if not seg
                else "  ->  " + " + ".join(f"{b.op}({b.oparg}, target={b.target})" for b in seg)
            )
            mark = (
                ""
                if not seg or (len(seg) == 1 and (a.op, a.oparg) == (seg[0].op, seg[0].oparg))
                else "  *"
            )
            print(f"  {a.i:3d} {a.op}({a.oparg}, target={a.target}){right}{mark}")
        return 0
    print("no record with that seq")
    return 1


def cmd_trace(path: str, seq: int, family: Family) -> int:
    consts = build_constants.default(ROOT)
    for r in capture.load(path):
        if isinstance(r, Rejected) or r.provenance.seq != seq:
            continue
        print(f"{r.code.qualname}: constants from {consts.source}")
        state = entry_for(r, consts)
        for step in execute(annotate_frames(r.input), state, allow_opaque=is_open(family)):
            print(f"--- {step.uop.i:3d} {step.uop.op}({step.uop.oparg}, target={step.uop.target})")
            for line in describe(step.state):
                print("   " + line)
            for fact in step.state.facts[len(state.facts) :]:
                shown = "" if fact.expr is None else f": {fact.expr}"
                print(f"   [{fact.kind}] {fact.note}{shown}")
            state = step.state
        print(f"constraints on fresh identities: {len(state.constraints)}")
        return 0
    print("no record with that seq")
    return 1


def cmd_exits(path: str, seq: int) -> int:
    consts = build_constants.default(ROOT)
    for r in capture.load(path):
        if isinstance(r, Rejected) or r.provenance.seq != seq:
            continue
        analysis = analyse(r, consts)
        print(
            f"{r.code.qualname}: {len(analysis.exits)} exits in the input trace; entry "
            f"assumptions: {len(analysis.assumptions)}"
        )
        for ex, feas in analysis.exits:
            print(
                f"  {ex.position:3d} {ex.uop:28s} {ex.kind:6s} target={ex.target!s:4s} "
                f"{feas.status}"
            )
            if feas.model is not None and feas.status == Reachability.FEASIBLE:
                for line in witness(analysis, feas.model)[:6]:
                    print(f"        {line}")
        print(f"  loop back at {r.input[-1].i}: {analysis.loop_back.status}")
        for position, uop, note, status in preconditions_implied(analysis):
            print(f"  precondition at {position:3d} {uop:24s} {status}: {note}")
        return 0
    print("no record with that seq")
    return 1


def selected(r: Unit, only: set[int] | None, lines: tuple[int, int] | None) -> bool:
    if only is not None and r.provenance.seq not in only:
        return False
    return lines is None or lines[0] <= r.provenance.line < lines[1]


def cmd_refine(
    paths: Sequence[str],
    timeout_ms: int,
    family: Family,
    only: set[int] | None,
    unit_budget_s: float | None,
    until: int | None,
    lines: tuple[int, int] | None,
    rewrites_first: bool,
) -> int:
    consts = build_constants.default(ROOT)
    counts: Counter[str] = Counter()
    for r in itertools.chain.from_iterable(capture.load(p) for p in paths):
        if isinstance(r, Rejected):
            counts["rejected"] += 1
            print(f"rejected  line {r.provenance.line}: {r.reason}: {r.detail}")
            continue
        if not selected(r, only, lines):
            continue
        try:
            v = refine(
                r,
                consts,
                timeout_ms,
                family=family,
                unit_budget_s=unit_budget_s,
                until=until,
                rewrites_first=rewrites_first,
            )
        except (MemoryError, z3.Z3Exception) as e:
            print(
                f"{unit_columns('tool error', r)} out of memory at line={r.provenance.line}: "
                f"{str(e)[:60]}",
                flush=True,
            )
            return 3
        counts[v.outcome.value] += 1
        print(verdict_line(v))
        print_counterexample(v)
    print("summary:", dict(sorted(counts.items())))
    return 0
