from ..checking.explain import explain
from ..checking.proof import ProofStatus
from ..checking.refine import Verdict
from ..domain.trace import Unit


def unit_columns(label: str, r: Unit) -> str:
    return f"{label:36s} seq={r.provenance.seq!s:5s} {r.code.qualname:50s} in={len(r.input):3d}"


def verdict_line(v: Verdict) -> str:
    proved = sum(1 for x in v.results if x.status in (ProofStatus.PROVED, ProofStatus.SYNTACTIC))
    slowest = max((x.seconds for x in v.results), default=0.0)
    return (
        f"{unit_columns(v.outcome, v.unit)} obligations={len(v.results):3d} proved={proved:3d} "
        f"time={v.seconds:6.2f}s slowest={slowest:5.2f}s  {v.detail}"
    )


def print_counterexample(v: Verdict) -> None:
    refuted = (
        x
        for x in v.results
        if x.status == ProofStatus.REFUTED and not x.obligation.conservative_if_refuted
    )
    first = next(refuted, None)
    if first is None:
        return
    print(
        f"    counterexample to {first.obligation.rule} at {first.obligation.position}: "
        f"{first.obligation.description}"
    )
    for line in explain(first)[:8]:
        print(f"        {line}")
