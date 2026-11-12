from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field

from .classify import Admitted, Unsupported


@dataclass
class Summary:
    no_output: int = 0
    admitted: int = 0
    unsupported_first_reason: Counter[str] = field(default_factory=Counter)
    unsupported_all_reasons: Counter[str] = field(default_factory=Counter)
    uops_outside_family: Counter[str] = field(default_factory=Counter)


def summarize(classified: Iterable[Admitted | Unsupported]) -> Summary:
    s = Summary()
    for c in classified:
        if isinstance(c, Admitted):
            s.admitted += 1
        else:
            if c.unit.output is None:
                s.no_output += 1
            s.unsupported_first_reason[c.reasons[0].kind] += 1
            seen: set[str] = set()
            for r in c.reasons:
                s.unsupported_all_reasons[r.kind] += 1
                if r.kind == "uop_outside_family":
                    for name in r.detail.split(", "):
                        s.uops_outside_family[name] += 1
                if r.kind in (
                    "rewrite_of_unmodelled_uop",
                    "unmodelled_uop",
                    "rewrite_to_unmodelled_uop",
                ):
                    name = r.detail.split(" ")[0]
                    if name not in seen:
                        seen.add(name)
                        s.uops_outside_family[name] += 1
    return s
