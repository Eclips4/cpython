from dataclasses import dataclass

from ..domain.trace import TERMINATORS, Unit, Uop
from ..semantics.execution import modelled
from ..semantics.opaque import opaque_eligible
from .families import F2, Family, is_open

CANCELLABLE_POPS: tuple[str, ...] = ("_POP_TOP", "_POP_TOP_INT")


@dataclass(frozen=True)
class Reason:
    kind: str
    position: int | None
    detail: str


Pair = tuple[Uop, tuple[Uop, ...]]


@dataclass(frozen=True)
class Admitted:
    unit: Unit
    family: Family
    pairs: tuple[Pair, ...]


@dataclass(frozen=True)
class Unsupported:
    unit: Unit
    reasons: tuple[Reason, ...]


def choose_segment(
    a: Uop, output: tuple[Uop, ...], j: int, family: Family
) -> tuple[Uop, ...] | None:
    if output[j].op == a.op:
        return (output[j],)
    for names in family.segments.get(a.op, ()):
        candidate = output[j : j + len(names)]
        if tuple(u.op for u in candidate) == names:
            return candidate
    return None


def check_segment(
    a: Uop, segment: tuple[Uop, ...], reasons: list[Reason], cancelled_loads: list[int]
) -> None:
    i = a.i
    b = segment[0]
    if len(segment) == 1 and b.op == a.op:
        if b.oparg != a.oparg and not (a.op == "_POP_TOP" and b.oparg == 0):
            reasons.append(Reason("oparg_changed", i, f"{a.op}: {a.oparg} -> {b.oparg}"))
    elif len(segment) == 1:
        if a.op == "_LOAD_SMALL_INT":
            c = b.constant
            if c is None or c.type != "int" or not c.immortal or c.int_value != a.oparg:
                reasons.append(Reason("constant_mismatch", i, f"_LOAD_SMALL_INT {a.oparg} -> {c}"))
        if a.op == "_LOAD_FAST_BORROW" and b.op == "_NOP":
            cancelled_loads.append(i)
        if a.op in CANCELLABLE_POPS and b.op == "_NOP":
            if cancelled_loads:
                cancelled_loads.pop()
            else:
                reasons.append(Reason("pop_cancelled_without_push", i, a.op))
    for u in segment:
        if u.op != "_NOP" and u.target != a.target:
            reasons.append(Reason("target_changed", i, f"{u.op}: {a.target} -> {u.target}"))


def entry_shape_reasons(unit: Unit, family: Family = F2) -> list[Reason]:
    reasons: list[Reason] = []
    depth = unit.entry.stack_depth
    has_range_header = any(u.op == "_ITER_CHECK_RANGE" for u in unit.input)
    header = "present" if has_range_header else "absent"
    if depth == 0:
        if not family.depth_zero or has_range_header:
            reasons.append(
                Reason("entry_shape", None, f"stack_depth=0, range loop header {header}")
            )
    elif depth == 2:
        if not has_range_header:
            reasons.append(
                Reason("entry_shape", None, f"stack_depth=2, range loop header {header}")
            )
    elif depth > 2 and family.lower_stack_free:
        if not has_range_header:
            reasons.append(
                Reason("entry_shape", None, f"stack_depth={depth}, range loop header {header}")
            )
    else:
        reasons.append(
            Reason("entry_shape", None, f"stack_depth={depth}, range loop header {header}")
        )
    if unit.entry.from_side_exit or unit.entry.chain_depth != 0:
        reasons.append(
            Reason(
                "entry_shape",
                None,
                f"side trace: from_side_exit={unit.entry.from_side_exit}, "
                f"chain_depth={unit.entry.chain_depth}",
            )
        )
    return reasons


def outside_reasons(unit: Unit, family: Family = F2) -> list[Reason]:
    outside: dict[str, int] = {}
    for u in unit.input + (unit.output or ()):
        if u.op not in family.uops:
            outside.setdefault(u.op, u.i)
    if not outside:
        return []
    return [Reason("uop_outside_family", min(outside.values()), ", ".join(sorted(outside)))]


def align(unit: Unit, output: tuple[Uop, ...], family: Family) -> tuple[list[Pair], list[Reason]]:
    reasons: list[Reason] = []
    pairs: list[Pair] = []
    cancelled_loads: list[int] = []
    j = 0
    for a in unit.input:
        if j >= len(output):
            reasons.append(
                Reason(
                    "length_mismatch", a.i, f"output ends after {len(output)} uops, input continues"
                )
            )
            break
        segment = choose_segment(a, output, j, family)
        if segment is None:
            reasons.append(Reason("rewrite_not_admitted", a.i, f"{a.op} -> {output[j].op}"))
            segment = (output[j],)
        check_segment(a, segment, reasons, cancelled_loads)
        pairs.append((a, segment))
        j += len(segment)
    if j < len(output):
        reasons.append(
            Reason(
                "length_mismatch",
                None,
                f"{len(output) - j} output uops remain after the input's {len(unit.input)}",
            )
        )
    if cancelled_loads:
        reasons.append(
            Reason("push_cancelled_without_pop", cancelled_loads[0], "_LOAD_FAST_BORROW")
        )
    if pairs and unit.input[-1].op != output[-1].op:
        reasons.append(
            Reason(
                "terminator_mismatch",
                len(unit.input) - 1,
                f"{unit.input[-1].op} -> {output[-1].op}",
            )
        )
    return pairs, reasons


def same_operands(a: Uop, b: Uop) -> bool:
    return a.oparg == b.oparg and a.operand0 == b.operand0 and a.operand1 == b.operand1


def align_by_origin(unit: Unit, output: tuple[Uop, ...]) -> tuple[list[Pair], list[Reason]]:
    reasons: list[Reason] = []
    groups: dict[int, list[Uop]] = {}
    for u in output:
        assert u.origin is not None
        groups.setdefault(u.origin, []).append(u)
    terminator = output[-1]
    assert terminator.origin is not None
    k = terminator.origin
    last_input = len(unit.input) - 1
    if k < last_input:
        exiting_form = (
            terminator.op == "_EXIT_TRACE"
            and k >= 1
            and unit.input[k].op not in TERMINATORS
            and any(r.i == k and r.op_after == "_EXIT_TRACE" for r in unit.rewritten_in_place)
            and groups.get(k) == [terminator]
        )
        if exiting_form:
            groups[k - 1] = groups.get(k - 1, []) + [terminator]
            del groups[k]
        else:
            reasons.append(
                Reason(
                    "unexpected_truncation",
                    k,
                    f"output ends at input position {k} of {last_input} without the exiting form "
                    "of guard elimination",
                )
            )
    pairs: list[Pair] = []
    for a in unit.input:
        segment = tuple(groups.get(a.i, ()))
        pairs.append((a, segment))
        if not segment:
            if a.i <= k:
                reasons.append(Reason("missing_output", a.i, f"{a.op} produced no output uop"))
            continue
        if len(segment) == 1 and segment[0].op == a.op:
            b = segment[0]
            if modelled(a.op):
                continue
            if not opaque_eligible(a.op):
                reasons.append(Reason("unmodelled_uop", a.i, a.op))
            elif not same_operands(a, b):
                reasons.append(
                    Reason(
                        "rewrite_of_unmodelled_uop",
                        a.i,
                        f"{a.op} -> {b.op} with different operands",
                    )
                )
            continue
        if not modelled(a.op):
            reasons.append(
                Reason(
                    "rewrite_of_unmodelled_uop",
                    a.i,
                    f"{a.op} -> {' + '.join(u.op for u in segment)}",
                )
            )
            continue
        unknown = [u.op for u in segment if not modelled(u.op)]
        if unknown:
            reasons.append(
                Reason(
                    "rewrite_to_unmodelled_uop",
                    a.i,
                    f"{a.op} -> {' + '.join(u.op for u in segment)}: "
                    f"{', '.join(sorted(set(unknown)))}",
                )
            )
    return pairs, reasons


def classify(unit: Unit, family: Family = F2) -> Admitted | Unsupported:
    reasons: list[Reason] = []
    if unit.output is None:
        reasons.append(
            Reason("no_output", None, f"result_length={unit.result_length}, noopt={unit.noopt}")
        )
    pairs: list[Pair] = []
    if is_open(family):
        if unit.output is not None:
            if any(u.origin is None for u in unit.output):
                reasons.append(Reason("no_origins", None, "the output uops carry no origin index"))
            else:
                pairs, alignment = align_by_origin(unit, unit.output)
                reasons.extend(alignment)
    else:
        reasons.extend(entry_shape_reasons(unit, family))
        reasons.extend(outside_reasons(unit, family))
        if unit.output is not None:
            pairs, alignment = align(unit, unit.output, family)
            reasons.extend(alignment)
    if reasons:
        return Unsupported(unit, tuple(reasons))
    return Admitted(unit, family, tuple(pairs))
