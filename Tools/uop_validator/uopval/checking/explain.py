import z3

from ..symbolic.identities import FALSE_ID, TRUE_ID
from ..symbolic.objects import is_small_id
from ..symbolic.sorts import Slot
from .equality import arrays_equal, balance_at, balance_differences, balances_equal
from .exits import Analysis
from .proof import Result
from .refine import Verdict


def explain(result: Result) -> list[str]:
    lines: list[str] = []
    model = result.model
    if model is None or result.obligation.states is None:
        return lines
    a, b = result.obligation.states

    def value(expr: z3.ExprRef) -> str:
        return str(model.eval(expr, model_completion=True)).replace("\n", " ")

    def differs(x: z3.ExprRef, y: z3.ExprRef) -> bool:
        return not z3.is_true(model.eval(x == y, model_completion=True))

    for j, (x, y) in enumerate(zip(a.locals, b.locals, strict=False)):
        if differs(x, y):
            lines.append(f"local[{j}]: input {value(x)} | output {value(y)}")
    for k, (x, y) in enumerate(zip(a.stack, b.stack, strict=False)):
        if differs(x, y):
            lines.append(f"stack[{k}]: input {value(x)} | output {value(y)}")
    if len(a.stack) != len(b.stack):
        lines.append(f"stack depth: input {len(a.stack)} | output {len(b.stack)}")
    slot_ids = [Slot.id(s) for s in list(a.locals) + list(a.stack) + list(b.locals) + list(b.stack)]
    if not z3.is_true(model.eval(arrays_equal(a.heap, b.heap, slot_ids), model_completion=True)):
        lines.append("heap differs")
    if not z3.is_true(model.eval(balances_equal(a, b), model_completion=True)):
        lines.append("reference balance differs")
        seen: dict[str, None] = {}
        for _, key, _ in balance_differences(a, b):
            ident = model.eval(key, model_completion=True)
            if str(ident) in seen:
                continue
            seen[str(ident)] = None
            ba = value(balance_at(a, ident))
            bb = value(balance_at(b, ident))
            if ba != bb:
                lines.append(f"  id {ident}: input balance {ba} | output balance {bb}")
    if differs(a.valid, b.valid):
        lines.append(f"valid: input {value(a.valid)} | output {value(b.valid)}")
    if differs(a.ip, b.ip):
        lines.append(f"ip: input {value(a.ip)} | output {value(b.ip)}")
    if len(a.callers) != len(b.callers):
        lines.append(f"frame depth: input {len(a.callers)} | output {len(b.callers)}")
    return lines


def witness(analysis: Analysis, model: z3.ModelRef) -> list[str]:
    lines: list[str] = []
    entry = analysis.entry

    def show(slot: z3.DatatypeRef) -> str:
        value = model.eval(slot, model_completion=True)
        if z3.is_true(model.eval(Slot.is_Ref(slot), model_completion=True)):
            ident = model.eval(Slot.id(slot), model_completion=True)
            if z3.is_true(model.eval(is_small_id(entry.consts, ident), model_completion=True)):
                return f"{value} -> small int {ident}"
            if ident.as_long() in (TRUE_ID, FALSE_ID):
                return f"{value} -> {'True' if ident.as_long() == TRUE_ID else 'False'}"
            obj = model.eval(z3.Select(entry.heap, ident), model_completion=True)
            return f"{value} -> {obj}"
        return str(value)

    for k, slot in enumerate(entry.stack):
        lines.append(f"stack[{k}] = {show(slot)}")
    for j, slot in enumerate(entry.locals):
        lines.append(f"local[{j}] = {show(slot)}")
    for term in analysis.final.oracle:
        lines.append(f"{term} = {model.eval(term, model_completion=True)}")
    return lines


def counterexample_smt2(verdict: Verdict, result: Result) -> str:
    solver = z3.Solver()
    solver.add(*verdict.assumptions)
    solver.add(z3.Not(result.obligation.formula))
    text: str = solver.to_smt2()
    return text
