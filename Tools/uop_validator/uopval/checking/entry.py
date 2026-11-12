import z3

from ..domain.build import IntConstants
from ..domain.trace import CodeSummary, Constant, Unit
from ..symbolic.functions import frozen_member, int_as_float, named_type_tag, subtype
from ..symbolic.identities import type_tag
from ..symbolic.objects import is_immortal_id, singleton_facts
from ..symbolic.sorts import Float64, Slot
from ..symbolic.state import State, entry_state
from ..symbolic.summaries import constant_id, type_tag_of_summary


def collect_codes(unit: Unit) -> dict[int, CodeSummary]:
    codes: dict[int, CodeSummary] = {}

    def visit(summary: Constant | None) -> None:
        if summary is None:
            return
        if summary.code is not None:
            codes.setdefault(summary.code.co_version, summary.code)
        visit(summary.im_func)
        visit(summary.im_self)
        if summary.items:
            for item in summary.items:
                visit(item)

    for u in unit.input + (unit.output or ()):
        visit(u.constant)
        visit(u.recorded)
        visit(u.operand0_object)
        visit(u.operand1_object)
        visit(u.global_value)
        if u.constant_code is not None:
            codes.setdefault(u.constant_code.co_version, u.constant_code)
    return codes


def entry_for(unit: Unit, consts: IntConstants) -> State:
    return entry_state(
        consts,
        unit.code.nlocalsplus,
        unit.entry.stack_depth,
        unit.input[0].operand0 // 2,
        unit.code.co_version,
        collect_codes(unit),
    )


def unit_constants(unit: Unit) -> list[Constant]:
    out: list[Constant] = []
    seen: set[int] = set()

    def visit(summary: Constant | None) -> None:
        if summary is None or id(summary) in seen:
            return
        seen.add(id(summary))
        out.append(summary)
        for child in (summary.im_func, summary.im_self, summary.init, summary.getitem):
            visit(child)
        for item in summary.items or ():
            visit(item)
        for key, value in summary.pairs or ():
            visit(key)
            visit(value)

    for u in unit.input + (unit.output or ()):
        for summary in (
            u.constant,
            u.recorded,
            u.operand0_object,
            u.operand1_object,
            u.global_value,
            u.lookup,
            u.lookup_type,
        ):
            visit(summary)
        if u.binop_descr is not None:
            for summary in (
                u.binop_descr.lhs_type,
                u.binop_descr.rhs_type,
                u.binop_descr.result_type,
            ):
                visit(summary)
    return out


BUILTIN_BASES = {
    "bool": ("int", "object"),
    "int": ("object",),
    "float": ("object",),
    "str": ("object",),
    "bytes": ("object",),
    "tuple": ("object",),
    "list": ("object",),
    "dict": ("object",),
    "set": ("object",),
    "frozenset": ("object",),
    "NoneType": ("object",),
    "type": ("object",),
    "range": ("object",),
    "slice": ("object",),
    "function": ("object",),
    "method": ("object",),
    "module": ("object",),
    "builtin_function_or_method": ("object",),
    "method_descriptor": ("object",),
    "generator": ("object",),
    "range_iterator": ("object",),
    "object": (),
}


def builtin_hierarchy_facts() -> list[z3.BoolRef]:
    facts: list[z3.BoolRef] = []
    for name, bases in BUILTIN_BASES.items():
        tag = z3.IntVal(type_tag(name))
        facts.append(subtype(tag, tag))
        for other in BUILTIN_BASES:
            if other != name:
                facts.append(subtype(tag, z3.IntVal(type_tag(other))) == z3.BoolVal(other in bases))
    return facts


def constant_relation_facts(unit: Unit) -> list[z3.BoolRef]:
    facts: list[z3.BoolRef] = builtin_hierarchy_facts()
    seen_ints: set[int] = set()
    types: dict[int, Constant] = {}
    for summary in unit_constants(unit):
        if (
            summary.int_value is not None
            and abs(summary.int_value) < 2**53
            and summary.int_value not in seen_ints
        ):
            seen_ints.add(summary.int_value)
            facts.append(
                int_as_float(z3.IntVal(summary.int_value))
                == z3.FPVal(float(summary.int_value), Float64)
            )
        if summary.type_name is not None and summary.addr is not None:
            types.setdefault(summary.addr, summary)
    by_name: dict[str, set[int]] = {}
    for t in types.values():
        tag = z3.IntVal(type_tag_of_summary(t))
        facts.append(subtype(tag, tag))
        assert t.type_name is not None
        by_name.setdefault(t.type_name, set()).add(type_tag_of_summary(t))
    for name, tags in by_name.items():
        if len(tags) == 1:
            facts.append(named_type_tag(z3.IntVal(type_tag(name))) == z3.IntVal(next(iter(tags))))
        facts.append(subtype(tag, z3.IntVal(type_tag("object"))))
        if t.mro_addrs is None:
            continue
        if not t.immutable:
            for name in BUILTIN_BASES:
                if name != "object":
                    facts.append(
                        subtype(tag, z3.IntVal(type_tag(name)))
                        == z3.BoolVal(
                            any(
                                u.type_name == name and u.immutable and u.addr in t.mro_addrs
                                for u in types.values()
                            )
                        )
                    )
        for u in types.values():
            if u is t:
                continue
            facts.append(
                subtype(tag, z3.IntVal(type_tag_of_summary(u))) == z3.BoolVal(u.addr in t.mro_addrs)
            )
    candidate_keys: dict[int, None] = {}
    containers: list[tuple[int, set[int]]] = []
    for summary in unit_constants(unit):
        try:
            ident = constant_id(summary)
        except ValueError:
            continue
        candidate_keys.setdefault(ident, None)
        members = frozen_member_ids(summary)
        if members is not None:
            containers.append((ident, members))
    for container, members in containers:
        for key in candidate_keys:
            facts.append(
                frozen_member(z3.IntVal(container), z3.IntVal(key)) == z3.BoolVal(key in members)
            )
    return facts


def frozen_member_ids(summary: Constant) -> set[int] | None:
    if (
        summary.type == "frozenset"
        and summary.items is not None
        and summary.length is not None
        and len(summary.items) == summary.length
    ):
        entries: list[Constant] = list(summary.items)
    elif (
        summary.type == "frozendict"
        and summary.pairs is not None
        and summary.length is not None
        and len(summary.pairs) == summary.length
    ):
        entries = [key for key, _ in summary.pairs]
    else:
        return None
    try:
        return {constant_id(k) for k in entries}
    except ValueError:
        return None


def entry_constraints(unit: Unit, state: State, generic: bool = False) -> list[z3.BoolRef]:
    constraints: list[z3.BoolRef] = constant_relation_facts(unit)
    if len(state.stack) >= 2 and not generic:
        constraints.append(Slot.is_Ref(state.stack[-2]))
        constraints.append(state.stack[-1] == Slot.Null)
    for uop in unit.input:
        if uop.op in ("_PUSH_FRAME", "_RETURN_VALUE", "_YIELD_VALUE", "_RETURN_GENERATOR"):
            break
        if uop.op in ("_LOAD_FAST_BORROW", "_LOAD_FAST") and uop.oparg < len(state.locals):
            constraints.append(Slot.is_Ref(state.locals[uop.oparg]))
    for slot in state.locals + state.stack:
        immortal = z3.And(Slot.is_Ref(slot), is_immortal_id(state.consts, Slot.id(slot)))
        constraints.append(z3.Implies(immortal, Slot.borrowed(slot)))
    constraints.extend(singleton_facts(state.heap))
    return constraints
