from dataclasses import replace

import z3

from ...domain.trace import CodeSummary, Uop
from ...symbolic.escape import escape
from ...symbolic.functions import (
    code_framesize,
    code_nlocals,
    func_argcount,
    im_func,
    im_self,
    type_getitem,
    type_getitem_version,
    type_init,
    type_is_heap,
)
from ...symbolic.heap import new_object
from ...symbolic.identities import METHOD_TAG, type_tag
from ...symbolic.ledger import decref_on_pop, decref_on_pop_if, ledger_add
from ...symbolic.objects import (
    borrowed_if_immortal,
    is_func,
    is_immortal_id,
    is_type_object,
    none_slot,
    slot_is_none,
    slot_object,
)
from ...symbolic.protocols import harmless_to_close
from ...symbolic.sorts import Obj, Slot
from ...symbolic.state import (
    ExitKind,
    FactKind,
    Frame,
    State,
    push,
    set_local,
    set_stack,
    with_constraint,
    with_fact,
)
from ...symbolic.summaries import constant_facts
from ...symbolic.types import has_tag, type_tag_of, version_check
from ..step import (
    Handler,
    UnsupportedUop,
    allocation_exit_if,
    at_depth,
    exit_if,
    local_slot,
    new_ref,
    nos,
    pop_n,
    precondition,
)
from .common import take_ownership

FRAME_MARKER = Slot.TaggedInt(z3.IntVal(-1))
code_start = z3.Function("code_start", z3.IntSort(), z3.IntSort())
func_code_of = z3.Function("func_code", z3.IntSort(), z3.IntSort())
closure_cell = z3.Function("closure_cell", z3.IntSort(), z3.IntSort(), z3.IntSort())


def code_facts(state: State, summary: CodeSummary) -> State:
    version = z3.IntVal(summary.co_version)
    facts = [
        code_nlocals(version) == summary.nlocalsplus,
        code_framesize(version) == summary.framesize,
        func_argcount(version) == summary.argcount,
    ]
    if summary.code_start is not None:
        facts.append(code_start(version) == summary.code_start // 2)
    return with_constraint(state, *facts)


def func_version_axiom(state: State, slot: z3.DatatypeRef) -> State:
    function = slot_object(state, slot)
    return with_constraint(
        state,
        z3.Implies(
            z3.And(Slot.is_Ref(slot), Obj.is_Func(function), Obj.version(function) != 0),
            Obj.code(function) == Obj.version(function),
        ),
    )


def callee_frame(state: State, function: z3.DatatypeRef, locals_: list[z3.DatatypeRef]) -> Frame:
    return Frame(
        code=Obj.code(slot_object(state, function)),
        locals=tuple(locals_),
        stack=(),
        ip=code_start(Obj.code(slot_object(state, function))),
        return_offset=None,
        func=function,
    )


def current_frame(state: State, return_offset: z3.ArithRef | None) -> Frame:
    return Frame(
        code=state.code,
        locals=state.locals,
        stack=state.stack,
        ip=state.ip,
        return_offset=return_offset,
        func=state.func,
        c_recursion_checked=state.c_recursion_checked,
        globals_version=state.globals_version,
        frame_id=state.frame_id,
    )


def resume_caller(state: State, ip_offset: z3.ArithRef) -> State:
    caller = state.callers[-1]
    return replace(
        state,
        callers=state.callers[:-1],
        locals=caller.locals,
        stack=caller.stack,
        ip=caller.ip + ip_offset,
        code=caller.code,
        func=caller.func,
        return_offset=None,
        c_recursion_checked=caller.c_recursion_checked,
        frame_id=caller.frame_id,
        globals_version=caller.globals_version,
    )


def return_offset_of(caller: Frame, uop: Uop) -> z3.ArithRef:
    if caller.return_offset is not None:
        return caller.return_offset
    return z3.Int(f"return_offset_p{uop.i}")


def check_function_version_at(extra: int) -> Handler:
    def handler(uop: Uop, state: State) -> State:
        callable_ = at_depth(state, uop, uop.oparg + 2 + extra)
        state = func_version_axiom(state, callable_)
        matches = z3.And(
            is_func(state, callable_),
            Obj.version(slot_object(state, callable_)) == z3.IntVal(uop.operand0),
        )
        return exit_if(uop, state, ExitKind.SIDE, z3.Not(matches))

    return handler


check_function_version = check_function_version_at(0)


def check_function_version_inline(uop: Uop, state: State) -> State:
    recorded = uop.operand1_object
    if recorded is None or recorded.func_version is None:
        raise UnsupportedUop(f"{uop.op} at {uop.i}: no function recorded for the operand")
    ident, state = constant_facts(state, recorded)
    condition = Obj.version(z3.Select(state.heap, ident)) != z3.IntVal(uop.operand0)
    return exit_if(uop, state, ExitKind.SIDE, condition)


def check_function_exact_args(uop: Uop, state: State) -> State:
    callable_ = at_depth(state, uop, uop.oparg + 2)
    self_or_null = at_depth(state, uop, uop.oparg + 1)
    state = precondition(uop, state, is_func(state, callable_), "callable is a function")
    expected = z3.IntVal(uop.oparg) + z3.If(Slot.is_Null(self_or_null), z3.IntVal(0), z3.IntVal(1))
    condition = func_argcount(Obj.code(slot_object(state, callable_))) != expected
    return exit_if(uop, state, ExitKind.SIDE, condition)


def check_stack_space(uop: Uop, state: State) -> State:
    callable_ = at_depth(state, uop, uop.oparg + 2)
    state = precondition(uop, state, is_func(state, callable_), "callable is a function")
    condition = code_framesize(Obj.code(slot_object(state, callable_))) > state.stack_space
    return exit_if(uop, state, ExitKind.DEOPT, condition)


def check_stack_space_operand(uop: Uop, state: State) -> State:
    return exit_if(uop, state, ExitKind.DEOPT, z3.IntVal(uop.operand0) > state.stack_space)


def check_recursion_remaining(uop: Uop, state: State) -> State:
    return exit_if(uop, state, ExitKind.DEOPT, state.recursion_remaining <= 1)


def check_recursion_limit(uop: Uop, state: State) -> State:
    reached = z3.Bool(f"c_recursion_limit_f{state.frame_id}")
    state = replace(state, oracle=state.oracle + (reached,), c_recursion_checked=True)
    return exit_if(uop, state, ExitKind.SIDE, reached)


def init_call_py_exact_args(uop: Uop, state: State) -> State:
    if uop.callee_version is None or uop.callee_version not in state.codes:
        raise UnsupportedUop(
            f"{uop.op} at {uop.i}: the callee code object is not known from the trace"
        )
    summary = state.codes[uop.callee_version]
    state = code_facts(state, summary)
    slots, state = pop_n(state, uop, uop.oparg + 2)
    callable_, self_or_null, args = slots[0], slots[1], slots[2:]
    state = precondition(uop, state, is_func(state, callable_), "callable is a function")
    state = func_version_axiom(state, callable_)
    has_self = z3.Not(Slot.is_Null(self_or_null))
    nlocals = summary.nlocalsplus
    if len(args) > nlocals:
        raise UnsupportedUop(
            f"{uop.op} at {uop.i}: {len(args)} arguments do not fit {nlocals} locals"
        )
    with_self = [self_or_null] + args
    locals_: list[z3.DatatypeRef] = []
    for index in range(nlocals):
        plain = args[index] if index < len(args) else Slot.Null
        shifted = with_self[index] if index < len(with_self) else Slot.Null
        locals_.append(z3.If(has_self, shifted, plain))
    state = replace(state, pending_frame=callee_frame(state, callable_, locals_))
    return push(state, FRAME_MARKER)


def push_frame(uop: Uop, state: State) -> State:
    marker, state = pop_n(state, uop, 1)
    if state.pending_frame is None:
        raise UnsupportedUop(f"{uop.op} at {uop.i}: no frame was prepared")
    caller = current_frame(state, state.return_offset)
    callee = state.pending_frame
    return replace(
        state,
        callers=state.callers + (caller,),
        locals=callee.locals,
        stack=callee.stack,
        ip=callee.ip,
        code=callee.code,
        func=callee.func,
        return_offset=None,
        pending_frame=None,
        stack_space=state.stack_space - code_framesize(callee.code),
        recursion_remaining=state.recursion_remaining - 1,
        frame_count=state.frame_count + 1,
        frame_id=state.frame_count + 1,
        c_recursion_checked=False,
        globals_version=z3.Int(f"globals_version_f{state.frame_count + 1}_p{uop.i}"),
    )


def save_return_offset(uop: Uop, state: State) -> State:
    return replace(state, return_offset=z3.IntVal(uop.oparg))


def clear_frame_locals(uop: Uop, state: State) -> tuple[z3.BoolRef, State]:
    dangerous: list[z3.BoolRef] = []
    for slot in state.locals:
        state = replace(state, ledger=decref_on_pop(state, slot))
        dangerous.append(z3.Not(harmless_to_close(state, slot)))
    state = replace(state, ledger=decref_on_pop(state, state.func))
    dangerous.append(z3.Not(harmless_to_close(state, state.func)))
    return z3.Or(*dangerous) if dangerous else z3.BoolVal(False), state


def return_value(uop: Uop, state: State) -> State:
    (retval,), state = pop_n(state, uop, 1)
    if state.stack:
        raise UnsupportedUop(
            f"{uop.op} at {uop.i}: the callee stack is not empty after popping the result"
        )
    happens, state = clear_frame_locals(uop, state)
    _, state = escape(state, uop.i, uop.op, happens, False)
    returned_code = state.code
    if state.callers:
        state = resume_caller(state, return_offset_of(state.callers[-1], uop))
    else:
        state = unknown_caller(uop, state, None)
    space = state.stack_space
    remaining = state.recursion_remaining
    state = replace(
        state, stack_space=space + code_framesize(returned_code), recursion_remaining=remaining + 1
    )
    state = with_fact(state, uop.i, FactKind.FRAME, uop.op, None, "frame popped")
    return push(state, retval)


def guard_code_version(uop: Uop, state: State) -> State:
    return exit_if(uop, state, ExitKind.SIDE, state.code != z3.IntVal(uop.operand0))


def guard_ip(uop: Uop, state: State) -> State:
    return exit_if(uop, state, ExitKind.SIDE, state.ip != z3.IntVal(uop.operand0 // 2))


def copy_free_vars(uop: Uop, state: State) -> State:
    nlocals = len(state.locals)
    if uop.oparg > nlocals:
        raise UnsupportedUop(
            f"{uop.op} at {uop.i}: {uop.oparg} free variables exceed {nlocals} locals"
        )
    offset = nlocals - uop.oparg
    locals_ = list(state.locals)
    for index in range(uop.oparg):
        cell_id = closure_cell(Slot.id(state.func), z3.IntVal(index))
        slot, state = new_ref(state, cell_id)
        locals_[offset + index] = slot
    return replace(state, locals=tuple(locals_))


def load_deref(uop: Uop, state: State) -> State:
    before = state
    cell = local_slot(uop, state)
    state = precondition(
        uop,
        state,
        z3.And(Slot.is_Ref(cell), Obj.is_Cell(slot_object(state, cell))),
        f"local {uop.oparg} is a cell",
    )
    content = Obj.content(slot_object(state, cell))
    state = exit_if(uop, state, ExitKind.ERROR, Slot.is_Null(content), handover=before)
    slot, state = new_ref(state, Slot.id(content))
    return push(state, slot)


def store_deref(uop: Uop, state: State) -> State:
    (value,), state = pop_n(state, uop, 1)
    cell = local_slot(uop, state)
    state = precondition(
        uop,
        state,
        z3.And(Slot.is_Ref(cell), Obj.is_Cell(slot_object(state, cell))),
        f"local {uop.oparg} is a cell",
    )
    old = Obj.content(slot_object(state, cell))
    state = replace(state, ledger=decref_on_pop(state, old))
    _, state = escape(state, uop.i, uop.op, z3.Not(harmless_to_close(state, old)), False)
    stored = z3.If(Slot.is_Ref(value), Slot.Ref(Slot.id(value), z3.BoolVal(False)), value)
    state = take_ownership(state, value)
    return replace(state, heap=z3.Store(state.heap, Slot.id(cell), Obj.Cell(stored)))


def make_cell(uop: Uop, state: State) -> State:
    before = state
    initial = local_slot(uop, state)
    stored = z3.If(Slot.is_Ref(initial), Slot.Ref(Slot.id(initial), z3.BoolVal(False)), initial)
    slot, failure, state = new_object(state, Obj.Cell(stored), uop.i, "_c")
    state = allocation_exit_if(uop, state, ExitKind.ERROR, failure, handover=before)
    return set_local(state, uop.oparg, slot)


INLINE_CACHE_ENTRIES_CALL = 3
INLINE_CACHE_ENTRIES_SEND = 1
INLINE_CACHE_ENTRIES_FOR_ITER = 1


def symbolic_locals(
    uop: Uop, state: State, count: int, prefix: str
) -> tuple[list[z3.DatatypeRef], State]:
    slots = [z3.Const(f"{prefix}{j}_p{uop.i}", Slot) for j in range(count)]
    for slot in slots:
        state = with_constraint(state, borrowed_if_immortal(state, slot))
    state = replace(state, ids=state.ids + tuple(Slot.id(slot) for slot in slots))
    return slots, state


def callee_summary(uop: Uop, state: State) -> CodeSummary:
    if uop.callee_version is None or uop.callee_version not in state.codes:
        raise UnsupportedUop(
            f"{uop.op} at {uop.i}: the callee code object is not known from the trace"
        )
    return state.codes[uop.callee_version]


def check_object(uop: Uop, state: State) -> State:
    callable_ = at_depth(state, uop, uop.oparg + 2)
    self_or_null = at_depth(state, uop, uop.oparg + 1)
    matches, state = version_check(state, Obj.ttag(slot_object(state, callable_)), uop.operand0)
    allocatable = z3.And(Slot.is_Null(self_or_null), is_type_object(state, callable_), matches)
    return exit_if(uop, state, ExitKind.SIDE, z3.Not(allocatable))


def allocate_object(uop: Uop, state: State) -> State:
    before = state
    depth = uop.oparg + 2
    callable_ = at_depth(state, uop, depth)
    self_or_null = at_depth(state, uop, depth - 1)
    state = precondition(
        uop,
        state,
        z3.And(is_type_object(state, callable_), Slot.is_Null(self_or_null)),
        "callable is a type and self is NULL",
    )
    tag = Obj.ttag(slot_object(state, callable_))
    init_id = type_init(tag)
    state = with_constraint(state, Obj.is_Func(z3.Select(state.heap, init_id)))
    state = func_version_axiom(state, Slot.Ref(init_id, z3.BoolVal(True)))
    shim = state.codes.get(0)
    shim_size = z3.IntVal(shim.framesize) if shim is not None else z3.Int("init_cleanup_framesize")
    needed = code_framesize(Obj.code(z3.Select(state.heap, init_id))) + shim_size
    state = exit_if(uop, state, ExitKind.SIDE, needed > state.stack_space, handover=before)
    self_slot, failure, state = new_object(state, Obj.Other(tag), uop.i, "_o")
    state = allocation_exit_if(uop, state, ExitKind.ERROR, failure, handover=before)
    init_ref, state = new_ref(state, init_id)
    state = replace(state, ledger=decref_on_pop(state, callable_))
    state = set_stack(state, depth, init_ref)
    return set_stack(state, depth - 1, self_slot)


def create_init_frame(uop: Uop, state: State) -> State:
    summary = callee_summary(uop, state)
    state = code_facts(state, summary)
    if uop.return_code is None:
        raise UnsupportedUop(
            f"{uop.op} at {uop.i}: the init shim code is not described by the trace"
        )
    shim = uop.return_code
    state = code_facts(state, shim)
    slots, state = pop_n(state, uop, uop.oparg + 2)
    init, self_, args = slots[0], slots[1], slots[2:]
    state = precondition(uop, state, is_func(state, init), "init is a function")
    state = func_version_axiom(state, init)
    nlocals = summary.nlocalsplus
    if len(args) + 1 > nlocals:
        raise UnsupportedUop(
            f"{uop.op} at {uop.i}: {len(args) + 1} arguments do not fit {nlocals} locals"
        )
    init_locals = [self_] + args + [Slot.Null] * (nlocals - len(args) - 1)
    self_dup = Slot.Ref(Slot.id(self_), is_immortal_id(state.consts, Slot.id(self_)))
    state = replace(
        state,
        ledger=ledger_add(
            state.ledger, z3.Not(is_immortal_id(state.consts, Slot.id(self_))), Slot.id(self_), -1
        ),
    )
    shim_code = z3.IntVal(shim.co_version)
    caller = current_frame(state, z3.IntVal(1 + INLINE_CACHE_ENTRIES_CALL))
    pending = callee_frame(state, init, init_locals)
    state = replace(
        state,
        callers=state.callers + (caller,),
        locals=(),
        stack=(self_dup,),
        ip=code_start(shim_code),
        code=shim_code,
        func=Slot.Null,
        return_offset=z3.IntVal(0),
        pending_frame=pending,
        stack_space=state.stack_space - z3.IntVal(shim.framesize),
        recursion_remaining=state.recursion_remaining - 1,
        frame_count=state.frame_count + 1,
        globals_version=z3.Int(f"shim_globals_version_p{uop.i}"),
    )
    return push(state, FRAME_MARKER)


def exit_init_check(uop: Uop, state: State) -> State:
    before = state
    (value,), state = pop_n(state, uop, 1)
    return exit_if(uop, state, ExitKind.ERROR, z3.Not(slot_is_none(value)), handover=before)


def frame_init_error(uop: Uop) -> z3.BoolRef:
    error = z3.Function(f"frame_init_error_p{uop.i}", z3.BoolSort())
    return error()


def py_frame_general(uop: Uop, state: State) -> State:
    before = state
    summary = callee_summary(uop, state)
    state = code_facts(state, summary)
    extra = 1 if uop.op == "_PY_FRAME_KW" else 0
    slots, state = pop_n(state, uop, uop.oparg + 2 + extra)
    callable_ = slots[0]
    state = precondition(uop, state, is_func(state, callable_), "callable is a function")
    state = func_version_axiom(state, callable_)
    locals_, state = symbolic_locals(uop, state, summary.nlocalsplus, "callee_local")
    state = exit_if(uop, state, ExitKind.ERROR, frame_init_error(uop), handover=before)
    state = replace(state, pending_frame=callee_frame(state, callable_, locals_))
    return push(state, FRAME_MARKER)


def py_frame_ex(uop: Uop, state: State) -> State:
    before = state
    summary = callee_summary(uop, state)
    state = code_facts(state, summary)
    (func, null, callargs, kwargs), state = pop_n(state, uop, 4)
    state = precondition(uop, state, is_func(state, func), "callable is a function")
    state = func_version_axiom(state, func)
    locals_, state = symbolic_locals(uop, state, summary.nlocalsplus, "callee_local")
    state = exit_if(uop, state, ExitKind.ERROR, frame_init_error(uop), handover=before)
    state = replace(state, pending_frame=callee_frame(state, func, locals_))
    return push(state, FRAME_MARKER)


def load_attr_property_frame(uop: Uop, state: State) -> State:
    before = state
    getter = uop.operand1_object
    if getter is None or getter.func_version is None or getter.code is None:
        raise UnsupportedUop(f"{uop.op} at {uop.i}: no getter function recorded for the operand")
    ident, state = constant_facts(state, getter)
    state = code_facts(state, getter.code)
    (owner,), state = pop_n(state, uop, 1)
    fget = z3.Select(state.heap, ident)
    state = exit_if(
        uop, state, ExitKind.SIDE, Obj.version(fget) != z3.IntVal(uop.operand0), handover=before
    )
    state = exit_if(
        uop,
        state,
        ExitKind.SIDE,
        code_framesize(Obj.code(fget)) > state.stack_space,
        handover=before,
    )
    func_slot, state = new_ref(state, ident)
    locals_ = [owner] + [Slot.Null] * (getter.code.nlocalsplus - 1)
    frame = Frame(
        code=Obj.code(fget),
        locals=tuple(locals_),
        stack=(),
        ip=code_start(Obj.code(fget)),
        return_offset=None,
        func=func_slot,
    )
    state = replace(state, pending_frame=frame)
    return push(state, FRAME_MARKER)


def generator_frame(
    uop: Uop, state: State, gen: z3.DatatypeRef, pushed: z3.DatatypeRef, return_offset: int
) -> State:
    summary = callee_summary(uop, state)
    state = code_facts(state, summary)
    depth = uop.return_depth if uop.return_depth is not None else 0
    locals_, state = symbolic_locals(uop, state, summary.nlocalsplus, "gen_local")
    stack, state = symbolic_locals(uop, state, depth, "gen_stack")
    gen_code = z3.Function("gen_code", z3.IntSort(), state.heap.sort(), z3.IntSort(), z3.IntSort())(
        Slot.id(gen), state.heap, state.epoch
    )
    gen_ip = z3.Function("gen_ip", z3.IntSort(), state.heap.sort(), z3.IntSort(), z3.IntSort())(
        Slot.id(gen), state.heap, state.epoch
    )
    frame = Frame(
        code=gen_code,
        locals=tuple(locals_),
        stack=tuple(stack) + (pushed,),
        ip=gen_ip,
        return_offset=None,
        func=Slot.Null,
    )
    state = replace(state, pending_frame=frame, return_offset=z3.IntVal(return_offset))
    return push(state, FRAME_MARKER)


def generator_executing(state: State, generator: z3.DatatypeRef) -> z3.BoolRef:
    executing = z3.Function(
        "gen_executing", z3.IntSort(), state.heap.sort(), z3.IntSort(), z3.BoolSort()
    )
    return executing(Slot.id(generator), state.heap, state.epoch)


def for_iter_gen_frame(uop: Uop, state: State) -> State:
    iterator = nos(state, uop)
    state = precondition(uop, state, Slot.is_Ref(iterator), "iterator is an object")
    executing = generator_executing(state, iterator)
    condition = z3.Or(z3.Not(has_tag(state, iterator, type_tag("generator"))), executing)
    state = exit_if(uop, state, ExitKind.SIDE, condition)
    return generator_frame(
        uop, state, iterator, none_slot(), 1 + INLINE_CACHE_ENTRIES_FOR_ITER + uop.oparg
    )


def send_gen_frame(uop: Uop, state: State) -> State:
    (sent,), state = pop_n(state, uop, 1)
    receiver = nos(state, uop)
    state = precondition(uop, state, Slot.is_Ref(receiver), "receiver is an object")
    executing = generator_executing(state, receiver)
    condition = z3.Or(
        z3.Not(
            z3.Or(
                has_tag(state, receiver, type_tag("generator")),
                has_tag(state, receiver, type_tag("coroutine")),
            )
        ),
        executing,
    )
    state = exit_if(uop, state, ExitKind.SIDE, condition)
    mortal_borrowed = z3.And(
        Slot.is_Ref(sent), Slot.borrowed(sent), z3.Not(is_immortal_id(state.consts, Slot.id(sent)))
    )
    owned = Slot.Ref(Slot.id(sent), z3.BoolVal(False))
    state = replace(state, ledger=ledger_add(state.ledger, mortal_borrowed, Slot.id(sent), -1))
    return generator_frame(
        uop,
        state,
        receiver,
        z3.If(mortal_borrowed, owned, sent),
        1 + INLINE_CACHE_ENTRIES_SEND + uop.oparg,
    )


def unexposed_fresh_distinct(state: State, slot: z3.DatatypeRef) -> list[z3.BoolRef]:
    exposed = {z3.simplify(Slot.id(e)).get_id() for e, _ in state.exposed}
    return [
        z3.Implies(Slot.is_Ref(slot), Slot.id(slot) != fresh)
        for fresh in state.fresh_ids
        if fresh.get_id() not in exposed
    ]


def unknown_caller(uop: Uop, state: State, resume_delta: z3.ArithRef | None) -> State:
    if uop.return_code is None or uop.return_depth is None:
        raise UnsupportedUop(f"{uop.op} at {uop.i}: the caller frame is not described by the trace")
    summary = uop.return_code
    state = code_facts(state, summary)
    locals_ = tuple(z3.Const(f"caller_local{j}_p{uop.i}", Slot) for j in range(summary.nlocalsplus))
    depth = max(uop.return_depth - 1, 0)
    stack = tuple(z3.Const(f"caller_stack{k}_p{uop.i}", Slot) for k in range(depth))
    ip = z3.Int(f"caller_ip_p{uop.i}")
    offset = resume_delta if resume_delta is not None else z3.Int(f"caller_return_offset_p{uop.i}")
    func = z3.Const(f"caller_func_p{uop.i}", Slot)
    for slot in locals_ + stack + (func,):
        state = with_constraint(state, borrowed_if_immortal(state, slot))
        state = with_constraint(state, *unexposed_fresh_distinct(state, slot))
    return replace(
        state,
        locals=locals_,
        stack=stack,
        ip=ip + offset,
        code=z3.Int(f"caller_code_p{uop.i}"),
        func=func,
        return_offset=None,
        ids=state.ids + tuple(Slot.id(s) for s in locals_ + stack),
        c_recursion_checked=False,
        frame_id=-(uop.i + 1),
        globals_version=z3.Int(f"caller_globals_version_p{uop.i}"),
    )


def yield_value(uop: Uop, state: State) -> State:
    (retval,), state = pop_n(state, uop, 1)
    remaining = state.recursion_remaining
    if not state.callers:
        state = unknown_caller(uop, state, z3.IntVal(1 + INLINE_CACHE_ENTRIES_SEND))
        state = replace(state, recursion_remaining=remaining + 1)
        state = with_fact(
            state,
            uop.i,
            FactKind.FRAME,
            uop.op,
            None,
            "generator frame suspended into an unknown caller",
        )
        return push(state, retval)
    state = resume_caller(state, z3.IntVal(1 + INLINE_CACHE_ENTRIES_SEND))
    state = replace(state, recursion_remaining=remaining + 1)
    state = with_fact(state, uop.i, FactKind.FRAME, uop.op, None, "generator frame suspended")
    return push(state, retval)


def return_generator(uop: Uop, state: State) -> State:
    before = state
    if not state.callers:
        raise UnsupportedUop(f"{uop.op} at {uop.i}: returning a generator from the root frame")
    gen, failure, state = new_object(
        state, Obj.Other(z3.IntVal(type_tag("generator"))), uop.i, "_g"
    )
    state = allocation_exit_if(uop, state, ExitKind.ERROR, failure, handover=before)
    returned_code = state.code
    offset = return_offset_of(state.callers[-1], uop)
    space = state.stack_space
    remaining = state.recursion_remaining
    state = resume_caller(state, offset)
    state = replace(
        state,
        stack_space=space + code_framesize(returned_code),
        recursion_remaining=remaining + 1,
    )
    return push(state, gen)


def check_method_version_at(extra: int) -> Handler:
    def handler(uop: Uop, state: State) -> State:
        callable_ = at_depth(state, uop, uop.oparg + 2 + extra)
        null = at_depth(state, uop, uop.oparg + 1 + extra)
        func_id = im_func(Slot.id(callable_))
        function = z3.Select(state.heap, func_id)
        state = func_version_axiom(state, Slot.Ref(func_id, z3.BoolVal(True)))
        matches = z3.And(
            has_tag(state, callable_, METHOD_TAG),
            Obj.is_Func(function),
            Obj.version(function) == z3.IntVal(uop.operand0),
            Slot.is_Null(null),
        )
        return exit_if(uop, state, ExitKind.SIDE, z3.Not(matches))

    return handler


check_method_version = check_method_version_at(0)


def check_call_bound_method_exact_args(uop: Uop, state: State) -> State:
    callable_ = at_depth(state, uop, uop.oparg + 2)
    null = at_depth(state, uop, uop.oparg + 1)
    bound_method = z3.And(Slot.is_Null(null), has_tag(state, callable_, METHOD_TAG))
    return exit_if(uop, state, ExitKind.SIDE, z3.Not(bound_method))


def expand_method_at(extra: int) -> Handler:
    def handler(uop: Uop, state: State) -> State:
        depth = uop.oparg + 2 + extra
        callable_ = at_depth(state, uop, depth)
        null = at_depth(state, uop, depth - 1)
        state = precondition(
            uop,
            state,
            z3.And(Slot.is_Null(null), has_tag(state, callable_, METHOD_TAG)),
            "callable is a bound method and self is NULL",
        )
        func_slot, state = new_ref(state, im_func(Slot.id(callable_)))
        self_slot, state = new_ref(state, im_self(Slot.id(callable_)))
        state = replace(state, ledger=decref_on_pop(state, callable_))
        _, state = escape(state, uop.i, uop.op, z3.Not(harmless_to_close(state, callable_)), False)
        state = set_stack(state, depth, func_slot)
        return set_stack(state, depth - 1, self_slot)

    return handler


expand_method = expand_method_at(0)


def maybe_expand_method(uop: Uop, state: State) -> State:
    depth = uop.oparg + 2
    callable_ = at_depth(state, uop, depth)
    null = at_depth(state, uop, depth - 1)
    bound = z3.And(Slot.is_Null(null), has_tag(state, callable_, METHOD_TAG))
    func_slot, state = new_ref(state, im_func(Slot.id(callable_)))
    self_slot, state = new_ref(state, im_self(Slot.id(callable_)))
    state = replace(state, ledger=decref_on_pop_if(state, bound, callable_))
    _, state = escape(
        state, uop.i, uop.op, z3.And(bound, z3.Not(harmless_to_close(state, callable_))), False
    )
    state = set_stack(state, depth, z3.If(bound, func_slot, callable_))
    return set_stack(state, depth - 1, z3.If(bound, self_slot, null))


def binary_op_subscr_check_func(uop: Uop, state: State) -> State:
    before = state
    container = nos(state, uop)
    state = precondition(uop, state, Slot.is_Ref(container), "container is an object")
    tag = type_tag_of(state, Slot.id(container))
    getitem_id = type_getitem(tag)
    getitem = z3.Select(state.heap, getitem_id)
    state = func_version_axiom(state, Slot.Ref(getitem_id, z3.BoolVal(True)))
    condition = z3.Or(
        z3.Not(type_is_heap(tag)),
        z3.Not(Obj.is_Func(getitem)),
        Obj.version(getitem) != type_getitem_version(tag),
        code_framesize(Obj.code(getitem)) > state.stack_space,
    )
    state = exit_if(uop, state, ExitKind.SIDE, condition, handover=before)
    slot, state = new_ref(state, getitem_id)
    return push(state, slot)


def binary_op_subscr_init_call(uop: Uop, state: State) -> State:
    summary = callee_summary(uop, state)
    state = code_facts(state, summary)
    (container, sub, getitem), state = pop_n(state, uop, 3)
    state = precondition(uop, state, is_func(state, getitem), "getitem is a function")
    state = func_version_axiom(state, getitem)
    nlocals = summary.nlocalsplus
    if nlocals < 2:
        raise UnsupportedUop(f"{uop.op} at {uop.i}: the getitem code has fewer than two locals")
    locals_ = [container, sub] + [Slot.Null] * (nlocals - 2)
    frame = callee_frame(state, getitem, locals_)
    state = replace(state, pending_frame=frame, return_offset=z3.IntVal(6))
    return push(state, FRAME_MARKER)


HANDLERS: dict[str, Handler] = {
    "_BINARY_OP_SUBSCR_CHECK_FUNC": binary_op_subscr_check_func,
    "_BINARY_OP_SUBSCR_INIT_CALL": binary_op_subscr_init_call,
    "_CHECK_OBJECT": check_object,
    "_ALLOCATE_OBJECT": allocate_object,
    "_CREATE_INIT_FRAME": create_init_frame,
    "_EXIT_INIT_CHECK": exit_init_check,
    "_PY_FRAME_GENERAL": py_frame_general,
    "_PY_FRAME_KW": py_frame_general,
    "_PY_FRAME_EX": py_frame_ex,
    "_LOAD_ATTR_PROPERTY_FRAME": load_attr_property_frame,
    "_FOR_ITER_GEN_FRAME": for_iter_gen_frame,
    "_SEND_GEN_FRAME": send_gen_frame,
    "_YIELD_VALUE": yield_value,
    "_RETURN_GENERATOR": return_generator,
    "_GUARD_IP_YIELD_VALUE": guard_ip,
    "_GUARD_IP_RETURN_GENERATOR": guard_ip,
    "_GUARD_CODE_VERSION_YIELD_VALUE": guard_code_version,
    "_GUARD_CODE_VERSION_RETURN_GENERATOR": guard_code_version,
    "_CHECK_METHOD_VERSION": check_method_version,
    "_CHECK_CALL_BOUND_METHOD_EXACT_ARGS": check_call_bound_method_exact_args,
    "_EXPAND_METHOD": expand_method,
    "_INIT_CALL_BOUND_METHOD_EXACT_ARGS": expand_method,
    "_MAYBE_EXPAND_METHOD": maybe_expand_method,
    "_CHECK_FUNCTION_VERSION": check_function_version,
    "_CHECK_FUNCTION_VERSION_KW": check_function_version_at(1),
    "_CHECK_METHOD_VERSION_KW": check_method_version_at(1),
    "_EXPAND_METHOD_KW": expand_method_at(1),
    "_CHECK_FUNCTION_VERSION_INLINE": check_function_version_inline,
    "_CHECK_FUNCTION_EXACT_ARGS": check_function_exact_args,
    "_CHECK_STACK_SPACE": check_stack_space,
    "_CHECK_STACK_SPACE_OPERAND": check_stack_space_operand,
    "_CHECK_RECURSION_REMAINING": check_recursion_remaining,
    "_CHECK_RECURSION_LIMIT": check_recursion_limit,
    "_INIT_CALL_PY_EXACT_ARGS": init_call_py_exact_args,
    "_PUSH_FRAME": push_frame,
    "_SAVE_RETURN_OFFSET": save_return_offset,
    "_RETURN_VALUE": return_value,
    "_GUARD_CODE_VERSION__PUSH_FRAME": guard_code_version,
    "_GUARD_CODE_VERSION_RETURN_VALUE": guard_code_version,
    "_GUARD_IP__PUSH_FRAME": guard_ip,
    "_GUARD_IP_RETURN_VALUE": guard_ip,
    "_COPY_FREE_VARS": copy_free_vars,
    "_LOAD_DEREF": load_deref,
    "_STORE_DEREF": store_deref,
    "_MAKE_CELL": make_cell,
}
