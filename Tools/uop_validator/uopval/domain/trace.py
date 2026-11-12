from dataclasses import dataclass

TERMINATORS = ("_JUMP_TO_TOP", "_EXIT_TRACE", "_DEOPT", "_DYNAMIC_EXIT")


@dataclass(frozen=True)
class Provenance:
    path: str
    line: int
    pid: int | None
    seq: int | None


@dataclass(frozen=True, kw_only=True)
class CodeSummary:
    qualname: str
    addr: int | None = None
    code_start: int | None = None
    co_version: int
    nlocalsplus: int
    argcount: int
    stacksize: int
    framesize: int
    flags: int
    code_units: int
    base_code_hash: int


@dataclass(frozen=True, kw_only=True)
class Constant:
    type: str
    immortal: bool
    int_value: int | None = None
    compact: bool | None = None
    overflow: int | None = None
    value: str | None = None
    str_value: str | None = None
    str_length: int | None = None
    addr: int | None = None
    float_text: str | None = None
    float_hex: str | None = None
    bytes_length: int | None = None
    length: int | None = None
    items: tuple[Constant, ...] | None = None
    pairs: tuple[tuple[Constant, Constant], ...] | None = None
    items_truncated: bool = False
    type_name: str | None = None
    version_tag: int | None = None
    immutable: bool | None = None
    func_version: int | None = None
    qualname: str | None = None
    code: CodeSummary | None = None
    im_func: Constant | None = None
    im_self: Constant | None = None
    iter_kind: str | None = None
    iternext_inline: bool | None = None
    dict_subscript: bool | None = None
    dict_store_subscript: bool | None = None
    vectorcall: bool | None = None
    ml_meth: int | None = None
    ml_flags: int | None = None
    callable_cache: str | None = None
    builtin_name: str | None = None
    d_type: str | None = None
    init: Constant | None = None
    getitem: Constant | None = None
    getitem_version: int | None = None
    heap_type: bool | None = None
    generic_getattr: bool | None = None
    mro_addrs: tuple[int, ...] | None = None
    type_addr: int | None = None
    type_version: int | None = None
    type_immutable: bool | None = None


@dataclass(frozen=True)
class BinopDescr:
    lhs_type: Constant | None
    rhs_type: Constant | None
    result_type: Constant | None
    result_unique: bool
    has_guard: bool


@dataclass(frozen=True, kw_only=True)
class Uop:
    i: int
    op: str
    opcode: int
    oparg: int
    format: int
    target: int | None = None
    operand0: int
    operand1: int
    origin: int | None = None
    instr_index: int | None = None
    constant: Constant | None = None
    constant_code: CodeSummary | None = None
    global_value: Constant | None = None
    lookup: Constant | None = None
    lookup_type: Constant | None = None
    recorded_type: str | None = None
    recorded: Constant | None = None
    operand0_object: Constant | None = None
    operand1_object: Constant | None = None
    binop_descr: BinopDescr | None = None
    callee_version: int | None = None
    return_code: CodeSummary | None = None
    return_depth: int | None = None


@dataclass(frozen=True, kw_only=True)
class CodeIdentity:
    qualname: str
    filename: str
    firstlineno: int
    code_units: int
    nlocalsplus: int
    stacksize: int
    base_code_hash: int
    co_version: int | None = None


@dataclass(frozen=True)
class Entry:
    ip0: int
    stack_depth: int
    chain_depth: int
    from_side_exit: bool


@dataclass(frozen=True)
class Rewrite:
    i: int
    op_after: str


@dataclass(frozen=True)
class Unit:
    provenance: Provenance
    python_version: str
    code: CodeIdentity
    entry: Entry
    noopt: bool
    result_length: int
    input: tuple[Uop, ...]
    input_tail: tuple[Uop, ...]
    rewritten_in_place: tuple[Rewrite, ...]
    output: tuple[Uop, ...] | None
