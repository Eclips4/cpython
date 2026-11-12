import gzip
import json
import os
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any, TextIO

from adaptix import Chain, DebugTrail, P, Retort, dumper, loader, name_mapping
from adaptix.load_error import LoadError, NoRequiredFieldsLoadError, TypeLoadError, ValueLoadError
from adaptix.struct_trail import get_trail

from ..domain.trace import (
    TERMINATORS,
    CodeIdentity,
    CodeSummary,
    Constant,
    Entry,
    Provenance,
    Rewrite,
    Unit,
    Uop,
)

SCHEMAS = (1, 2)
HEX_FIELDS = (
    P[CodeSummary].addr
    | P[CodeSummary].code_start
    | P[CodeSummary].base_code_hash
    | P[CodeIdentity].base_code_hash
    | P[Constant].addr
    | P[Constant].ml_meth
    | P[Constant].type_addr
    | P[Uop].operand0
    | P[Uop].operand1
)


@dataclass(frozen=True)
class Rejected:
    provenance: Provenance
    reason: str
    detail: str


class CaptureError(Exception):
    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


class HexLoadError(ValueLoadError):
    pass


@dataclass(frozen=True)
class Record:
    schema: int
    python_version: str
    code: CodeIdentity
    entry: Entry
    noopt: bool
    result_length: int
    input_length: int
    input: tuple[Uop, ...]
    input_rewritten_in_place: tuple[Rewrite, ...]
    output: tuple[Uop, ...] | None


def load_hex(data: object) -> int:
    if isinstance(data, str) and data.startswith("0x"):
        try:
            return int(data, 16)
        except ValueError:
            pass
    raise HexLoadError("expected a 0x-prefixed hex string", data)


def dump_hex(value: int) -> str:
    return f"0x{value:x}"


def load_hex_list(data: object) -> tuple[int, ...]:
    if not isinstance(data, list):
        raise TypeLoadError(list, data)
    return tuple(load_hex(x) for x in data)


def dump_hex_list(values: tuple[int, ...]) -> list[str]:
    return [dump_hex(x) for x in values]


def split_pairs(data: Any) -> Any:
    items = data.get("items") if isinstance(data, dict) else None
    if isinstance(items, list) and items and all(isinstance(x, dict) and "key" in x for x in items):
        return data | {"items": [], "pairs": [(x["key"], x.get("value")) for x in items]}
    return data


def join_pairs(data: dict[str, Any]) -> dict[str, Any]:
    if "pairs" not in data:
        return data
    rest = {key: value for key, value in data.items() if key != "pairs"}
    return rest | {"items": [{"key": key, "value": value} for key, value in data["pairs"]]}


retort = Retort(
    debug_trail=DebugTrail.FIRST,
    recipe=[
        loader(HEX_FIELDS, load_hex),
        dumper(HEX_FIELDS, dump_hex),
        loader(P[Constant].mro_addrs, load_hex_list),
        dumper(P[Constant].mro_addrs, dump_hex_list),
        loader(Constant, split_pairs, Chain.FIRST),
        dumper(Constant, join_pairs, Chain.LAST),
        name_mapping(CodeSummary, map={"base_code_hash": "base_code_fnv1a64"}, omit_default=True),
        name_mapping(CodeIdentity, map={"base_code_hash": "base_code_fnv1a64"}),
        name_mapping(
            Constant,
            map={
                "int_value": "int",
                "overflow": "int_overflows_long",
                "str_value": "str",
                "float_text": "float",
                "length": "len",
                "mro_addrs": "mro",
            },
            omit_default=True,
        ),
        name_mapping(Uop, map={"global_value": "global"}, omit_default=True),
    ],
)


def first_cause(error: BaseException) -> BaseException:
    if not isinstance(error, BaseExceptionGroup):
        return error
    causes = [
        first_cause(e)
        for e in error.exceptions
        if not (isinstance(e, TypeLoadError) and e.expected_type is None)
    ]
    return causes[0] if causes else error


def trail_text(error: BaseException) -> str:
    return "".join(f"[{x}]" if isinstance(x, int) else f".{x}" for x in get_trail(error)).lstrip(
        "."
    )


def load_failure(error: LoadError, provenance: Provenance) -> Rejected:
    where = trail_text(error) or "record"
    match first_cause(error):
        case HexLoadError(input_value=value):
            return Rejected(provenance, "bad_operand", f"{where}={value!r}")
        case NoRequiredFieldsLoadError(fields=fields):
            return Rejected(provenance, "missing_field", f"{where}: {', '.join(sorted(fields))}")
        case TypeLoadError(expected_type=expected, input_value=value):
            expected_name = getattr(expected, "__name__", expected)
            got = type(value).__name__
            return Rejected(
                provenance, "bad_field_type", f"{where}: expected {expected_name}, got {got}"
            )
        case cause:
            return Rejected(provenance, "bad_field_type", f"{where}: {cause}")


def check_uops(uops: Sequence[Uop], what: str) -> None:
    for expected, u in enumerate(uops):
        if u.i != expected:
            raise CaptureError("uop_index", f"{what} {expected}: got {u.i}")
        if not u.op.startswith("_"):
            raise CaptureError("unnamed_uop", f"{what} {u.i}: {u.op!r}")
        if u.format == 1:
            raise CaptureError("format_at_seam", f"{what} {u.i} in jump format")
        if u.format != 0:
            raise CaptureError("bad_format", f"{what} {u.i}: {u.format}")
        if u.target is None:
            raise CaptureError("missing_field", f"{what}[{u.i}].target")


def terminator_index(uops: Sequence[Uop]) -> int | None:
    for u in uops:
        if u.op in TERMINATORS:
            return u.i
    return None


def check_origins(output: Sequence[Uop], input_length: int) -> None:
    previous = -1
    for u in output:
        if u.origin is None:
            raise CaptureError("origin", f"output {u.i} has no origin")
        if not 0 <= u.origin < input_length:
            raise CaptureError("origin", f"output {u.i} origin {u.origin} outside the input")
        if u.origin < previous:
            raise CaptureError(
                "origin", f"output {u.i} origin {u.origin} goes backwards after {previous}"
            )
        previous = u.origin


def check_output(record: Record) -> tuple[Uop, ...] | None:
    output = record.output
    if record.noopt or record.result_length <= 0:
        if output is not None:
            raise CaptureError(
                "output_presence",
                f"output present with result_length {record.result_length}, noopt {record.noopt}",
            )
        return None
    if output is None:
        raise CaptureError(
            "output_presence", f"output null with result_length {record.result_length}"
        )
    check_uops(output, "output")
    if len(output) != record.result_length:
        raise CaptureError("output_length", f"declared {record.result_length}, found {len(output)}")
    end = terminator_index(output)
    if end != len(output) - 1:
        raise CaptureError("output_terminator", f"terminator at {end} of {len(output)}")
    if record.schema >= 2:
        check_origins(output, len(record.input))
    return output


def unit_from(record: Record, provenance: Provenance) -> Unit:
    if record.schema not in SCHEMAS:
        raise CaptureError("schema", f"got {record.schema}, expected one of {SCHEMAS}")
    uops = record.input
    check_uops(uops, "input")
    if record.input_length != len(uops):
        raise CaptureError("input_length", f"declared {record.input_length}, found {len(uops)}")
    if not uops:
        raise CaptureError("empty_input")
    first, ip0 = uops[0], record.entry.ip0
    if first.op != "_START_EXECUTOR":
        raise CaptureError("entry_uop", f"first uop is {first.op}")
    if first.instr_index != ip0 or first.target != ip0:
        raise CaptureError(
            "entry_uop",
            f"_START_EXECUTOR index {first.instr_index}/target {first.target} vs ip0 {ip0}",
        )
    end = terminator_index(uops)
    if end is None:
        raise CaptureError("no_terminator")
    for r in record.input_rewritten_in_place:
        if not 0 <= r.i < len(uops):
            raise CaptureError("rewrite_index", str(r.i))
    return Unit(
        provenance=provenance,
        python_version=record.python_version,
        code=record.code,
        entry=record.entry,
        noopt=record.noopt,
        result_length=record.result_length,
        input=uops[: end + 1],
        input_tail=uops[end + 1 :],
        rewritten_in_place=record.input_rewritten_in_place,
        output=check_output(record),
    )


def parse_line(line: str, provenance: Provenance) -> Unit | Rejected:
    try:
        data = json.loads(line)
    except json.JSONDecodeError as e:
        return Rejected(provenance, "json", str(e))
    if isinstance(data, dict):
        provenance = replace(provenance, pid=data.get("pid"), seq=data.get("seq"))
    try:
        record = retort.load(data, Record)
    except LoadError as e:
        return load_failure(e, provenance)
    try:
        return unit_from(record, provenance)
    except CaptureError as e:
        return Rejected(provenance, e.reason, e.detail)


def open_text(path: str | os.PathLike[str]) -> TextIO:
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path)


def load(path: str | os.PathLike[str]) -> list[Unit | Rejected]:
    with open_text(path) as f:
        return [
            parse_line(line, Provenance(path=str(path), line=line_no, pid=None, seq=None))
            for line_no, line in enumerate(f, 1)
            if line.strip()
        ]
