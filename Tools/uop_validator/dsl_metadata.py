import json
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Tools/cases_generator"))

from analyzer import Uop, analyze_files  # noqa: E402
from generators_common import cflags  # noqa: E402
from parsing import SimpleStmt, Stmt  # noqa: E402


def simple_statements(stmt: Stmt) -> Iterator[SimpleStmt]:
    if isinstance(stmt, SimpleStmt):
        yield stmt
        return
    for child in getattr(stmt, "body", []) if isinstance(getattr(stmt, "body", None), list) else []:
        yield from simple_statements(child)
    for attr in ("else_body", "body"):
        child = getattr(stmt, attr, None)
        if child is not None and not isinstance(child, list) and child is not stmt:
            yield from simple_statements(child)


def passthroughs(uop: Uop) -> dict[str, str]:
    inputs = {i.name for i in uop.stack.inputs if i.name != "unused"}
    outputs = {o.name for o in uop.stack.outputs if o.name != "unused"}
    found: dict[str, str] = {}
    for stmt in simple_statements(uop.body):
        tokens = [t for t in stmt.contents if t.kind != "COMMENT"]
        if len(tokens) == 4 and tokens[1].kind == "EQUALS" and tokens[3].kind == "SEMI":
            if tokens[0].kind == "IDENTIFIER" and tokens[2].kind == "IDENTIFIER":
                if tokens[0].text in outputs and tokens[2].text in inputs:
                    found[tokens[0].text] = tokens[2].text
    return found


def constant_outputs(uop: Uop) -> dict[str, str]:
    outputs = {o.name for o in uop.stack.outputs if o.name != "unused"}
    found: dict[str, str] = {}
    for stmt in simple_statements(uop.body):
        tokens = [t for t in stmt.contents if t.kind != "COMMENT"]
        if len(tokens) >= 4 and tokens[1].kind == "EQUALS" and tokens[0].kind == "IDENTIFIER" and tokens[0].text in outputs:
            texts = {t.text for t in tokens[2:]}
            if "PyStackRef_True" in texts and "PyStackRef_False" in texts:
                found[tokens[0].text] = "bool"
    return found


def dead_inputs(uop: Uop) -> list[str]:
    inputs = {i.name for i in uop.stack.inputs if i.name != "unused"}
    dead: list[str] = []
    for stmt in simple_statements(uop.body):
        tokens = [t for t in stmt.contents if t.kind != "COMMENT"]
        if len(tokens) >= 4 and tokens[0].text == "DEAD" and tokens[2].text in inputs:
            dead.append(tokens[2].text)
        if tokens and tokens[0].text == "INPUTS_DEAD":
            dead.append("*")
    return dead


def table() -> dict[str, Any]:
    analysis = analyze_files([str(ROOT / "Python/bytecodes.c")])
    out: dict[str, Any] = {}
    for uop in analysis.uops.values():
        if not uop.is_viable() or uop.properties.tier == 1 or uop.is_super():
            continue
        flags = cflags(uop.properties)
        out[uop.name] = {
            "inputs": [[i.name, i.size] for i in uop.stack.inputs],
            "outputs": [[o.name, o.size] for o in uop.stack.outputs],
            "caches": [[c.name, c.size] for c in uop.caches],
            "flags": [] if flags == "0" else flags.split(" | "),
            "passthrough": passthroughs(uop),
            "dead": dead_inputs(uop),
            "constant_outputs": constant_outputs(uop),
        }
    return out


def main(argv: list[str]) -> int:
    target = Path(argv[0]) if argv else ROOT / "Tools/uop_validator/uopval/semantics/uop_metadata.json"
    data = {"commit": "f2eaf174b2505be1b3dfbe817db243adf7fd5d6b", "source": "Python/bytecodes.c", "uops": table()}
    target.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")
    print(f"{len(data['uops'])} uops -> {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
