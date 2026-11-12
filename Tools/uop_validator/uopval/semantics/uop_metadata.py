import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path

TABLE_PATH = Path(__file__).with_name("uop_metadata.json")


@dataclass(frozen=True)
class StackItem:
    name: str
    size: str

    def count(self, oparg: int) -> int:
        if self.size == "":
            return 1
        return int(eval(self.size, {"__builtins__": {}}, {"oparg": oparg}))


@dataclass(frozen=True)
class UopInfo:
    name: str
    inputs: tuple[StackItem, ...]
    outputs: tuple[StackItem, ...]
    caches: tuple[tuple[str, int], ...]
    flags: frozenset[str]
    passthrough: dict[str, str]
    dead: tuple[str, ...]
    constant_outputs: dict[str, str]

    def popped(self, oparg: int) -> int:
        return sum(item.count(oparg) for item in self.inputs)

    def pushed(self, oparg: int) -> int:
        return sum(item.count(oparg) for item in self.outputs)

    def has(self, flag: str) -> bool:
        return f"HAS_{flag}_FLAG" in self.flags

    def input_offsets(self, oparg: int) -> dict[str, tuple[int, int]]:
        offsets: dict[str, tuple[int, int]] = {}
        depth = 0
        for item in self.inputs:
            n = item.count(oparg)
            offsets[item.name] = (depth, n)
            depth += n
        return offsets

    def output_offsets(self, oparg: int) -> dict[str, tuple[int, int]]:
        offsets: dict[str, tuple[int, int]] = {}
        depth = 0
        for item in self.outputs:
            n = item.count(oparg)
            offsets[item.name] = (depth, n)
            depth += n
        return offsets


@cache
def table() -> dict[str, UopInfo]:
    data = json.loads(TABLE_PATH.read_text())
    out: dict[str, UopInfo] = {}
    for name, d in data["uops"].items():
        out[name] = UopInfo(
            name=name,
            inputs=tuple(StackItem(n, s) for n, s in d["inputs"]),
            outputs=tuple(StackItem(n, s) for n, s in d["outputs"]),
            caches=tuple((n, s) for n, s in d["caches"]),
            flags=frozenset(d["flags"]),
            passthrough=dict(d["passthrough"]),
            dead=tuple(d["dead"]),
            constant_outputs=dict(d.get("constant_outputs", {})),
        )
    return out


def info(op: str) -> UopInfo | None:
    return table().get(op)
