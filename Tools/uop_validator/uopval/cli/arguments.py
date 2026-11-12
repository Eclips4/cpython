import argparse

from ..alignment.families import FAMILIES


def seq_set(text: str) -> set[int]:
    return {int(x) for x in text.split(",")}


def line_range(text: str) -> tuple[int, int]:
    first, last = text.split(":")
    return int(first), int(last)


def unit_budget(text: str) -> float | None:
    seconds = float(text)
    return seconds if seconds > 0 else None


def common_options(with_defaults: bool) -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--family",
        choices=sorted(FAMILIES),
        default="F3" if with_defaults else argparse.SUPPRESS,
        help="uop family (default: F3)",
    )
    common.add_argument(
        "--memory-mb",
        type=int,
        default=3500 if with_defaults else argparse.SUPPRESS,
        metavar="MB",
        help="z3 memory cap per process (default: 3500)",
    )
    return common


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m uopval", parents=[common_options(True)])
    common = common_options(False)
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    stats = commands.add_parser(
        "stats", parents=[common], help="classify the units of capture files"
    )
    stats.add_argument("files", nargs="+", metavar="FILE")

    for name, summary in (
        ("show", "show one unit's input and output side by side"),
        ("trace", "symbolically execute one unit's input trace"),
        ("exits", "check which exits of one unit's input trace are feasible"),
    ):
        one = commands.add_parser(name, parents=[common], help=summary)
        one.add_argument("file", metavar="FILE")
        one.add_argument("seq", type=int, metavar="SEQ")

    check = commands.add_parser(
        "refine", parents=[common], help="validate units against the refinement obligations"
    )
    check.add_argument("files", nargs="+", metavar="FILE")
    check.add_argument(
        "--timeout-ms",
        type=int,
        default=10000,
        metavar="N",
        help="solver timeout per obligation (default: 10000)",
    )
    check.add_argument(
        "--unit-budget-s",
        type=unit_budget,
        default=120.0,
        metavar="S",
        help="time budget per unit, 0 for none (default: 120)",
    )
    check.add_argument(
        "--until", type=int, metavar="POSITION", help="skip obligations after this position"
    )
    check.add_argument("--only", type=seq_set, metavar="SEQ,...", help="only these record seqs")
    check.add_argument(
        "--lines",
        type=line_range,
        metavar="FIRST:LAST",
        help="only records on lines FIRST <= line < LAST",
    )
    check.add_argument(
        "--rewrites-first", action="store_true", help="check rewritten positions before the others"
    )
    return parser
