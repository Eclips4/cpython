from collections.abc import Sequence

import z3

from ..alignment.families import FAMILIES
from .arguments import build_parser
from .commands import cmd_exits, cmd_refine, cmd_show, cmd_stats, cmd_trace


def main(argv: Sequence[str]) -> int:
    args = build_parser().parse_args(argv)
    z3.set_param("memory_max_size", args.memory_mb)
    family = FAMILIES[args.family]
    match args.command:
        case "stats":
            return cmd_stats(args.files, family)
        case "show":
            return cmd_show(args.file, args.seq, family)
        case "trace":
            return cmd_trace(args.file, args.seq, family)
        case "exits":
            return cmd_exits(args.file, args.seq)
        case _:
            return cmd_refine(
                args.files,
                args.timeout_ms,
                family,
                args.only,
                args.unit_budget_s,
                args.until,
                args.lines,
                args.rewrites_first,
            )
