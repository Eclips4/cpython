import json
import sys
from collections.abc import Sequence


def main(argv: Sequence[str]) -> int:
    if len(argv) < 4:
        sys.stderr.write("usage: python mutate_record.py FILE SEQ POSITION NEW_OP [NEW_OPARG] > mutated.jsonl\n")
        return 2
    path, seq, position, op = argv[0], int(argv[1]), int(argv[2]), argv[3]
    oparg = int(argv[4]) if len(argv) > 4 else None
    with open(path) as f:
        for line in f:
            record = json.loads(line)
            if record.get("seq") != seq:
                continue
            uop = record["output"][position]
            sys.stderr.write(f"{record['code']['qualname']}: output[{position}] {uop['op']}({uop['oparg']}) -> {op}({oparg if oparg is not None else uop['oparg']})\n")
            uop["op"] = op
            if oparg is not None:
                uop["oparg"] = oparg
            sys.stdout.write(json.dumps(record) + "\n")
            return 0
    sys.stderr.write("no record with that seq\n")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
