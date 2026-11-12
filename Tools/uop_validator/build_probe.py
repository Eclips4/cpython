import json
import sys


def make(v):
    return int(str(v))


def compact(v):
    return sys.getsizeof(make(v)) == sys.getsizeof(1)


def small(v):
    return make(v) is make(v)


def range_walk(start, stop, step, n):
    it = iter(range(start, stop, step))
    out = []
    for _ in range(n):
        v = next(it)
        out.append([v, it.__length_hint__()])
    return out


OPS = {"<": lambda a, b: a < b, "<=": lambda a, b: a <= b, "==": lambda a, b: a == b, "!=": lambda a, b: a != b, ">": lambda a, b: a > b, ">=": lambda a, b: a >= b}


def main():
    query = json.load(sys.stdin)
    result = {
        "bits_per_digit": sys.int_info.bits_per_digit,
        "sizeof_one_digit_int": sys.getsizeof(1),
        "version": sys.version,
        "true_id_low": id(True) & 0xFFF,
        "false_id_low": id(False) & 0xFFF,
        "bool_is_int_subclass": issubclass(bool, int),
        "true_is_exact_int": type(True) is int,
        "compact": {str(v): compact(v) for v in query.get("compact", [])},
        "small": {str(v): small(v) for v in query.get("small", [])},
        "add": {f"{a},{b}": make(a) + make(b) for a, b in query.get("add", [])},
        "sub": {f"{a},{b}": make(a) - make(b) for a, b in query.get("sub", [])},
        "mul": {f"{a},{b}": make(a) * make(b) for a, b in query.get("mul", [])},
        "cmp": {f"{a}{op}{b}": OPS[op](make(a), make(b)) for a, op, b in query.get("cmp", [])},
        "range": {f"{s},{e},{st},{n}": range_walk(s, e, st, n) for s, e, st, n in query.get("range", [])},
    }
    json.dump(result, sys.stdout)


if __name__ == "__main__":
    main()
