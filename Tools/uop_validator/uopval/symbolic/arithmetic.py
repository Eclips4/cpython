import z3

from .sorts import RNE, Float64

FP_BUILTINS = {"add": z3.fpAdd, "sub": z3.fpSub, "mul": z3.fpMul, "div": z3.fpDiv}


def fp_arith(name: str, x: z3.FPRef, y: z3.FPRef) -> z3.FPRef:
    folded = z3.simplify(FP_BUILTINS[name](RNE, x, y))
    if z3.is_fp_value(folded):
        return folded
    fn = z3.Function(f"fp_{name}", Float64, Float64, Float64)
    return fn(x, y)


def py_floordiv(a: z3.ArithRef, b: z3.ArithRef) -> z3.ArithRef:
    q = a / b
    return z3.If(z3.Or(b > 0, a % b == 0), q, q - 1)


def py_mod(a: z3.ArithRef, b: z3.ArithRef) -> z3.ArithRef:
    return a - b * py_floordiv(a, b)
