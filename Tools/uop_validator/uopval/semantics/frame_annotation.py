from collections.abc import Sequence
from dataclasses import replace

from ..domain.trace import Uop

FRAME_PUSHES = frozenset({"_PUSH_FRAME", "_FOR_ITER_GEN_FRAME", "_SEND_GEN_FRAME"})


def annotate_frames(uops: Sequence[Uop]) -> list[Uop]:
    out = list(uops)
    for k, uop in enumerate(out):
        if uop.op in (
            "_INIT_CALL_PY_EXACT_ARGS",
            "_PUSH_FRAME",
            "_CREATE_INIT_FRAME",
            "_PY_FRAME_GENERAL",
            "_PY_FRAME_KW",
            "_PY_FRAME_EX",
            "_FOR_ITER_GEN_FRAME",
            "_SEND_GEN_FRAME",
            "_LOAD_ATTR_PROPERTY_FRAME",
            "_BINARY_OP_SUBSCR_INIT_CALL",
        ):
            for later in out[k + 1 : k + 8]:
                if later.op == "_GUARD_CODE_VERSION__PUSH_FRAME":
                    out[k] = replace(out[k], callee_version=later.operand0)
                    break
            for later in out[k + 1 : k + 8]:
                if later.op == "_GUARD_IP__PUSH_FRAME":
                    out[k] = replace(out[k], return_depth=later.operand1)
                    break
        if uop.op == "_CREATE_INIT_FRAME":
            nesting = 0
            seen_return = False
            for later in out[k + 1 :]:
                if later.op in FRAME_PUSHES:
                    nesting += 1
                elif later.op in ("_RETURN_VALUE", "_RETURN_GENERATOR", "_YIELD_VALUE"):
                    nesting -= 1
                    seen_return = nesting == 0
                elif (
                    seen_return
                    and later.op == "_RECORD_CODE"
                    and later.recorded is not None
                    and later.recorded.code is not None
                ):
                    out[k] = replace(out[k], return_code=later.recorded.code)
                    break
        if uop.op in ("_RETURN_VALUE", "_YIELD_VALUE", "_RETURN_GENERATOR"):
            code = None
            depth = None
            for later in out[k + 1 : k + 6]:
                if (
                    later.op == "_RECORD_CODE"
                    and later.recorded is not None
                    and later.recorded.code is not None
                ):
                    code = later.recorded.code
                if later.op in (
                    "_GUARD_IP_RETURN_VALUE",
                    "_GUARD_IP_YIELD_VALUE",
                    "_GUARD_IP_RETURN_GENERATOR",
                ):
                    depth = later.operand1
            out[k] = replace(out[k], return_code=code, return_depth=depth)
    return out


def carry_annotations(
    uops: Sequence[Uop], annotated_input: Sequence[Uop], positions: Sequence[int]
) -> list[Uop]:
    out = list(uops)
    for k, uop in enumerate(out):
        source = annotated_input[positions[k]]
        if source.op == uop.op or uop.op in (
            "_INIT_CALL_PY_EXACT_ARGS",
            "_PUSH_FRAME",
            "_RETURN_VALUE",
            "_YIELD_VALUE",
            "_RETURN_GENERATOR",
            "_CREATE_INIT_FRAME",
            "_PY_FRAME_GENERAL",
            "_PY_FRAME_KW",
            "_PY_FRAME_EX",
            "_FOR_ITER_GEN_FRAME",
            "_SEND_GEN_FRAME",
            "_LOAD_ATTR_PROPERTY_FRAME",
        ):
            out[k] = replace(
                uop,
                callee_version=source.callee_version,
                return_code=source.return_code,
                return_depth=source.return_depth,
            )
    return out
