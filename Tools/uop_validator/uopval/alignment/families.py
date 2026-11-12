from collections.abc import Mapping
from dataclasses import dataclass

Segments = Mapping[str, tuple[tuple[str, ...], ...]]


@dataclass(frozen=True)
class Family:
    name: str
    uops: frozenset[str]
    segments: Segments
    depth_zero: bool
    lower_stack_free: bool


F1_UOPS: frozenset[str] = frozenset(
    {
        "_START_EXECUTOR",
        "_MAKE_WARM",
        "_CHECK_VALIDITY",
        "_SET_IP",
        "_NOP",
        "_JUMP_TO_TOP",
        "_EXIT_TRACE",
        "_DEOPT",
        "_CHECK_PERIODIC",
        "_ITER_CHECK_RANGE",
        "_GUARD_NOT_EXHAUSTED_RANGE",
        "_ITER_NEXT_RANGE",
        "_LOAD_FAST_BORROW",
        "_SWAP_FAST",
        "_POP_TOP",
        "_POP_TOP_INT",
        "_POP_TOP_NOP",
        "_LOAD_SMALL_INT",
        "_LOAD_CONST_INLINE_BORROW",
        "_GUARD_TOS_INT",
        "_GUARD_NOS_INT",
        "_GUARD_TOS_OVERFLOWED",
        "_GUARD_NOS_OVERFLOWED",
        "_BINARY_OP_ADD_INT",
    }
)

F1_SEGMENTS: dict[str, tuple[tuple[str, ...], ...]] = {
    "_GUARD_TOS_INT": (("_GUARD_TOS_OVERFLOWED",), ("_NOP",)),
    "_GUARD_NOS_INT": (("_GUARD_NOS_OVERFLOWED",), ("_NOP",)),
    "_ITER_CHECK_RANGE": (("_NOP",),),
    "_POP_TOP": (("_POP_TOP_INT",), ("_POP_TOP_NOP",), ("_NOP",)),
    "_POP_TOP_INT": (("_POP_TOP_NOP",), ("_NOP",)),
    "_LOAD_SMALL_INT": (("_LOAD_CONST_INLINE_BORROW",),),
    "_SET_IP": (("_NOP",),),
    "_CHECK_VALIDITY": (("_NOP",),),
    "_LOAD_FAST_BORROW": (("_NOP",),),
}

FOLD: tuple[str, ...] = ("_LOAD_CONST_INLINE_BORROW", "_RROT_3")

BRANCH_GUARD_SEGMENTS: tuple[tuple[str, ...], ...] = (
    ("_GUARD_BIT_IS_SET_POP",),
    ("_GUARD_BIT_IS_UNSET_POP",),
    ("_POP_TOP",),
)

F2_UOPS: frozenset[str] = F1_UOPS | frozenset(
    {
        "_LOAD_FAST",
        "_RROT_3",
        "_BINARY_OP_SUBTRACT_INT",
        "_BINARY_OP_MULTIPLY_INT",
        "_COMPARE_OP_INT",
        "_TO_BOOL_BOOL",
        "_GUARD_IS_TRUE_POP",
        "_GUARD_IS_FALSE_POP",
        "_GUARD_BIT_IS_SET_POP",
        "_GUARD_BIT_IS_UNSET_POP",
    }
)

F2_SEGMENTS: dict[str, tuple[tuple[str, ...], ...]] = {
    **F1_SEGMENTS,
    "_BINARY_OP_ADD_INT": (FOLD,),
    "_BINARY_OP_SUBTRACT_INT": (FOLD,),
    "_BINARY_OP_MULTIPLY_INT": (FOLD,),
    "_COMPARE_OP_INT": (FOLD,),
    "_TO_BOOL_BOOL": (("_NOP",),),
    "_GUARD_IS_TRUE_POP": BRANCH_GUARD_SEGMENTS,
    "_GUARD_IS_FALSE_POP": BRANCH_GUARD_SEGMENTS,
}

F1 = Family("F1", F1_UOPS, F1_SEGMENTS, False, False)
F2 = Family("F2", F2_UOPS, F2_SEGMENTS, True, True)
F3 = Family("F3", frozenset(), {}, True, True)
FAMILIES: dict[str, Family] = {"F1": F1, "F2": F2, "F3": F3}


def is_open(family: Family) -> bool:
    return family.name == "F3"
