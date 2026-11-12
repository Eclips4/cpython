import hashlib

TRUE_ID = -7
FALSE_ID = -6
NONE_ID = -8
EMPTY_STR_ID = -9
CONSTANT_ID_BASE = -(1 << 20)


def type_tag(name: str) -> int:
    digest = hashlib.sha1(name.encode()).digest()
    return 1 + int.from_bytes(digest[:4], "big") % ((1 << 31) - 1)


INT_TAG = type_tag("int")
BOOL_TAG = type_tag("bool")
NONE_TAG = type_tag("NoneType")
FLOAT_TAG = type_tag("float")
STR_TAG = type_tag("str")
TUPLE_TAG = type_tag("tuple")
RANGE_ITER_TAG = type_tag("range_iterator")
FUNCTION_TAG = type_tag("function")
TYPE_TAG = type_tag("type")
LIST_TAG = type_tag("list")
DICT_TAG = type_tag("dict")
FROZENDICT_TAG = type_tag("frozendict")
SET_TAG = type_tag("set")
FROZENSET_TAG = type_tag("frozenset")
SLICE_TAG = type_tag("slice")
BYTES_TAG = type_tag("bytes")
METHOD_TAG = type_tag("method")
MODULE_TAG = type_tag("module")
CFUNCTION_TAG = type_tag("builtin_function_or_method")
METHOD_DESCRIPTOR_TAG = type_tag("method_descriptor")

METHOD_DESCRIPTOR_FLAG_TAGS = (
    METHOD_DESCRIPTOR_TAG,
    type_tag("wrapper_descriptor"),
    type_tag("functools._lru_cache_wrapper"),
)

MODELLED_TAGS = (
    INT_TAG,
    BOOL_TAG,
    NONE_TAG,
    FLOAT_TAG,
    STR_TAG,
    TUPLE_TAG,
    RANGE_ITER_TAG,
    FUNCTION_TAG,
    TYPE_TAG,
    type_tag("cell"),
)

INSTANCE_TAGS = tuple(
    type_tag(n)
    for n in (
        "list",
        "dict",
        "frozendict",
        "set",
        "frozenset",
        "bytes",
        "slice",
        "method",
        "module",
        "builtin_function_or_method",
        "method_descriptor",
        "generator",
        "coroutine",
        "range",
        "range_iterator",
    )
)

TAG_SHIFT = 1 << 40
