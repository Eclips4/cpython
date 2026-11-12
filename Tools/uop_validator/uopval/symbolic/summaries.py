import z3

from ..domain.trace import Constant
from .functions import (
    BUILTIN_IDS,
    ITER_KINDS,
    builtin_object,
    cfunc_flags,
    code_framesize,
    code_nlocals,
    descr_d_type,
    descr_ml_flags,
    descr_ml_meth,
    dict_store_subscript,
    dict_subscript,
    frozen_member,
    frozen_members_known,
    frozen_value,
    func_argcount,
    has_known_length,
    im_func,
    im_self,
    immortal_constant,
    iter_kind,
    iternext_inline,
    known_length,
    metatype,
    named_type_tag,
    pinned,
    type_generic_getattr,
    type_getitem,
    type_getitem_version,
    type_init,
    type_is_heap,
    type_object_id,
    type_of_version,
    type_vectorcall,
    type_version,
)
from .identities import CONSTANT_ID_BASE, EMPTY_STR_ID, FALSE_ID, NONE_ID, TRUE_ID, type_tag
from .sorts import Float64, ItemsSort, Obj, Slot
from .state import State, with_constraint
from .types import sanitized_metatype


def type_tag_of_summary(summary: Constant) -> int:
    if summary.type_name is None:
        raise ValueError("not a type summary")
    if summary.immutable or summary.addr is None:
        return type_tag(summary.type_name)
    return -(summary.addr >> 4)


def constant_id(summary: Constant) -> int:
    if (
        summary.type == "int"
        and summary.int_value is not None
        and summary.immortal
        and -5 <= summary.int_value <= 1024
    ):
        return summary.int_value
    if summary.value == "True":
        return TRUE_ID
    if summary.value == "False":
        return FALSE_ID
    if summary.value == "None":
        return NONE_ID
    if summary.type == "str" and (summary.str_value == "" or summary.str_length == 0):
        return EMPTY_STR_ID
    if summary.addr is None:
        raise ValueError(f"constant of type {summary.type} has no address")
    return CONSTANT_ID_BASE - (summary.addr >> 4)


def item_slot(summary: Constant) -> z3.DatatypeRef:
    return Slot.Ref(z3.IntVal(constant_id(summary)), z3.BoolVal(True))


def big_int_value(summary: Constant) -> z3.ArithRef:
    assert summary.addr is not None
    return z3.Int(f"big_int_{summary.addr:x}")


def big_int_facts(summary: Constant) -> list[z3.BoolRef]:
    if (
        summary.type != "int"
        or summary.int_value is not None
        or not summary.overflow
        or summary.addr is None
    ):
        return []
    v = big_int_value(summary)
    return [z3.Or(v > z3.IntVal(2**63 - 1), v < z3.IntVal(-(2**63)))]


def constant_object(summary: Constant) -> z3.DatatypeRef | None:
    if summary.type == "int":
        if summary.int_value is not None:
            return Obj.ExactInt(z3.IntVal(summary.int_value))
        if summary.overflow and summary.addr is not None:
            return Obj.ExactInt(big_int_value(summary))
        return None
    if summary.type == "bool":
        return Obj.Bool(z3.BoolVal(summary.value == "True"))
    if summary.type == "NoneType":
        return Obj.NoneObj
    if summary.type == "float" and summary.float_hex is not None:
        return Obj.Float(z3.FPVal(float.fromhex(summary.float_hex), Float64))
    if summary.type == "str":
        if summary.str_value is not None:
            return Obj.Str(z3.StringVal(summary.str_value))
        return None
    if summary.type == "tuple" and summary.length is not None:
        items: z3.ArrayRef = z3.K(z3.IntSort(), Slot.Null)
        if (
            summary.items is not None
            and not summary.items_truncated
            and len(summary.items) == summary.length
        ):
            for k, item in enumerate(summary.items):
                try:
                    items = z3.Store(items, z3.IntVal(k), item_slot(item))
                except ValueError:
                    return Obj.Tuple(
                        z3.IntVal(summary.length),
                        z3.Const(f"tuple_items_{summary.addr}", ItemsSort),
                    )
            return Obj.Tuple(z3.IntVal(summary.length), items)
        return Obj.Tuple(
            z3.IntVal(summary.length), z3.Const(f"tuple_items_{summary.addr}", ItemsSort)
        )
    if summary.type == "function" and summary.func_version is not None:
        code_id = summary.code.co_version if summary.code is not None else 0
        return Obj.Func(z3.IntVal(summary.func_version), z3.IntVal(code_id))
    if summary.type_name is not None:
        return Obj.TypeObj(z3.IntVal(type_tag_of_summary(summary)))
    return Obj.Other(z3.IntVal(object_type_tag(summary)))


def object_type_tag(summary: Constant) -> int:
    if summary.type_immutable is False and summary.type_addr is not None:
        return -(summary.type_addr >> 4)
    return type_tag(summary.type)


def identity_facts(
    state: State, summary: Constant, ident: z3.ArithRef, held: bool
) -> list[z3.BoolRef]:
    obj = constant_object(summary)
    facts: list[z3.BoolRef] = [immortal_constant(ident) == z3.BoolVal(summary.immortal)]
    if held:
        facts.append(pinned(ident))
    if obj is not None:
        facts.append(z3.Select(state.heap, ident) == obj)
    return facts + big_int_facts(summary)


def own_type_version_facts(summary: Constant) -> list[z3.BoolRef]:
    if summary.type_version is None or summary.type_version == 0:
        return []
    facts: list[z3.BoolRef] = []
    own_tag: z3.ArithRef = z3.IntVal(object_type_tag(summary))
    if summary.type_name is not None:
        own_tag = sanitized_metatype(z3.IntVal(object_type_tag(summary)))
        facts.append(
            metatype(z3.IntVal(type_tag_of_summary(summary))) == z3.IntVal(object_type_tag(summary))
        )
    facts.append(type_version(own_tag) == z3.IntVal(summary.type_version))
    return facts


def registry_facts(summary: Constant, ident: z3.ArithRef) -> list[z3.BoolRef]:
    facts: list[z3.BoolRef] = []
    if summary.type_name is not None:
        facts.append(type_object_id(z3.IntVal(type_tag_of_summary(summary))) == ident)
    if summary.callable_cache in BUILTIN_IDS:
        facts.append(builtin_object(z3.IntVal(BUILTIN_IDS[summary.callable_cache])) == ident)
    return facts


def callable_facts(summary: Constant, ident: z3.ArithRef) -> list[z3.BoolRef]:
    facts: list[z3.BoolRef] = []
    if summary.type == "function" and summary.code is not None:
        version = z3.IntVal(summary.code.co_version)
        facts.append(func_argcount(version) == summary.code.argcount)
        facts.append(code_framesize(version) == summary.code.framesize)
        facts.append(code_nlocals(version) == summary.code.nlocalsplus)
    if summary.ml_flags is not None and summary.type == "builtin_function_or_method":
        facts.append(cfunc_flags(ident) == summary.ml_flags)
    return facts


def known_size(summary: Constant) -> int | None:
    if summary.type == "bytes":
        return summary.bytes_length
    if summary.type == "str":
        return summary.str_length
    if summary.type in ("frozendict", "frozenset", "tuple"):
        return summary.length
    return None


def length_facts(summary: Constant, ident: z3.ArithRef) -> list[z3.BoolRef]:
    size = known_size(summary)
    if size is None:
        return []
    return [has_known_length(ident), known_length(ident) == size]


def frozenset_facts(summary: Constant, ident: z3.ArithRef) -> list[z3.BoolRef]:
    if (
        summary.type != "frozenset"
        or summary.items is None
        or summary.length is None
        or len(summary.items) != summary.length
    ):
        return []
    known = frozen_members_known(ident)
    try:
        members = [constant_id(item) for item in summary.items]
    except ValueError:
        return []
    return [known] + [frozen_member(ident, z3.IntVal(member)) for member in members]


def frozendict_facts(summary: Constant, ident: z3.ArithRef) -> list[z3.BoolRef]:
    facts: list[z3.BoolRef] = []
    if (
        summary.type != "frozendict"
        or summary.pairs is None
        or summary.length is None
        or len(summary.pairs) != summary.length
    ):
        return facts
    for key, value in summary.pairs:
        try:
            key_id, value_id = constant_id(key), constant_id(value)
        except ValueError:
            return facts
        facts.append(frozen_member(ident, z3.IntVal(key_id)))
        facts.append(frozen_value(ident, z3.IntVal(key_id)) == z3.IntVal(value_id))
    facts.append(frozen_members_known(ident))
    return facts


def method_descriptor_facts(summary: Constant, ident: z3.ArithRef) -> list[z3.BoolRef]:
    facts: list[z3.BoolRef] = []
    if summary.type != "method_descriptor":
        return facts
    if summary.ml_meth is not None:
        facts.append(descr_ml_meth(ident) == z3.IntVal(summary.ml_meth))
    if summary.ml_flags is not None:
        facts.append(descr_ml_flags(ident) == z3.IntVal(summary.ml_flags))
    if summary.d_type is not None:
        facts.append(descr_d_type(ident) == named_type_tag(z3.IntVal(type_tag(summary.d_type))))
    return facts


def tuple_item_facts(state: State, summary: Constant, held: bool) -> State:
    if summary.type != "tuple" or summary.items is None:
        return state
    for item in summary.items:
        try:
            constant_id(item)
        except ValueError:
            continue
        _, state = constant_facts(state, item, held)
    return state


def attribute_facts(state: State, summary: Constant, ident: z3.ArithRef) -> State:
    if summary.type == "method":
        for attr, function in ((summary.im_func, im_func), (summary.im_self, im_self)):
            if attr is None or (attr.addr is None and attr.value is None):
                continue
            try:
                inner, state = constant_facts(state, attr)
            except ValueError:
                continue
            state = with_constraint(state, function(ident) == inner)
    if summary.type_name is not None:
        for attr, function in ((summary.init, type_init), (summary.getitem, type_getitem)):
            if attr is None:
                continue
            try:
                inner, state = constant_facts(state, attr)
                state = with_constraint(
                    state, function(z3.IntVal(type_tag_of_summary(summary))) == inner
                )
            except ValueError:
                pass
    return state


def constant_facts(state: State, summary: Constant, held: bool = True) -> tuple[z3.ArithRef, State]:
    ident = z3.IntVal(constant_id(summary))
    facts = identity_facts(state, summary, ident, held)
    facts += type_facts(summary)
    facts += own_type_version_facts(summary)
    facts += registry_facts(summary, ident)
    facts += callable_facts(summary, ident)
    facts += length_facts(summary, ident)
    facts += frozenset_facts(summary, ident)
    facts += frozendict_facts(summary, ident)
    state = tuple_item_facts(state, summary, held)
    facts += method_descriptor_facts(summary, ident)
    state = with_constraint(state, *facts)
    return ident, attribute_facts(state, summary, ident)


def type_facts(summary: Constant) -> list[z3.BoolRef]:
    facts: list[z3.BoolRef] = []
    if summary.type_name is None:
        return facts
    tag = z3.IntVal(type_tag_of_summary(summary))
    if summary.version_tag is not None:
        facts.append(type_version(tag) == summary.version_tag)
        if summary.version_tag != 0:
            facts.append(type_of_version(z3.IntVal(summary.version_tag)) == tag)
    if summary.iter_kind in ITER_KINDS:
        facts.append(iter_kind(tag) == ITER_KINDS[summary.iter_kind])
    if summary.iternext_inline is not None:
        facts.append(iternext_inline(tag) == z3.BoolVal(summary.iternext_inline))
    if summary.dict_subscript is not None:
        facts.append(dict_subscript(tag) == z3.BoolVal(summary.dict_subscript))
    if summary.dict_store_subscript is not None:
        facts.append(dict_store_subscript(tag) == z3.BoolVal(summary.dict_store_subscript))
    if summary.vectorcall is not None:
        facts.append(type_vectorcall(tag) == z3.BoolVal(summary.vectorcall))
    if summary.heap_type is not None:
        facts.append(type_is_heap(tag) == z3.BoolVal(summary.heap_type))
    if summary.generic_getattr is not None:
        facts.append(type_generic_getattr(tag) == z3.BoolVal(summary.generic_getattr))
    if summary.getitem_version is not None:
        facts.append(type_getitem_version(tag) == summary.getitem_version)
    return facts
