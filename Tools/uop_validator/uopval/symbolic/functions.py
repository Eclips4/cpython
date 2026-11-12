import z3

from .sorts import Float64

metatype = z3.Function("metatype", z3.IntSort(), z3.IntSort())
immortal_constant = z3.Function("immortal_constant", z3.IntSort(), z3.BoolSort())
pinned = z3.Function("pinned", z3.IntSort(), z3.BoolSort())
known_length = z3.Function("known_length", z3.IntSort(), z3.IntSort())
has_known_length = z3.Function("has_known_length", z3.IntSort(), z3.BoolSort())
frozen_member = z3.Function("frozen_member", z3.IntSort(), z3.IntSort(), z3.BoolSort())
frozen_members_known = z3.Function("frozen_members_known", z3.IntSort(), z3.BoolSort())
frozen_value = z3.Function("frozen_value", z3.IntSort(), z3.IntSort(), z3.IntSort())
type_version = z3.Function("type_version", z3.IntSort(), z3.IntSort())
type_of_version = z3.Function("type_of_version", z3.IntSort(), z3.IntSort())
iter_kind = z3.Function("iter_kind", z3.IntSort(), z3.IntSort())
iternext_inline = z3.Function("iternext_inline", z3.IntSort(), z3.BoolSort())
dict_subscript = z3.Function("dict_subscript", z3.IntSort(), z3.BoolSort())
dict_store_subscript = z3.Function("dict_store_subscript", z3.IntSort(), z3.BoolSort())
type_vectorcall = z3.Function("type_vectorcall", z3.IntSort(), z3.BoolSort())
type_object_id = z3.Function("type_object_id", z3.IntSort(), z3.IntSort())
func_argcount = z3.Function("func_argcount", z3.IntSort(), z3.IntSort())
code_framesize = z3.Function("code_framesize", z3.IntSort(), z3.IntSort())
code_nlocals = z3.Function("code_nlocals", z3.IntSort(), z3.IntSort())
builtin_object = z3.Function("builtin_object", z3.IntSort(), z3.IntSort())
type_init = z3.Function("type_init", z3.IntSort(), z3.IntSort())
type_getitem = z3.Function("type_getitem", z3.IntSort(), z3.IntSort())
type_getitem_version = z3.Function("type_getitem_version", z3.IntSort(), z3.IntSort())
type_is_heap = z3.Function("type_is_heap", z3.IntSort(), z3.BoolSort())
type_generic_getattr = z3.Function("type_generic_getattr", z3.IntSort(), z3.BoolSort())
special_descr = z3.Function("special_descr", z3.IntSort(), z3.IntSort(), z3.IntSort())
super_descr = z3.Function("super_descr", z3.IntSort(), z3.IntSort(), z3.IntSort(), z3.IntSort())
descr_ml_meth = z3.Function("descr_ml_meth", z3.IntSort(), z3.IntSort())
descr_ml_flags = z3.Function("descr_ml_flags", z3.IntSort(), z3.IntSort())
descr_d_type = z3.Function("descr_d_type", z3.IntSort(), z3.IntSort())
named_type_tag = z3.Function("named_type_tag", z3.IntSort(), z3.IntSort())
im_func = z3.Function("im_func", z3.IntSort(), z3.IntSort())
im_self = z3.Function("im_self", z3.IntSort(), z3.IntSort())
cfunc_flags = z3.Function("cfunc_flags", z3.IntSort(), z3.IntSort())
pow2 = z3.Function("pow2", z3.IntSort(), z3.IntSort())
ipow = z3.Function("ipow", z3.IntSort(), z3.IntSort(), z3.IntSort())
bit_and = z3.Function("bit_and", z3.IntSort(), z3.IntSort(), z3.IntSort())
bit_or = z3.Function("bit_or", z3.IntSort(), z3.IntSort(), z3.IntSort())
bit_xor = z3.Function("bit_xor", z3.IntSort(), z3.IntSort(), z3.IntSort())
int_as_float = z3.Function("int_as_float", z3.IntSort(), Float64)
subtype = z3.Function("subtype", z3.IntSort(), z3.IntSort(), z3.BoolSort())
ITER_NONE = 0
ITER_VIRTUAL = 1
ITER_SELF = 2
ITER_TRAD = 3
ITER_KINDS = {"none": ITER_NONE, "virtual": ITER_VIRTUAL, "self": ITER_SELF, "trad": ITER_TRAD}
BUILTIN_LEN = 1
BUILTIN_ISINSTANCE = 2
BUILTIN_LIST_APPEND = 3

BUILTIN_IDS = {
    "len": BUILTIN_LEN,
    "isinstance": BUILTIN_ISINSTANCE,
    "list_append": BUILTIN_LIST_APPEND,
}
