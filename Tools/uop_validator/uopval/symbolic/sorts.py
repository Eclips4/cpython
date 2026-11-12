import z3

SlotDT = z3.Datatype("Slot")
SlotDT.declare("Null")
SlotDT.declare("TaggedInt", ("k", z3.IntSort()))
SlotDT.declare("Ref", ("id", z3.IntSort()), ("borrowed", z3.BoolSort()))

Slot = SlotDT.create()

Float64 = z3.FPSort(11, 53)
ItemsSort = z3.ArraySort(z3.IntSort(), Slot)

ObjDT = z3.Datatype("Obj")
ObjDT.declare("ExactInt", ("val", z3.IntSort()))
ObjDT.declare("RangeIter", ("start", z3.IntSort()), ("step", z3.IntSort()), ("len", z3.IntSort()))
ObjDT.declare("Bool", ("b", z3.BoolSort()))
ObjDT.declare("NoneObj")
ObjDT.declare("Float", ("fval", Float64))
ObjDT.declare("Str", ("sval", z3.StringSort()))
ObjDT.declare("Tuple", ("tlen", z3.IntSort()), ("titems", ItemsSort))
ObjDT.declare("Func", ("version", z3.IntSort()), ("code", z3.IntSort()))
ObjDT.declare("TypeObj", ("ttag", z3.IntSort()))
ObjDT.declare("Cell", ("content", Slot))
ObjDT.declare("Other", ("tag", z3.IntSort()))

Obj = ObjDT.create()

HeapSort = z3.ArraySort(z3.IntSort(), Obj)
LedgerSort = z3.ArraySort(z3.IntSort(), z3.IntSort())
RNE = z3.RNE()
