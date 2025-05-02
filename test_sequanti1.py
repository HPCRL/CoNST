from parsing import TensorRef, SparseIndex, BinaryContraction, NaryContraction, IntermediateResult, BaseTensor
from fused_ir import FusedIR, get_includes

m1 = SparseIndex("m1")
m2 = SparseIndex("m2")
i1 = SparseIndex("i1")
i2 = SparseIndex("i2")
k1 = SparseIndex("k1")

g_tensor = BaseTensor("g")
c_tensor = BaseTensor("c")
result = TensorRef("Ires", [i2, i1])
g = TensorRef("g", [m1, m2, k1], base_tensor=g_tensor)
g1 = TensorRef("g1", [i1, i2, k1], base_tensor=g_tensor)
c0 = TensorRef("c0", [i1, m1], base_tensor=c_tensor)
c1 = TensorRef("c1", [i2, m2], base_tensor=c_tensor)

temp0 = IntermediateResult(g, c0, [m1])
temp1 = IntermediateResult(temp0, c1, [m2])


statements = [BinaryContraction(temp0, g, c0), BinaryContraction(temp1, temp0, c1), BinaryContraction(result, g1, temp1)]
contraction = NaryContraction(result, [g, g1, c0, c1])
contraction.statements = statements
gen = contraction.fuse_loops()
fir = FusedIR(gen)
#print(fir)
