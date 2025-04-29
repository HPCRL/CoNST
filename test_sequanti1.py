from parsing import Tensor, SparseIndex, BinaryContraction, NaryContraction, IntermediateResult
from fused_ir import FusedIR, get_includes

m1 = SparseIndex("m1")
m2 = SparseIndex("m2")
i1 = SparseIndex("i1")
i2 = SparseIndex("i2")
k1 = SparseIndex("k1")

result = Tensor("Ires", [i2, i1])
g = Tensor("g", [m1, m2, k1])
g1 = Tensor("g1", [i1, i2, k1])
c0 = Tensor("c0", [i1, m1])
c1 = Tensor("c1", [i2, m2])

temp0 = IntermediateResult(g, c0, [m1])
temp1 = IntermediateResult(temp0, c1, [m2])


statements = [BinaryContraction(temp0, g, c0), BinaryContraction(temp1, temp0, c1), BinaryContraction(result, g1, temp1)]
contraction = NaryContraction(result, [g, g1, c0, c1])
contraction.statements = statements
print(contraction)
gen = contraction.fuse_loops()
fir = FusedIR(gen)
fir.reduce_intermediates()
print(fir)
print(fir.emit_taco_kernel("balanced_tree_fused"))
with open("where_test_fused.hpp", "w") as f:
    f.write(get_includes())
    f.write(fir.emit_taco_kernel("fused"))

