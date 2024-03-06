from parsing import Tensor, SparseIndex, IntermediateResult, BinaryContraction, NaryContraction
from fused_ir import FusedIR

# R(i, a, b, c) += I(i, j, k, l) * M1(j, a) * M2(k, b) * M3(l, c)
i = SparseIndex("i")
j = SparseIndex("j")
k = SparseIndex("k")
a = SparseIndex("a")
b = SparseIndex("b")

res1 = Tensor("R1", [i, a, b])
I = Tensor("I", [i, j, k])
M1 = Tensor("M1", [j, a])
M2 = Tensor("M2", [k, b])
IM2 = IntermediateResult(I, M2, [k])
statements = [BinaryContraction(IM2, I, M2), BinaryContraction(res1, IM2, M1)]
contraction = NaryContraction(res1, [I, M1, M2])
contraction.statements = statements
print(contraction)
gen = contraction.fuse_loops(workspace=False)
fir = FusedIR(gen)
fir.reduce_intermediates()
print(fir)
print(fir.emit_taco_kernel("ttmc_fused"))

