from parsing import TensorRef, SparseIndex, IntermediateResult, BinaryContraction, NaryContraction
from fused_ir import FusedIR
import time

# R(i, a, b, c) += I(i, j, k, l) * M1(j, a) * M2(k, b) * M3(l, c)
i = SparseIndex("i")
j = SparseIndex("j")
k = SparseIndex("k")
a = SparseIndex("a")
b = SparseIndex("b")

res1 = TensorRef("R1", [i, a, b])
I = TensorRef("I", [i, j, k])
M1 = TensorRef("M1", [j, a])
M2 = TensorRef("M2", [k, b])
IM2 = IntermediateResult(I, M2, [k])
statements = [BinaryContraction(res1, IM2, M1), BinaryContraction(IM2, I, M2)]
contraction = NaryContraction(res1, [I, M1, M2])
contraction.statements = statements
#print(contraction)
gen = contraction.fuse_loops()
fir = FusedIR(gen)
#fir.reduce_intermediates()
#print(fir)
#print(fir.emit_taco_kernel("ttmc_fused"))
