from parsing import Tensor, SparseIndex, IntermediateResult, BinaryContraction, NaryContraction
from fused_ir import FusedIR
import time

i = SparseIndex("i", 500)
j = SparseIndex("j", 500)
k = SparseIndex("k", 500)
r = SparseIndex("r", 50)

res = Tensor("R", [i, r])
I = Tensor("I", [i, j, k])
M1 = Tensor("M1", [j, r])
M2 = Tensor("M2", [k, r])

#res += I * M2 * M1
IM2 = IntermediateResult(I, M2, [k])
statements = [BinaryContraction(IM2, I, M2), BinaryContraction(res, IM2, M1)]
contraction = NaryContraction(res, [I, M1, M2])
contraction.statements = statements
print(contraction)
gen = contraction.fuse_loops(workspace=False)
start = time.time()
fir = FusedIR(gen)
fir.reduce_intermediates()
print(fir)
print(fir.emit_taco_kernel("mttkrp_manbin_fused"))
end = time.time()
print("Time taken to lower solver output to TACO: ", end-start)

#print(res.validate())
#res.binarize()
#gen = res.fuse_loops()
#fir = FusedIR(gen)
#fir.reduce_intermediates()
#print(fir)
#print(fir.emit_taco_kernel("mttkrp_fused"))
