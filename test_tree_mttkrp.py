from parsing import TensorRef, SparseIndex, IntermediateResult, BinaryContraction, NaryContraction, BaseTensor
from fused_ir import FusedIR
import time

i = SparseIndex("i")
j = SparseIndex("j")
k = SparseIndex("k")
r = SparseIndex("r")


res = TensorRef("R", [i, r])
I = TensorRef("I", [i, j, k])
M1 = TensorRef("M1", [j, r])
M2 = TensorRef("M2", [k, r])

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
