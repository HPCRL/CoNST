from parsing import Tensor, SparseIndex
from fused_ir import FusedIR

i = SparseIndex("i", 500)
j = SparseIndex("j", 500)
k = SparseIndex("k", 500)
r = SparseIndex("r", 50)

res = Tensor("R", [i, r])
I = Tensor("I", [i, j, k])
M1 = Tensor("M1", [j, r])
M2 = Tensor("M2", [k, r])

res += I * M2 * M1

print(res.validate())
res.binarize()
gen = res.fuse_loops()
fir = FusedIR(gen)
fir.reduce_intermediates()
print(fir)
print(fir.emit_taco_kernel("mttkrp_fused"))
