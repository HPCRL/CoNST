from parsing import Tensor, SparseIndex
from fused_ir import FusedIR

# R(i, a, b, c) += I(i, j, k, l) * M1(j, a) * M2(k, b) * M3(l, c)
i = SparseIndex("i", 200)
j = SparseIndex("j", 200)
k = SparseIndex("k", 200)
a = SparseIndex("a", 10)
b = SparseIndex("b", 10)

res1 = Tensor("R1", [i, a, b])
I = Tensor("I", [i, j, k])
M1 = Tensor("M1", [j, a])
M2 = Tensor("M2", [k, b])
#M3 = Tensor("M3", [l, c])


res1 += I * M2 * M1

#print(res1.validate())
#res1.binarize()
##print(res.statements)
#gen = res1.fuse_loops()
#fir = FusedIR(gen)
#fir.reduce_intermediates()
#print(fir)
#print(fir.emit_taco_kernel("ttmc_fused"))
#print(fir)
#print(fir.lower_to_taco())
#print(fir.lower_intermediate_tensors())
#print(fir.lower_og_tensors())



#res2 = Tensor("R2", [a, j, b])
#M1 = Tensor("M1", [i, a])
#M2 = Tensor("M2", [k, b])
#res2 += I * M2 * M1
#res2.binarize()
#gen = res2.fuse_loops()
#fir = FusedIR(gen)
#fir.reduce_intermediates()
#print(fir)
#print(fir.emit_taco_kernel("ttmc_fused2"))
#
res3 = Tensor("R3", [a, b, k])
M1 = Tensor("M1", [i, a])
M2 = Tensor("M2", [j, b])
res3 += I * M2 * M1
res3.binarize()
gen = res3.fuse_loops()
fir = FusedIR(gen)
fir.reduce_intermediates()
print(fir)
print(fir.emit_taco_kernel("ttmc_fused3"))
#
