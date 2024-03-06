from parsing import Tensor, SparseIndex
from fused_ir import FusedIR

k = SparseIndex("k", 175)
l = SparseIndex("l", 175)
mu = SparseIndex("mu", 75)
nu = SparseIndex("nu", 75)
i = SparseIndex("i", 25)
j = SparseIndex("j", 25)
muhat = SparseIndex("muhat", 75)
nuhat = SparseIndex("nuhat", 75)
D = Tensor("D", [k, l])
I1 = Tensor("I1", [i, muhat, k], base_tensor="3c")
I2 = Tensor("I2", [j, nuhat, l], base_tensor="3c")
X = Tensor("Result", [i, j, muhat, nuhat])
X += I1 * D * I2

print(X.validate())
X.binarize()
print(X.statements)
gen = X.fuse_loops()
fir = FusedIR(gen)
fir.reduce_intermediates()
print(fir)
print(fir.emit_taco_kernel("3c_to_4c_fused"))
