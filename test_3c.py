from parsing import Tensor, SparseIndex
from fused_ir import FusedIR

k = SparseIndex("k", 100)
mu = SparseIndex("mu", 100)
nu = SparseIndex("nu", 100)
i = SparseIndex("i", 100)
muhat = SparseIndex("muhat", 100)
Int = Tensor("Int", [k, mu, nu])
C = Tensor("C", [nu, i])
Phat = Tensor("Phat", [mu, muhat])
L = Tensor("L", [k, i])
X = Tensor("X", [k, muhat, i])
X += Int * C * Phat * L

print(X.validate())
X.binarize()
print(X.statements)
gen = X.fuse_loops()
fir = FusedIR(gen)
fir.reduce_intermediates()
print(fir)
print(fir.emit_taco_kernel("filter_fused"))

Int = Tensor("Int", [k, nu, mu])
C = Tensor("C", [i, nu])
X_nofilter = Tensor("X_nofilter", [k, muhat, i])
X_nofilter += Int * C * Phat
print(X_nofilter.validate())
X_nofilter.binarize()
print(X_nofilter.statements)
gen = X_nofilter.fuse_loops()
fir = FusedIR(gen)
fir.reduce_intermediates()
print(fir)
print(fir.emit_taco_kernel("nofilter_fused"))
