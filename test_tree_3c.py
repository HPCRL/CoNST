from parsing import Tensor, SparseIndex, BinaryContraction, NaryContraction, IntermediateResult
from fused_ir import FusedIR, get_includes

k = SparseIndex("k", 100)
mu = SparseIndex("mu", 100)
nu = SparseIndex("nu", 100)
i = SparseIndex("i", 100)
muhat = SparseIndex("muhat", 100)

Int = Tensor("Int", [mu, nu, k])
C = Tensor("C", [nu, i])
Phat = Tensor("Phat", [mu, muhat])
L = Tensor("L", [k, i])
X = Tensor("X", [k, i, muhat])
IntC = IntermediateResult(Int, C, [nu])
IntCPhat = IntermediateResult(IntC, Phat, [mu])
statements = [BinaryContraction(IntC, Int, C), BinaryContraction(
    IntCPhat, IntC, Phat), BinaryContraction(X, IntCPhat, L)]
contraction = NaryContraction(X, [Int, C, Phat, L])
contraction.statements = statements
print(contraction)
gen = contraction.fuse_loops()
fir = FusedIR(gen)
fir.reduce_intermediates()
print(fir)
print(fir.emit_taco_kernel("filter_fused"))
with open("3c_filter_fused.hpp", "w") as f:
    f.write(get_includes())
    f.write(fir.emit_taco_kernel("filter_fused"))


X_nofilter = Tensor("X_nofilter", [k, i, muhat])
IntC = IntermediateResult(Int, C, [nu])
statements = [BinaryContraction(IntC, Int, C),
              BinaryContraction(X_nofilter, IntC, Phat)]
contraction = NaryContraction(X_nofilter, [Int, C, Phat])
contraction.statements = statements
print(contraction)
gen = contraction.fuse_loops()
fir = FusedIR(gen)
fir.reduce_intermediates()
print(fir)
print(fir.emit_taco_kernel("nofilter_fused"))
with open("3c_nofilter_fused.hpp", "w") as f:
    f.write(get_includes())
    f.write(fir.emit_taco_kernel("nofilter_fused"))
