from parsing import Tensor, SparseIndex, BinaryContraction, NaryContraction, IntermediateResult
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


Int = Tensor("Int", [k, nu, mu])
C = Tensor("C", [i, nu])
X_nofilter = Tensor("X_nofilter", [k, muhat, i])
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
