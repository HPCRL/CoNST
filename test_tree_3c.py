from parsing import TensorRef, SparseIndex, BinaryContraction, NaryContraction, IntermediateResult
from fused_ir import FusedIR, get_includes

k = SparseIndex("k", 100)
mu = SparseIndex("mu", 100)
nu = SparseIndex("nu", 100)
i = SparseIndex("i", 100)
muhat = SparseIndex("muhat", 100)

Int = TensorRef("Int", [mu, nu, k])
C = TensorRef("C", [nu, i])
Phat = TensorRef("Phat", [mu, muhat])
L = TensorRef("L", [k, i])
X = TensorRef("X", [k, i, muhat])
IntC = IntermediateResult(Int, C, [nu], const_shape="PAO")
IntCPhat = IntermediateResult(IntC, Phat, [mu], const_shape="PAO")
statements = [BinaryContraction(IntC, Int, C), BinaryContraction(
    IntCPhat, IntC, Phat), BinaryContraction(X, IntCPhat, L)]
contraction = NaryContraction(X, [Int, C, Phat, L])
contraction.statements = statements
print(contraction)
gen = contraction.fuse_loops()
fir = FusedIR(gen)
fir.reduce_intermediates()
print(fir)
print(fir.emit_taco_kernel("filter_const"))
with open("3c_filter_fused.hpp", "w") as f:
    f.write(get_includes())
    f.write(fir.emit_taco_kernel("filter_const"))


X_nofilter = TensorRef("X_nofilter", [k, i, muhat])
IntC = IntermediateResult(Int, C, [nu], const_shape="PAO")
statements = [BinaryContraction(IntC, Int, C),
              BinaryContraction(X_nofilter, IntC, Phat)]
contraction = NaryContraction(X_nofilter, [Int, C, Phat])
contraction.statements = statements
print(contraction)
gen = contraction.fuse_loops()
fir = FusedIR(gen)
fir.reduce_intermediates()
print(fir)
print(fir.emit_taco_kernel("nofilter_const"))
with open("3c_nofilter_fused.hpp", "w") as f:
    f.write(get_includes())
    f.write(fir.emit_taco_kernel("nofilter_const"))
