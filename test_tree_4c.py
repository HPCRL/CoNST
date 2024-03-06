from parsing import Tensor, SparseIndex, BinaryContraction, NaryContraction, IntermediateResult
from fused_ir import FusedIR

k = SparseIndex("k")
l = SparseIndex("l")
mu = SparseIndex("mu")
nu = SparseIndex("nu")
i = SparseIndex("i")
j = SparseIndex("j")
muhat = SparseIndex("muhat")
nuhat = SparseIndex("nuhat")
D = Tensor("D", [k, l])
I1 = Tensor("I1", [i, muhat, k], base_tensor="3c")
I2 = Tensor("I2", [j, nuhat, l], base_tensor="3c")
X = Tensor("Result", [i, j, muhat, nuhat])
IntD = IntermediateResult(I1, D, [k])
statements = [BinaryContraction(IntD, I1, D), BinaryContraction(X, IntD, I2)]
contraction = NaryContraction(X, [I1, D, I2])
contraction.statements = statements
print(contraction)
gen = contraction.fuse_loops()
fir = FusedIR(gen)
fir.reduce_intermediates()
print(fir)
print(fir.emit_taco_kernel("3c_to_4c_fused"))
