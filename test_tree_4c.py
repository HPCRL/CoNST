from parsing import Tensor, SparseIndex, BinaryContraction, NaryContraction, IntermediateResult
from fused_ir import FusedIR, get_includes

k = SparseIndex("k")
l = SparseIndex("l")
mu = SparseIndex("mu")
nu = SparseIndex("nu")
i = SparseIndex("i")
j = SparseIndex("j")
muhat = SparseIndex("muhat")
nuhat = SparseIndex("nuhat")
D = Tensor("D", [k, l])
I1 = Tensor("I1", [k, i, muhat], base_tensor="3c")
I2 = Tensor("I2", [l, j, nuhat], base_tensor="3c")
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

with open("4c_fused.hpp", "w") as f:
    f.write(get_includes())
    f.write(fir.emit_taco_kernel("fused_3c_to_4c"))
