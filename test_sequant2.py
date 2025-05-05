from parsing import TensorRef, SparseIndex, BinaryContraction, NaryContraction, IntermediateResult, BaseTensor
from fused_ir import FusedIR

m2 = SparseIndex("m2")
i1 = SparseIndex("i1")
i2 = SparseIndex("i2")
i3 = SparseIndex("i3")
i4 = SparseIndex("i4")
k1 = SparseIndex("k1")

g_tensor = BaseTensor("g")
c_tensor = BaseTensor("c")
i_tensor = BaseTensor("I")
result = TensorRef("result", [i1, i2, i3, i4])
g = TensorRef("g", [i3, m2, k1], base_tensor=g_tensor)
c0 = TensorRef("c0", [i4, i2, m2], base_tensor=c_tensor)
g_tensor.add_reference(g)
c_tensor.add_reference(c0)

temp0 = IntermediateResult(g, c0, [m2], shape = [i4, i2, i3, k1], base_tensor=i_tensor)
temp0_ref = TensorRef("temp0ref", [i3, i1, i4, k1], base_tensor=i_tensor)
i_tensor.add_reference(temp0_ref)
i_tensor.add_reference(temp0)
print(i_tensor.get_consistent_ids())
print(i_tensor.print_consistent_positions())

statements = [BinaryContraction(temp0, g, c0), BinaryContraction(result, temp0, temp0_ref)]
contraction = NaryContraction(result, [g, c0, temp0, temp0_ref])
contraction.statements = statements
gen = contraction.fuse_loops(temp0)
fir = FusedIR(gen)
fir.reduce_intermediates()
print(fir)
