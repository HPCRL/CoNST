from parsing import TensorRef, SparseIndex, BinaryContraction, NaryContraction, IntermediateResult, BaseTensor
from fused_ir import FusedIR

def example_2():
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


def example_l82():
#      {
#        "tag1": "Product",
#        "annot": "g(μ̃1,u2,Κ_1) * C(i_1,μ̃1;a_1i_1) -> I(i_1,u2,Κ_1;a_1i_1)"
#      },
#      {
#        "tag1": "Product",
#        "annot": "I(i_1,u2,Κ_1;a_1i_1) * C(i_2,u2;a_2i_2) -> I(i_1,i_2,Κ_1;a_1i_1,a_2i_2)"
#      },
#      {
#        "tag1": "Product",
#        "annot": "g(i_2,i_1,Κ_1) * I(i_1,i_2,Κ_1;a_1i_1,a_2i_2) -> I(i_2,i_1;a_1i_1,a_2i_2)"
#      }

    muhat1 = SparseIndex("μ̃1")
    muhat2 = SparseIndex("μ̃2")
    i1 = SparseIndex("i1")
    i2 = SparseIndex("i2")
    k = SparseIndex("Κ")
    result = TensorRef("result", [i2, i1])
    g_ref = TensorRef("g", [muhat1, muhat2, k])
    g2_ref = TensorRef("g2", [i2, i1, k])
    c1_ref = TensorRef("c1", [i1, muhat1])
    c2_ref = TensorRef("c2", [i2, muhat2])
    temp0 = IntermediateResult(g_ref, c1_ref, [muhat1], shape = [i1, muhat2, k])
    temp1 = IntermediateResult(temp0, c2_ref, [muhat2], shape = [i1, i2, k])
    statements = [BinaryContraction(temp0, g_ref, c1_ref), BinaryContraction(temp1, temp0, c2_ref), BinaryContraction(result, g2_ref, temp1)]
    contraction = NaryContraction(result, [g_ref, g2_ref, c1_ref, c2_ref])
    contraction.statements = statements
    gen = contraction.fuse_loops(temp0)
    fir = FusedIR(gen)
    fir.reduce_intermediates()
    print(fir)





def example_l143():
#      {
#        "tag1": "Product",
#        "annot": "g(μ̃_19602,i_1,Κ_1) * C(i_2,i_1,μ̃_19602;a_1i_1i_2) -> I(i_2,i_1,Κ_1;a_1i_1i_2)"
#      },
#      {
#        "tag1": "Product",
#        "annot": "I(i_2,i_1,Κ_1;a_1i_1i_2) * I(i_1,i_2,Κ_1;a_2i_1i_2) -> I(i_2,i_1;a_2i_1i_2,a_1i_1i_2)"
#      }

    muhat = SparseIndex("μ̃")
    i1 = SparseIndex("i1")
    i2 = SparseIndex("i2")
    k = SparseIndex("Κ")
    g_tensor = BaseTensor("g")
    c_tensor = BaseTensor("C")
    i_tensor = BaseTensor("I")
    result = TensorRef("result", [i1, i2])
    g_ref = TensorRef("g", [muhat, i1, k], base_tensor=g_tensor)
    c_ref = TensorRef("c", [i2, i1, muhat], base_tensor=c_tensor)
    g_tensor.add_reference(g_ref)
    c_tensor.add_reference(c_ref)
    i_intermediate = IntermediateResult(g_ref, c_ref, [muhat], shape = [i2, i1, k], base_tensor=i_tensor)
    i_intermediate_ref = TensorRef("i_intermediate", [i1, i2, k], base_tensor=i_tensor)
    i_tensor.add_reference(i_intermediate)
    i_tensor.add_reference(i_intermediate_ref)
    i_tensor.print_consistent_positions()
    statements = [BinaryContraction(i_intermediate, g_ref, c_ref), BinaryContraction(result, i_intermediate, i_intermediate_ref)]
    contraction = NaryContraction(result, [g_ref, c_ref, i_intermediate, i_intermediate_ref])
    contraction.statements = statements
    gen = contraction.fuse_loops(i_intermediate)
    fir = FusedIR(gen)
    fir.reduce_intermediates()
    print(fir)


def example_l169():
#      {
#        "tag1": "Product",
#        "annot": "g(i_3,μ̃_19604,Κ_1) * C(i_3,i_2,μ̃_19604;a_2i_2i_3) -> I(i_2,i_3,Κ_1;a_2i_2i_3)"
#      },
#      {
#        "tag1": "Product",
#        "annot": "g(i_2,i_1,Κ_1) * I(i_2,i_3,Κ_1;a_2i_2i_3) -> I(i_3,i_2,i_1;a_2i_2i_3)"
#      }
    i3 = SparseIndex("i3")
    muhat = SparseIndex("μ̃")
    i1 = SparseIndex("i1")
    i2 = SparseIndex("i2")
    k = SparseIndex("Κ")
    result = TensorRef("result", [i3, i2, i1])
    g_ref = TensorRef("g", [i3, muhat, k])
    g_ref2 = TensorRef("g2", [i2, i1, k])
    c_ref = TensorRef("c", [i3, i2, muhat])
    i_intermediate = IntermediateResult(g_ref, c_ref, [muhat], shape = [i2, i3, k])
    statements = [BinaryContraction(i_intermediate, g_ref, c_ref), BinaryContraction(result, g_ref2, i_intermediate)]
    contraction = NaryContraction(result, [g_ref, c_ref, g_ref2])
    contraction.statements = statements
    gen = contraction.fuse_loops(i_intermediate)
    fir = FusedIR(gen)
    fir.reduce_intermediates()
    print(fir)

example_l82()

