from typing import List
from s import run_solver, SolverError


class Index:
    def __init__(self, name: str, span: List[int]):
        self.name = name
        self.span = span

    def get_id(self):
        return self.__hash__()


class SparseIndex(Index):
    def __init__(self, name: str, size: int = 1):
        self.name = name
        self.span = size

    def get_span(self):
        return self.span

    def __repr__(self):
        return self.name

    def lower_to_taco(self):
        return f"IndexVar {self.name}(" + "\"" + self.name + "\");"


class DenseIndex(Index):
    def __init__(self, name: str, span: List[int]):
        self.name = name
        self.span = span


class Tensor:
    def __init__(self, name: str, shape: List[Index], base_tensor=None, dense=False):
        self.name = name
        self.shape = shape
        self.base_tensor = base_tensor

    def is_equivalent(self, other):
        if self.base_tensor is not None and other.base_tensor is not None:
            return self.base_tensor == other.base_tensor
        else:
            return False

    def get_shape(self):
        return self.shape

    def get_varname(self):
        return self.name + "_var"

    def get_dim(self, index):
        return self.shape[index].get_span()

    def emit_access(self, index_order: List[SparseIndex], varname=True):
        if index_order is None:
            index_order = self.last_used_order
        else:
            self.last_used_order = index_order
        return (self.get_varname() if varname else self.name) + "(" + ",".join([str(i) for i in filter(lambda idx: idx in self.shape, index_order)]) + ")"

    def __repr__(self, ordered_shape=None):
        if ordered_shape is None:
            shape = self.shape
        else:
            shape = ordered_shape
        return self.name+"("+",".join([str(s) for s in shape])+")"

    def __iadd__(self, other):
        assert isinstance(other, MultExpr)
        return NaryContraction(self, other.ops)

    def lower_var_decl(self):
        # TensorVar teov1_var = teov1.getTensorVar();
        return f"TensorVar {self.name}_var = {self.name}.getTensorVar();"

    def __mul__(self, other):
        assert isinstance(other, Tensor)
        return MultExpr(self, other)


class IntermediateResult(Tensor):
    def __init__(self, left_tensor, right_tensor, contraction_indices: List[SparseIndex], const_shape=""):
        self.name = left_tensor.name + right_tensor.name
        self.shape = set(left_tensor.shape).union(set(right_tensor.shape)).difference(
            set(contraction_indices))
        self.fused_shape = self.shape
        self.const_shape = const_shape

    def fuse(self, indices: List[SparseIndex]):
        self.fused_shape = self.shape.difference(set(indices))

    def get_varname(self):
        return self.name

    def emit_access(self, index_order: List[SparseIndex]):
        if index_order is None:
            index_order = self.fused_shape
        fused_ordered_shape = [i for i in filter(
            lambda idx: idx in self.fused_shape, index_order)]
        assert len(fused_ordered_shape) == len(self.fused_shape)
        return self.get_varname() + "(" + ",".join([str(i) for i in fused_ordered_shape]) + ")"

    def _generate_shape_str(self):
        if self.const_shape != "":
            return self.const_shape
        else:
            return ",".join([str(s.get_span()) for s in self.fused_shape])

    def _generate_format_str(self):
        return ",".join([f"taco::dense" for _ in self.fused_shape])

    def __repr__(self, ordered_shape=None):
        return self.name+"("+",".join([str(s) for s in self.fused_shape])+")"

    def lower_to_taco(self):
        # TODO replace the component type
        return f"TensorVar {self.name} = TensorVar(" + "\"" + f"{self.name}"+"\"" + f", Type(Float64, "+"{"+self._generate_shape_str() + "}), " + "{" + self._generate_format_str() + "});"


class MultExpr:
    def __init__(self, op_left, op_right):
        self.ops = [op_left, op_right]

    def add_operand(self, op):
        self.ops.append(op)

    def __mul__(self, other):
        assert isinstance(other, Tensor)
        self.add_operand(other)
        return self


class BinaryContraction:
    def __init__(self, lhs: Tensor, rhs_left: Tensor, rhs_right: Tensor):
        self.lhs = lhs
        self.op_left = rhs_left
        self.op_right = rhs_right
        self.loops = self.__all_iterators()
        contraction_indices = set([s for op in [self.op_left, self.op_right]
                                  for s in op.get_shape()]).difference(set(self.lhs.get_shape()))
        assert len(contraction_indices) == 1 or len(contraction_indices) == 0
        if len(contraction_indices) == 1:
            self.contraction_index = contraction_indices.pop()
        else:
            self.contraction_index = None

    def get_lhs(self):
        return self.lhs

    def get_rhs(self):
        return [self.op_left, self.op_right]

    # return a (potentially empty) list of input tensors that are not generated by binarization. come from the OG sparse tensor network
    def get_input_tensors(self):
        inp_tensors = []
        for op in [self.op_left, self.op_right]:
            if not isinstance(op, IntermediateResult):
                inp_tensors.append(op)
        return inp_tensors

    def get_loop_ids(self):
        return list(map(lambda it: it.get_id(), self.loops))

    def get_contraction_id(self):
        if self.contraction_index is None:
            return None
        return self.contraction_index.get_id()

    def is_last(self):
        return not isinstance(self.lhs, IntermediateResult)

    def get_lhs_shape_ids(self):
        return list(map(lambda it: it.get_id(), self.lhs.get_shape()))

    def __all_iterators(self):
        return set([i for op in [self.lhs, self.op_left, self.op_right] for i in op.get_shape()])

    def __repr__(self):
        return f"{self.loops}\n" + f"\t{self.lhs} += {self.op_left} * {self.op_right}"


class NaryContraction:
    def __init__(self, lhs: Tensor, rhs: List[Tensor]):
        self.lhs = lhs
        self.rhs = rhs
        self.statements = []
        self.loops = self.__all_iterators()
        self._make_contraction_map()
        # print(self.contr_index_tensor)

    def __all_iterators(self):
        return set([i for op in [self.lhs] + self.rhs for i in op.get_shape()])

    def _shape_check(self):
        lhs_shape = self.lhs.get_shape()
        rhs_shape = [s for op in self.rhs for s in op.get_shape()]
        assert len(lhs_shape) < len(rhs_shape)
        for ls in lhs_shape:
            try:
                assert ls in rhs_shape
            except AssertionError:
                return False
        return True

    def validate(self):
        assert len(self.rhs) >= 1
        return self._shape_check()

    def _make_contraction_map(self):
        self.contr_index_tensor = {}
        contraction_indices = set([s for op in self.rhs for s in op.get_shape()]).difference(
            set(self.lhs.get_shape()))
        # print(contraction_indices)
        for ci in contraction_indices:
            self.contr_index_tensor[ci] = []
            for op in self.rhs:
                if ci in op.get_shape():
                    self.contr_index_tensor[ci].append(op)

    def _consume_tensor(self, tensor):
        for _, tensors in self.contr_index_tensor.items():
            if tensor in tensors:
                tensors.remove(tensor)

    def get_contraction_edges(self, tens1, tens2):
        self._consume_tensor(tens1)
        self._consume_tensor(tens2)
        tens1_shape = tens1.get_shape()
        tens2_shape = tens2.get_shape()
        possible_contraction_indices = set(
            tens1_shape).intersection(set(tens2_shape))
        contraction_indices = []
        for pci in possible_contraction_indices:
            if pci in self.contr_index_tensor and len(self.contr_index_tensor[pci]) == 0:
                contraction_indices.append(pci)

        return contraction_indices

    def binarize(self):
        start_tensor = self.rhs[0]
        for ind, op in enumerate(self.rhs[1:]):
            if ind == len(self.rhs) - 2:
                self.statements.append(BinaryContraction(
                    self.lhs, start_tensor, op))
                break
            int_tensor = IntermediateResult(
                start_tensor, op, self.get_contraction_edges(start_tensor, op))
            self.statements.append(BinaryContraction(
                int_tensor, start_tensor, op))
            start_tensor = int_tensor

    def is_binarized(self):
        return len(self.statements) > 0

    def __repr__(self):
        if self.is_binarized():
            return "\n".join([str(s) for s in self.statements])
        else:
            return f"{self.loops}\n" + f"\t{self.lhs} += " + " * ".join([str(s) for s in self.rhs])

    def opdag(self):
        # each element of the list is a tuple of indices (i, j), statement i is producer, statement j is consumer
        dependence_edges = []
        for ind, s in enumerate(self.statements):
            for ind_second, s_second in enumerate(self.statements):
                if s.lhs.name == s_second.op_left.name or s.lhs.name == s_second.op_right.name:
                    dependence_edges.append((ind, ind_second))
        return dependence_edges

    def fuse_loops(self, workspace=True):
        if not self.is_binarized():
            self.binarize()
        index_map = {}
        for s in self.statements:
            for l in s.loops:
                index_map[l.get_id()] = l

        for thresh in range(1, 5):
            try:
                next(run_solver(self.statements, [contr.get_loop_ids() for contr in self.statements], self.opdag(), [
                    contr.get_lhs_shape_ids() for contr in self.statements], index_map, thresh, workspace))
                return run_solver(self.statements, [contr.get_loop_ids() for contr in self.statements], self.opdag(), [
                    contr.get_lhs_shape_ids() for contr in self.statements], index_map, thresh, workspace)
            except SolverError as _:
                print(f"Did not work for {thresh}")
                continue
        raise SolverError("Could not fuse loops")

        # print(run_solver(self.statements, [contr.get_loop_ids() for contr in self.statements], self.opdag(), [
        #      contr.get_lhs_shape_ids() for contr in self.statements], index_map, 2))
        # print(run_solver(self.statements, [contr.get_loop_ids() for contr in self.statements], self.opdag(), [
        #      contr.get_lhs_shape_ids() for contr in self.statements], index_map, 1))
