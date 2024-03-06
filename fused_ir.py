from typing import List, Dict
from parsing import Tensor, SparseIndex, IntermediateResult
import copy


class SparseContraction():
    def __init__(self, result: Tensor, operands: List[Tensor], iterands: List[SparseIndex], input_tensor_ordering: Dict[Tensor, List[SparseIndex]]):
        self.lhs = result
        self.rhs = operands
        self.loops = iterands
        self.loop_iter = iter(self.loops)
        self.inp_orders = input_tensor_ordering

    def get_loops(self):
        return self.loops

    def get_next_loop(self):
        return next(self.loop_iter)

    def lower_to_taco(self, index_order: List[SparseIndex]):
        return f"{self.lhs.emit_access(index_order)} += " + "*".join([t.emit_access(self.inp_orders[t] if t in self.inp_orders else index_order) for t in self.rhs])

    def __repr__(self, depth=0):
        return "".join(["\t"]*depth) + f"{self.lhs} += " + "*".join([t.__repr__(self.inp_orders[t] if t in self.inp_orders else None) for t in self.rhs])


def place_contraction(node, contr: SparseContraction):
    def make_single_fibre(root_node, contr):
        try:
            next_loop = contr.get_next_loop()
        except StopIteration:
            root_node.add_contraction(contr)
            return
        new_node = SparseIteration(next_loop, [contr])
        root_node.add_child(new_node)
        return make_single_fibre(new_node, contr)
    # to respect the dependency, we can only look at the last child node
    if len(node.children) == 0:
        #new_node = SparseIteration(next_loop, [contr])
        return make_single_fibre(node, contr)
    else:
        last_child = node.children[-1]
        try:
            next_loop = contr.get_next_loop()
        except StopIteration:
            # we are done
            last_child.add_contraction(contr)
            return
        if last_child.iterand == next_loop:
            # that's his problem
            return place_contraction(last_child, contr)
        # if we get here, we need to add a new node
        new_node = SparseIteration(next_loop, [contr])
        node.add_child(new_node)
        return make_single_fibre(new_node, contr)


class FusedIR:
    def __init__(self, generator):
        self.children = []
        solver_output = []
        self.intermediate_tensors = []
        self.og_tensors = []
        self.final_result_tensor = None
        self.tensor_orders = {}
        for unfused_set in generator:
            solver_output.append(unfused_set)
            stmt, _, _ = unfused_set
            if isinstance(stmt.get_lhs(), IntermediateResult):
                self.intermediate_tensors.append(stmt.get_lhs())
            else:
                self.og_tensors.append(stmt.get_lhs())
                self.final_result_tensor = stmt.get_lhs()
            this_og_tensors = filter(lambda t: not isinstance(
                t, IntermediateResult), stmt.get_rhs())
            self.og_tensors.extend(this_og_tensors)

        first_tree = None
        for child in solver_output:
            stmt, loop_order, tensor_order = child
            self.tensor_orders.update(tensor_order)
            if first_tree is None:
                contr_leaf = SparseContraction(
                    stmt.get_lhs(), stmt.get_rhs(), loop_order, tensor_order)
                this_iter = SparseIteration(loop_order[0], [contr_leaf])
                first_tree = this_iter
                self.add_child(first_tree)
                for l in loop_order[1:]:
                    new_iter = SparseIteration(l, [contr_leaf])
                    this_iter.add_child(new_iter)
                    this_iter = new_iter
            else:
                contr_leaf = SparseContraction(
                    stmt.get_lhs(), stmt.get_rhs(), loop_order, tensor_order)
                place_contraction(self, contr_leaf)

    def add_child(self, child):
        self.children.append(child)

    def get_path_suchthat(self, predicate):
        for child in self.children:
            try_path = child.get_path_suchthat(predicate)
            if try_path is not None:
                return try_path
        raise ValueError("No path found")

    # this reduces the fused dimensions from the intermediate tensors
    def reduce_intermediates(self):
        for int_tens in self.intermediate_tensors:
            producer_path = self.get_path_suchthat(lambda x: x.lhs == int_tens)
            consumer_path = self.get_path_suchthat(lambda x: int_tens in x.rhs)
            indices_to_fuse = []
            for i in range(len(producer_path)):
                if producer_path[i] == consumer_path[i]:
                    indices_to_fuse.append(producer_path[i])
                else:
                    break
            if len(indices_to_fuse) > 0:
                int_tens.fuse(indices_to_fuse)

    def lower_to_taco(self):
        if len(self.children) > 1:
            raise NotImplementedError("Only one root node supported")
        only_child = self.children[0]
        return only_child.lower_to_taco([]) + ";"

    def lower_nary_contraction(self):
        lhs_tensorder = self.tensor_orders[self.final_result_tensor] if self.final_result_tensor in self.tensor_orders else None
        lhs = f"{self.final_result_tensor.emit_access(lhs_tensorder, varname=False)}"
        rhs = "*".join([t.emit_access((self.tensor_orders[t]
                                      if t in self.tensor_orders else None), varname=False) for t in filter(lambda t: t != self.final_result_tensor, self.og_tensors)])
        return f"{lhs} += {rhs};"

    def lower_intermediate_tensors(self):
        return "\n".join([int_tens.lower_to_taco() for int_tens in self.intermediate_tensors])

    def lower_og_tensors(self):
        return "\n".join([og_tens.lower_var_decl() for og_tens in self.og_tensors])

    def emit_taco_kernel(self, kernel_name, add_timing=True):
        all_indices = set([i for t in self.og_tensors for i in t.get_shape()])
        header = f"void {kernel_name}(" + ", ".join(
            ["Tensor<double> " + t.name for t in self.og_tensors]) + ") {"
        header += "\n".join([i.lower_to_taco() for i in all_indices])
        header += self.lower_og_tensors()
        header += self.lower_intermediate_tensors()
        header += f"auto fused_ir = {self.lower_to_taco()};"
        header += self.lower_nary_contraction()
        header += f"{self.final_result_tensor.name}.compile(fused_ir);"
        header += "auto start = std::chrono::high_resolution_clock::now();"
        header += f"{self.final_result_tensor.name}.assemble();"
        header += f"{self.final_result_tensor.name}.compute();"
        header += "auto end = std::chrono::high_resolution_clock::now();"
        header += "std::chrono::duration<double, std::milli> elapsed = end - start;"
        header += "std::cout << \"Time " + \
            f"{kernel_name}:  \"<< elapsed.count() <<\" ms \"<< std::endl;"
        header += "}"
        return header

    def __repr__(self):
        # print(self.loop_sequence)
        return "\n".join([node.__repr__(0) for node in self.children])


class SparseIteration:
    def __init__(self, iterand, reachable_contractions):
        self.iterand = iterand
        self.contractions = reachable_contractions
        self.children = []

    def add_child(self, child):
        self.children.append(child)

    def add_contraction(self, contr):
        if contr not in self.contractions:
            self.contractions.append(contr)

    def get_path_suchthat(self, predicate):
        try_path = None
        for child in self.children:
            try_path = child.get_path_suchthat(predicate)
            if try_path is not None:
                return [self.iterand] + try_path

        if len(self.children) == 0:
            for contr in self.contractions:
                if predicate(contr):
                    return [self.iterand]
        return try_path

    def lower_to_taco(self, indices_so_far):
        child_str = ""
        indices_so_far = copy.copy(indices_so_far)
        indices_so_far.append(self.iterand)
        if len(self.children) == 0:
            # emit contractions
            try:
                assert len(self.contractions) == 1
            except AssertionError:
                raise AssertionError(
                    "Can't handle multiple contractions fused throughout")
            child_str = f"{self.contractions[0].lower_to_taco(indices_so_far)}"

        elif len(self.children) == 1:
            child_str = self.children[0].lower_to_taco(indices_so_far)
        else:
            child_str = f"{self.children[0].lower_to_taco(indices_so_far)}"
            for child in self.children[1:]:
                child_str = f"where({child.lower_to_taco(indices_so_far)}, {child_str})"
        my_str = f"forall({self.iterand}, {child_str})"
        return my_str

    def __repr__(self, depth=0):
        #print(f"in repr for {self.iterand}")
        if len(self.children) == 0:
            return "".join(["\t"]*depth) + f"{self.iterand}" + "\n".join([c.__repr__(depth+1) for c in self.contractions])
        else:
            return "".join(["\t"]*depth) + f"{self.iterand}\n" + "\n".join([c.__repr__(depth+1) for c in self.children])
