from z3 import *
from typing import List

# statements S0, S1, ..., S(n-1)
# indices ind0, ind1, ..., ind(m-1)

# Example:
# S0: T1[j,m,k] = dT1[j,f] * TEov[m,k,f]
# S1: T2[i,j,m,l] = d[i,j,l,a] * S_T2[m,i,j,a]
# S2: T3[i,j,m,n,k] = T2[i,j,m,l] * TEov[n,k,l]
# S3: T4[i,j,m,k] = sT1[n,i,j] * T3[i,j,m,n,k]
# S4: R[i,j] = T1[j,m,k] * T4[i,j,m,k]

# num statements
n = 4
# num indices: m, i, n, j, k, f_mu, e_mu
m = 8
# for printing
ind2str = ["a", "b", "c", "d", "p", "q", "r", "s"]
stmt2str = ["S0: T1[a, q, r, s] = A[p, q, r, s] * C4[p, a]",
            "S1: T2[a, b, r, s] = T1[a, q, r, s] * C3[q, b]",
            "S2: T3[a, b, c, s] = T2[a, b, r, s] * C2[r, c]",
            "S3: B[a, b, c, d] = T3[a, b, c, s] * C1[s, d]",
            ]
# "S4: R[i,j] = H[m, i, j] * T4[i,j,m,k]" ]
# dependences between statements

deps = [[0, 1], [1, 2], [2, 3]]  # , [3, 4]]

# indices used in statements
inds = [[0, 4, 5, 6, 7],      # S0: a, p, q, r, s
        [0, 1, 5, 6, 7],      # S1: a, b, q, r, s
        [0, 1, 2, 6, 7],      # S2: a, b, c, r, s
        [0, 1, 2, 3, 7],      # S3: a, b, c, d, s
        ]
# [ 0, 1, 2, 3 ] ]      # S4: i, j, k, m
# indices used in statement outputs
out_inds = [[0, 5, 6, 7],     # S0: a, q, r, s
            [0, 1, 6, 7],     # S1: a, b, r, s
            [0, 1, 2, 7],     # S2: a, b, c, s
            [0, 1, 2, 3]      # S3: a, b, c, d
            ]
# [ 0, 1 ] ]          # S4: i, j

class SolverError(Exception):
    pass

#indices is set of loops surrounding each statement, really the hash values of SparseIndexs
#indices_id_map is a dict{index_hash: SparseIndex}
def run_solver(statements:List, indices, deps, out_inds, indices_id_map, fusion_threshold, workspace):
    # constraint variables spos_0, spos_1, ... for topsort order
    spos = [Int("spos_%s" % i) for i in range(len(statements))]
    # constraints on the range of spos_*: 0 <= spos_* <= n-1
    spos_range = [And(0 <= spos[i], spos[i] < len(statements)) for i in range(len(statements))]
    # all spos_* are unique: this, their values define a permutation of 0, 1, ...
    spos_unique = [Distinct([spos[i] for i in range(len(statements))])]
    # dependences to be respected by the topsort order
    spos_depend = [And([spos[d[0]] < spos[d[1]] for d in deps])]

    # constraint variables lpos_0_0, lpos_0_1, ..., lpos_2_5
    # variable lpos_x_y is not used when statement x does not use index y
    lpos = []
    for s in range(len(statements)):
        lpos.append({ind: Int("lpos_%s_%s" % (s, ind)) for ind in indices_id_map.keys()})
    #lpos = [[Int("lpos_%s_%s" % (i, j)) for j in indices] for i in range(len(statements))]
    #print("indices_map", indices_id_map)

    # constraints on the range of lpos_*_*
    lpos_range = []
    lpos_unique = []
    lpos_absent = []
    lpos_contraction = []
    lpos_order_by_size = []
    for s in range(len(statements)):
        indxs = indices[s]
        lpos_range += [And(0 <= lpos[s][indxs[j]], lpos[s][indxs[j]]
                           < len(indxs)) for j in range(len(indxs))]
        lpos_unique += [Distinct([lpos[s][indxs[j]]
                                 for j in range(len(indxs))])]
        for i in indices_id_map.keys():
            if (i not in indxs):
                lpos_absent += [And(1000*(s+1) <= lpos[s][i],
                                    lpos[s][i] < 1000*(s+2))]
        # contraction index should not be innermost in the loop order, unless it is the last statement.
        # this is so that we use atleast a 1D workspace - contractions are fast that way.
        if not statements[s].is_last() and statements[s].get_contraction_id() is not None:
            lpos_contraction += [lpos[s][statements[s].get_contraction_id()] < (len(indxs)-1)]
        elif statements[s].is_last() and statements[s].get_contraction_id() is not None:
            lpos_contraction += [lpos[s][statements[s].get_contraction_id()] == (len(indxs)-1)]
        # add constraint to push small loop down
        for ind1 in indxs:
            for ind2 in indxs:
                if indices_id_map[ind1].get_span() < indices_id_map[ind2].get_span():
                    lpos_order_by_size += [lpos[s][ind1] > lpos[s][ind2]]
        
    #lpos_contraction = And(lpos_contraction)
    #lpos_order_by_size = And(lpos_order_by_size)

    # for every producer-consumer pair, try to reduce intermediate temp to a scalar
    prod_cons_fusion = []
    for d in deps:
        source = d[0]
        target = d[1]
        temp_indices = out_inds[source]
        # each level in the outermost band
        for level in range(len(temp_indices)-(fusion_threshold-1)):
            # first, certain indices of the temp should be in the outermost position
            prod_cons_fusion += [Or([lpos[source][temp_indices[j]]
                                    == level for j in range(len(temp_indices))])]
            # second, they should be the same for the producer and the consumer
            prod_cons_fusion += [And([Implies(lpos[source][temp_indices[j]] == level,
                                     lpos[target][temp_indices[j]] == level) for j in range(len(temp_indices))])]
            # third, all in-between statements must have the temp indices in the same positions
            for s in range(len(statements)):
                if (s != source and s != target):
                    prod_cons_fusion += [Implies(And(spos[source] < spos[s], spos[s] < spos[target]), And([Implies(
                        lpos[source][temp_indices[j]] == level, lpos[s][temp_indices[j]] == level) for j in range(len(temp_indices))]))]
        # don't get same permuataion for prod and cons

    # Now add constraints for data layout of input tensors.
    dpos_to_str = {}
    dpos_vars = {}
    dpos_unique = []
    dpos_bounds = []
    dpos_lpos_cons = []
    input_tensors = set([t for s in statements for t in s.get_input_tensors()])
    for ind_t, t in enumerate(input_tensors):
        this_tensor_vars = []
        for ind_d, d in enumerate(t.shape):
            some_dpos_var = Int("dpos_%s_%s" % (ind_t, ind_d))
            dpos_to_str[some_dpos_var] = d
            this_tensor_vars.append(some_dpos_var)
        dpos_vars[t] = this_tensor_vars
    for tens, dvars_list in dpos_vars.items():
        dpos_unique.append(Distinct(dvars_list))
        dpos_bounds.append(And([And(0 <= dvars_list[i], dvars_list[i] < len(tens.get_shape())) for i in range(len(dvars_list))]))
    for ind_s, s in enumerate(statements):
        for tens in s.get_input_tensors():
            for ind_s1, s1 in enumerate(tens.get_shape()):
                for ind_s2, s2 in enumerate(tens.get_shape()):
                    if ind_s1 == ind_s2:
                        continue

                    dpos_lpos_cons.append(Implies(lpos[ind_s][s1.get_id()] < lpos[ind_s][s2.get_id()], dpos_vars[tens][ind_s1] < dpos_vars[tens][ind_s2]))
    dpos_equality_constraints = []
    for t1 in input_tensors:
        for t2 in input_tensors:
            t1_vars = dpos_vars[t1]
            t2_vars = dpos_vars[t2]
            if t1.is_equivalent(t2) and t1 != t2:
                dpos_equality_constraints.append(And([t1_vars[i] == t2_vars[i] for i in range(len(t1_vars))]))

    all_dpos_constraints = dpos_unique + dpos_bounds + dpos_lpos_cons + dpos_equality_constraints




    # put it all together
    all_constraints = spos_depend + spos_range + spos_unique + \
        lpos_range + lpos_unique + lpos_absent + prod_cons_fusion +all_dpos_constraints + lpos_order_by_size
    if workspace:
        all_constraints += lpos_contraction

    # solve and print
    s = Solver()
    s.add(all_constraints)
    #print(s)
    if s.check() == sat:
        m = s.model()
        #print("Topsort order:")
        for i in range(len(statements)):
            for j in range(len(statements)):
                if (m[spos[j]] == i):
                    #print("%s" % statements[j])
                    input_orders = {}
                    for inp_t in statements[j].get_input_tensors():
                        #print("  %s" % inp_t.name)
                        dpvarlist = dpos_vars[inp_t]
                        dpvarlist.sort(key = lambda v: m[v].as_long())
                        #for dpvar in dpvarlist:
                        #    print("    %s" % dpos_to_str[dpvar])
                        input_orders[inp_t] = list(map(lambda v: dpos_to_str[v], dpvarlist))

                    indxs = indices[j]
                    #print(" Loop order:")
                    loop_order = []
                    for k in range(len(indxs)):
                        for p in range(len(indxs)):
                            if (m[lpos[j][indxs[p]]] == k):
                                loop_order.append(indices_id_map[indxs[p]])
                                #print("%s" % indices_id_map[indxs[p]])
                    stmt_tup = (statements[j], loop_order, input_orders)
                    yield stmt_tup
    else:
        print("Unsatisfiable")
        raise SolverError("Unsatisfiable")
