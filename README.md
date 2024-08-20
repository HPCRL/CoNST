# About CoNST
CoNST is a code generator for sparse tensor networks.
It generates a TACO-IR with optimized tensor layouts and fused loop structures, to accelerate sparse tensor contractions.
TACO compiler is then used to lower the IR to C/CUDA code.
The details of this approach are documented here: https://doi.org/10.1145/3689342
In order to reproduce the results of the paper, please use the docker image.

# Reproducing the results using docker
Run the docker image (public) using `docker run -it smr97/const_test:v0`

Once inside the container, run `source conda_setup.sh` to activate the conda environment.
Verify that the environment "py10" is active.

Go to the const source code: `cd /CoNST`

Next step is to build TACO
run `bash build_taco.sh` to build TACO.

To reproduce the quantum chemistry experiments, run `./run_qc_3cint.sh` in the CoNST directory.
To reproduce the tensor decomposition experiments, run `./run_frostt.sh` in the CoNST directory.

The graphs will be saved in the `/CoNST/graphs` directory as PDFs


If you found CoNST useful, please consider citing our paper:
```
@article{10.1145/3689342,
author = {Raje, Saurabh and Xu, Yufan and Rountev, Atanas and Valeev, Edward and Sadayappan, P.},
title = {CoNST: Code Generator for Sparse Tensor Networks},
year = {2024},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
issn = {1544-3566},
url = {https://doi.org/10.1145/3689342},
doi = {10.1145/3689342},
abstract = {Sparse tensor networks represent contractions over multiple sparse tensors. Tensor contractions are higher-order analogs of matrix multiplication. Tensor networks arise commonly in many domains of scientific computing and data science. Such networks are typically computed using a tree of binary contractions. Several critical inter-dependent aspects must be considered in the generation of efficient code for a contraction tree, including sparse tensor layout mode order, loop fusion to reduce intermediate tensors, and the mutual dependence of loop order, mode order, and contraction order. We propose CoNST, a novel approach that considers these factors in an integrated manner using a single formulation. Our approach creates a constraint system that encodes these decisions and their interdependence, while aiming to produce reduced-order intermediate tensors via fusion. The constraint system is solved by the Z3 SMT solver and the result is used to create the desired fused loop structure and tensor mode layouts for the entire contraction tree. This structure is lowered to the IR of the TACO compiler, which is then used to generate executable code. Our experimental evaluation demonstrates significant performance improvements over current state-of-the-art sparse tensor compiler/library alternatives.},
note = {Just Accepted},
journal = {ACM Trans. Archit. Code Optim.},
month = {aug},
keywords = {sparse tensors, tensor networks, tensor layout, loop fusion}
}
```
