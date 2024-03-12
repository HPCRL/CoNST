This repository contains the source code for the CoNST code generator.
In order to reproduce the results of the paper, please use the docker image.
# Running the docker image
Run `docker run -it smr97/const_test:v0`

# Once inside the docker image
First, run `source conda_setup.sh` to activate the conda environment.
Verify that the environment "py10" is active.

Go to the const source code: `cd /CoNST`

Next step is to build TACO
run `bash build_taco.sh` to build TACO.

To reproduce the quantum chemistry experiments, run `./run_qc_3cint.sh` in the CoNST directory.
To reproduce the tensor decomposition experiments, run `./run_frostt.sh` in the CoNST directory.

The graphs will be saved in the `/CoNST/graphs` directory as PDFs
