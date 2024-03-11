#!/bin/bash

export EXPERIMENT_MODES=3
echo "==========================="
numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt \
/home/yufan/HiParTI/benchmark/conf-x1.txt 3 \
I /home/yufan/hpcrl_taco/example/inp_scipy.tns \
M /home/yufan/hpcrl_taco/example/m2.tns  \
N /home/yufan/hpcrl_taco/example/m1.tns 
