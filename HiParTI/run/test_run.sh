#!/bin/bash

export EXPERIMENT_MODES=3

numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt \
/home/yufan/HiParTI/benchmark/conf.txt 3 \
X /home/yufan/data_3cent_small_sparta/Int_scipy.tns \
Y /home/yufan/data_3cent_small_sparta/C_scipy.tns \
Z /home/yufan/data_3cent_small_sparta/Phat_scipy.tns 


#-X $TENSOR_DIR/nips.bin -Y $TENSOR_DIR/nips.bin -m 3 -x 0 1 2 -y 0 1 2 -t 12




# default
# (base) yufan@zeus:~/HiParTI/run$ ./test_run.sh 
# 1st tensor file: /home/yufan/tensors/nips.bin
# 2nd tensor file: /home/yufan/tensors/nips.bin
# 3
# #Contraction modes: 3
# COO Sparse Tensor information (use sptIndex, sptValue))---------
# DIMS=2482x2862x14036x17 NNZ=3101609 DENSITY=1.829883e-06
# Average slice length (c): 1249.64 , 1083.72 , 220.98 , 182447.59
# COO-STORAGE=59.16 MiB

# COO Sparse Tensor information (use sptIndex, sptValue))---------
# DIMS=2482x2862x14036x17 NNZ=3101609 DENSITY=1.829883e-06
# Average slice length (c): 1249.64 , 1083.72 , 220.98 , 182447.59
# COO-STORAGE=59.16 MiB

# [Input Processing]: 0.248879 s
# [Index Search]: 0.075946 s
# [Accumulation]: 0.009404 s
# [Writeback]: 0.000016 s
# [Output Sorting]: 0.000443 s
# [Total time]: 0.334687 s

# COO Sparse Tensor information (use sptIndex, sptValue))---------
# DIMS=17x17 NNZ=175 DENSITY=6.055363e-01
# Average slice length (c): 10.29 , 10.29
# COO-STORAGE=2.05 KiB




# (base) yufan@zeus:~/HiParTI/run$ ./test_run.sh 

# (X * Y) 

# n 1
# n : 1

# m 3 x 0 1 2 y 0 1 2
# m : 3
# x : 0
# x : 1
# x : 2
# y : 0
# y : 1
# y : 2

# t 12expression: (X * Y) 

# n: 1
# t: 12
# operand1: 
# COO Sparse Tensor information (use sptIndex, sptValue))---------
# DIMS=2482x2862x14036x17 NNZ=3101609 DENSITY=1.829883e-06
# Average slice length (c): 1249.64 , 1083.72 , 220.98 , 182447.59
# COO-STORAGE=59.16 MiB

# operand2: 
# COO Sparse Tensor information (use sptIndex, sptValue))---------
# DIMS=2482x2862x14036x17 NNZ=3101609 DENSITY=1.829883e-06
# Average slice length (c): 1249.64 , 1083.72 , 220.98 , 182447.59
# COO-STORAGE=59.16 MiB

# Original Tensors: 
# cmode m 3
# x y 0
# x y 1
# x y 2
# [Input Processing]: 0.254398 s
# [Index Search]: 0.074429 s
# [Accumulation]: 0.009200 s
# [Writeback]: 0.000017 s
# [Output Sorting]: 0.000458 s
# [Total time]: 0.338501 s

# COO Sparse Tensor information (use sptIndex, sptValue))---------
# DIMS=17x17 NNZ=175 DENSITY=6.055363e-01
# Average slice length (c): 10.29 , 10.29
# COO-STORAGE=2.05 KiB
