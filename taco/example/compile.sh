#!/bin/bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/../build/lib
export TMPDIR=$PWD
# To get the source code, export TMPDIR
#g++ -g -std=c++17 -DTACO -fopenmp -I ../include -L ../build/lib mttkrp_exp.cc -ltaco -o mttkrp_exp
#gdb mttkrp_exp
g++ -g -O3 -march=native -mtune=native -ffast-math -std=c++17 -DTACO -fopenmp -I ../include -I ../src/lower $1.cc -L ../build/lib -ltaco -o $1
#g++ -g -std=c++17 -DTACO -fopenmp -I ../include -L ../build/lib $1.cc -ltaco -o $1
#g++ -std=c++17 -DTACO -I ../include -L ../build/lib $1.cc -ltaco -o $1
#g++ -std=c++17 -fsanitize=address -DTACO -I ../include -L ../build/lib $1.cc -ltaco -o $1
#OMP_THREAD_LIMIT=1 ./$1
numactl --physcpubind=+1 ./$1 $2 $3
#./$1
