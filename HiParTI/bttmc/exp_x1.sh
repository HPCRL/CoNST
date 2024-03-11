#!/bin/bash

#!/bin/bash
echo "The I is: $1"
echo "M2 is: $2"
echo "M1 is: $3"
echo "config file is: $4"

export EXPERIMENT_MODES=3
echo "==========================="
numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt \
 $4 3 \
I $1 \
M $2  \
N $3 
