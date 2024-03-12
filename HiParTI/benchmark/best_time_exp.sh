#!/bin/bash

DATA_PATH=/home/yufan/data_3cent_realmid_sparta

export EXPERIMENT_MODES=3
echo "==========================="
numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt \
$(pwd -P)/conf-t1-2.txt 3 \
P $DATA_PATH/Phatt.tns \
I $DATA_PATH/Intt.tns \
C $DATA_PATH/Ct.tns 



echo "============t2-3==============="
numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt \
$(pwd -P)/conf-t2-3.txt 3 \
I $DATA_PATH/int3c2.tns \
D $DATA_PATH/D.tns \
Z $DATA_PATH/threec_int.tns 
