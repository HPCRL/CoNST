#!/bin/bash

export EXPERIMENT_MODES=3
echo "==========================="
numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt \
/home/yufan/HiParTI/benchmark/conf-t1.txt 3 \
P /home/yufan/data_3cent_real_sparta/Phat.tns \
I /home/yufan/data_3cent_real_sparta/Intt.tns \
C /home/yufan/data_3cent_real_sparta/C.tns 

echo "===========t2-1================"

numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt \
/home/yufan/HiParTI/benchmark/conf-t2-1.txt 3 \
I /home/yufan/data_3cent_real_sparta/int3c1.tns \
D /home/yufan/data_3cent_real_sparta/D.tns \
Z /home/yufan/data_3cent_real_sparta/threec_int.tns 


echo "============= LARGE =============="
numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt \
/home/yufan/HiParTI/benchmark/conf-t1.txt 3 \
P /home/yufan/data_3cent_reallarge_sparta/Phat.tns \
I /home/yufan/data_3cent_reallarge_sparta/Intt.tns \
C /home/yufan/data_3cent_reallarge_sparta/C.tns 

echo "===========t2-1================"

numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt \
/home/yufan/HiParTI/benchmark/conf-t2-1.txt 3 \
I /home/yufan/data_3cent_reallarge_sparta/int3c1.tns \
D /home/yufan/data_3cent_reallarge_sparta/D.tns \
Z /home/yufan/data_3cent_reallarge_sparta/threec_int.tns 