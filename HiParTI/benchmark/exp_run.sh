#!/bin/bash

export EXPERIMENT_MODES=3
echo "==========================="
numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt \
/home/yufan/HiParTI/benchmark/conf-t1.txt 3 \
P /home/yufan/data_3cent_realmid_sparta/Phat.tns \
I /home/yufan/data_3cent_realmid_sparta/Intt.tns \
C /home/yufan/data_3cent_realmid_sparta/C.tns 

echo "===========t1-2================"

numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt \
/home/yufan/HiParTI/benchmark/conf-t1-2.txt 3 \
P /home/yufan/data_3cent_realmid_sparta/Phatt.tns \
I /home/yufan/data_3cent_realmid_sparta/Intt.tns \
C /home/yufan/data_3cent_realmid_sparta/Ct.tns 


echo "===========t2-1================"

numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt \
/home/yufan/HiParTI/benchmark/conf-t2-1.txt 3 \
I /home/yufan/data_3cent_realmid_sparta/int3c1.tns \
D /home/yufan/data_3cent_realmid_sparta/D.tns \
Z /home/yufan/data_3cent_realmid_sparta/threec_int.tns 

echo "============t2-2==============="

numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt \
/home/yufan/HiParTI/benchmark/conf-t2-2.txt 3 \
I /home/yufan/data_3cent_realmid_sparta/int3c1.tns \
D /home/yufan/data_3cent_realmid_sparta/D.tns \
Z /home/yufan/data_3cent_realmid_sparta/int3c3.tns 


echo "============t2-3==============="

numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt \
/home/yufan/HiParTI/benchmark/conf-t2-3.txt 3 \
I /home/yufan/data_3cent_realmid_sparta/int3c2.tns \
D /home/yufan/data_3cent_realmid_sparta/D.tns \
Z /home/yufan/data_3cent_realmid_sparta/threec_int.tns 

echo "============t2-4==============="

numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt \
/home/yufan/HiParTI/benchmark/conf-t2-4.txt 3 \
I /home/yufan/data_3cent_realmid_sparta/int3c2.tns \
D /home/yufan/data_3cent_realmid_sparta/D.tns \
Z /home/yufan/data_3cent_realmid_sparta/int3c3.tns 

echo "============t2-5==============="

numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt \
/home/yufan/HiParTI/benchmark/conf-t2-5.txt 3 \
I /home/yufan/data_3cent_realmid_sparta/threec_int.tns \
D /home/yufan/data_3cent_realmid_sparta/Dt.tns \
Z /home/yufan/data_3cent_realmid_sparta/int3c1.tns 

echo "============t2-6==============="

numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt \
/home/yufan/HiParTI/benchmark/conf-t2-6.txt 3 \
I /home/yufan/data_3cent_realmid_sparta/threec_int.tns \
D /home/yufan/data_3cent_realmid_sparta/Dt.tns \
Z /home/yufan/data_3cent_realmid_sparta/int3c2.tns 


echo "============t2-7==============="

numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt \
/home/yufan/HiParTI/benchmark/conf-t2-7.txt 3 \
I /home/yufan/data_3cent_realmid_sparta/int3c3.tns \
D /home/yufan/data_3cent_realmid_sparta/Dt.tns \
Z /home/yufan/data_3cent_realmid_sparta/int3c2.tns 

echo "============t2-8==============="

numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt \
/home/yufan/HiParTI/benchmark/conf-t2-8.txt 3 \
I /home/yufan/data_3cent_realmid_sparta/int3c3.tns \
D /home/yufan/data_3cent_realmid_sparta/Dt.tns \
Z /home/yufan/data_3cent_realmid_sparta/int3c1.tns 