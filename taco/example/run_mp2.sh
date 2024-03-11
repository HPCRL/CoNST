# run small size
./compile.sh mp2_fusion small &> ../../graphs/mp2_log_small.txt
./compile.sh mp2_fusion medium &> ../../graphs/mp2_log_medium.txt
./compile.sh mp2_fusion large &> ../../graphs/mp2_log_large.txt
./compile.sh 3c_to_4c small &> ../../graphs/4c_log_small.txt
./compile.sh 3c_to_4c medium &> ../../graphs/4c_log_medium.txt
./compile.sh 3c_to_4c large &> ../../graphs/4c_log_large.txt
