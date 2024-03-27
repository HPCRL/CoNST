# run small size
TENSORS=("flickr-3d.tns" "nell-1.tns" "nell-2.tns" "vast-2015-mc1-3d.tns")
METHODS=(0 1 2)
for tensor in "${TENSORS[@]}"
do
    for method in "${METHODS[@]}"
    do
        echo "Running MTTKRP1 for tensor $tensor $method"
        ./compile.sh mttkrp $tensor $method &>> ../../graphs/mttkrp_mode1.txt
        ./compile.sh mttkrp_mode2 $tensor $method &>> ../../graphs/mttkrp_mode2.txt
        ./compile.sh mttkrp_mode3 $tensor $method &>> ../../graphs/mttkrp_mode3.txt
    done
done
#./compile.sh mttkrp &> ../../graphs/mttkrp_mode1.txt
#./compile.sh mttkrp_mode2 &> ../../graphs/mttkrp_mode2.txt
#./compile.sh mttkrp_mode3 &> ../../graphs/mttkrp_mode3.txt
