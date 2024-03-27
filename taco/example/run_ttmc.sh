# run small size
TENSORS=("flickr-3d.tns" "nell-1.tns" "nell-2.tns" "vast-2015-mc1-3d.tns")
METHODS=(0 1 2)
for tensor in "${TENSORS[@]}"
do
    for method in "${METHODS[@]}"
    do
        echo "Running TTMC for tensor $tensor $method"
        ./compile.sh ttmc $tensor $method &>> ../../graphs/ttmc_mode1.txt
        ./compile.sh ttmc_mode2 $tensor $method &>> ../../graphs/ttmc_mode2.txt
        ./compile.sh ttmc_mode3 $tensor $method &>> ../../graphs/ttmc_mode3.txt
    done
done
