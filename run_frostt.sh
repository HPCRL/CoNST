#!/bin/bash
# first we make the data
cd taco/example
./get_frostt_tensors.sh
# run the code
./run_mttkrp.sh
./run_ttmc.sh
cd ../..
# process the logs to make graphs
cd graphs
python cpd.py
python ttmc.py
cd ..
