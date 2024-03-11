#!/bin/bash
# first we make the data
cd taco/example
./make_3cint_data.sh
cd ../..
# now generate code
source source_z3.sh
python test_tree_3c.py
# run the code
cd taco/example
./run_mp2.sh
cd ../..
# process the logs to make graphs
cd graphs
python mp2_3c.py
cd ..
