#!/bin/bash
# first we make the data
cd taco/example
./make_3cint_data.sh
cd ../..
# now generate code
python test_tree_3c.py
python test_tree_4c.py
# run the code
cd taco/example
./run_mp2.sh
cd ../..
# process the logs to make graphs
cd graphs
python mp2_3c.py
python mp2_4c.py
cd ..
