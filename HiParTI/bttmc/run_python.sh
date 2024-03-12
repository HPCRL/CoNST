#!/bin/bash

DATA_PATH=/home/yufan/data_frostt_sparta

python $(pwd -P)/generate_tns_scipy.py -t $DATA_PATH/nell-1.tns -c $(pwd -P)/ -b $(pwd -P)/exp_x1.sh

# python $(pwd -P)/generate_tns_scipy.py -t $DATA_PATH/nell-2.tns -c $(pwd -P)/ -b $(pwd -P)/exp_x1.sh

# python $(pwd -P)/generate_tns_scipy.py -t $DATA_PATH/flickr-3d.tns -c $(pwd -P)/ -b $(pwd -P)/exp_x1.sh

# python $(pwd -P)/generate_tns_scipy.py -t $DATA_PATH/vast-2015-mc1-3d.tns -c $(pwd -P)/ -b $(pwd -P)/exp_x1.sh
