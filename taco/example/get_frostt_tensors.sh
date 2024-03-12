#!/bin/bash
mkdir -p data_frostt
wget https://s3.us-east-2.amazonaws.com/frostt/frostt_data/nell/nell-1.tns.gz -O data_frostt/nell-1.tns.gz
wget https://s3.us-east-2.amazonaws.com/frostt/frostt_data/nell/nell-2.tns.gz -O data_frostt/nell-2.tns.gz
wget https://s3.us-east-2.amazonaws.com/frostt/frostt_data/flickr/flickr-3d.tns.gz -O data_frostt/flickr-3d.tns.gz
wget https://s3.us-east-2.amazonaws.com/frostt/frostt_data/vast-2015-mc1/vast-2015-mc1-3d.tns.gz -O data_frostt/vast-2015-mc1-3d.tns.gz
cd ./data_frostt
gunzip *.tns.gz
