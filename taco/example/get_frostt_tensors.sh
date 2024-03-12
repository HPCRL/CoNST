#!/bin/bash
if [ ! -d "data_frostt" ]; then
  mkdir data_frostt
fi
if [ ! -f "data_frostt/flickr-3d.tns" ]; then
  wget https://s3.us-east-2.amazonaws.com/frostt/frostt_data/flickr/flickr-3d.tns.gz -O data_frostt/flickr-3d.tns.gz
fi
if [ ! -f "data_frostt/vast-2015-mc1-3d.tns" ]; then
  wget https://s3.us-east-2.amazonaws.com/frostt/frostt_data/vast-2015-mc1/vast-2015-mc1-3d.tns.gz -O data_frostt/vast-2015-mc1-3d.tns.gz
fi
if [ ! -f "data_frostt/nell-1.tns" ]; then
  wget https://s3.us-east-2.amazonaws.com/frostt/frostt_data/nell/nell-1.tns.gz -O data_frostt/nell-1.tns.gz
fi
if [ ! -f "data_frostt/nell-2.tns" ]; then
  wget https://s3.us-east-2.amazonaws.com/frostt/frostt_data/nell/nell-2.tns.gz -O data_frostt/nell-2.tns.gz
fi
cd ./data_frostt
gunzip *.tns.gz
cd ..
