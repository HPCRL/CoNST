#!/bin/bash
mkdir ./taco/build
cd ./taco/build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j64
cd ../..
