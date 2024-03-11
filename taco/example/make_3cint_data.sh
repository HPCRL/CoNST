#!/bin/bash
# Make medium data if it doesn't exist
if [ ! -f ./data_qc_med/*.tns ]; then
    echo "Generating 3cint medium data"
    cd data_qc_med
    python make_d.py
    python untile.py
    cd ..
fi
if [ ! -f ./data_qc_large/*.tns ]; then
    echo "Generating 3cint large data"
    cd data_qc_large
    python make_d.py
    python untile.py
    cd ..
fi
