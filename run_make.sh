#!/bin/bash

for i in {1..20}
do
    make submit app=client.py
    if [ $? -eq 0 ]; then
        make clean-model
    else
        echo "make submit failed on iteration $i"
        exit 1
    fi
done
