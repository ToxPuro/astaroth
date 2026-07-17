#!/bin/bash

MAX_N=$1
rm -f *.dat
for ((N=128; N<=MAX_N; N*=2)); do
    ./build/integration-test $N $N
done
