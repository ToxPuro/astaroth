#!/bin/bash

export CC=/home/kayttaja/gcc15/bin/gcc
export CXX=/home/kayttaja/gcc15/bin/g++
export CUDAHOSTCXX=/home/kayttaja/gcc15/bin/g++

( mpirun -np 1 ./ac_run_mpi --config astaroth.conf --run-init-kernel randomize >& output.log & )

#mpirun -np 1 ./ac_run_mpi --config astaroth.conf --run-init-kernel randomize 

