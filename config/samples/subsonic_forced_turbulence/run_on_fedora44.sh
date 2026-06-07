#!/bin/bash

( mpirun -np 1 ./ac_run_mpi --config astaroth.conf --run-init-kernel randomize >& output.log & )

#mpirun -np 1 ./ac_run_mpi --config astaroth.conf --run-init-kernel randomize 

