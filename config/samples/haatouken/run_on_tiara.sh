#!/bin/bash

#mpirun -np 4 ./ac_run_mpi --config astaroth.conf --run-this-init-kernel Haatouken

( mpirun -np 4 ./ac_run_mpi --config astaroth.conf --run-this-init-kernel Haatouken >& output.log & )

