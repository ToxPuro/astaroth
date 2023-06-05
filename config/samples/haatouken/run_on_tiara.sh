#!/bin/bash

#mpirun -np 4 ./ac_run_mpi --config astaroth.conf --init-condition Haatouken

( mpirun -np 4 ./ac_run_mpi --config astaroth.conf --init-condition Haatouken >& output.log & )

