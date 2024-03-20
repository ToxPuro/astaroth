#!/bin/bash

#mpirun -np 4 ./ac_run_mpi --config astaroth.conf --init-condition BoundTest

( mpirun -np 4 ./ac_run_mpi --config astaroth.conf --init-condition BoundTest >& output.log & )

