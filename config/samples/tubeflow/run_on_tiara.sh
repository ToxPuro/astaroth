#!/bin/bash

#mpirun -np 4 ./ac_run_mpi --config astaroth.conf --init-condition HeatDuct

( mpirun -np 4 ./ac_run_mpi --config astaroth.conf --init-condition HeatDuct >& output.log & )

