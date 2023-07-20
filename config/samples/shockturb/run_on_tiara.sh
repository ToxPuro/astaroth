#!/bin/bash

#mpirun -np 4 ./ac_run_mpi --config astaroth.conf --init-condition ShockTurb

( mpirun -np 4 ./ac_run_mpi --config astaroth.conf --init-condition ShockTurb >& output.log & )

