#!/bin/bash

#mpirun -np 4 ./ac_run_mpi --config astaroth.conf 

( mpirun -np 4 ./ac_run_mpi --config astaroth.conf >& output.log & )

