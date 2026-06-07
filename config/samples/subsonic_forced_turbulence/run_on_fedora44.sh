#!/bin/bash

( mpirun -np 1 ./ac_run_mpi --config astaroth.conf >& output.log & )

