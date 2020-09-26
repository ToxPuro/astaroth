How to do this:

install mpi4py with 

`env MPICC=/path/to/mpicc pip install mpi4py`

E.g. 

`env MPICC=/usr/bin/mpicc pip install mpi4py`

Othewise libraries do not link correctly! Installing from mpi4py from anaconda can cause issues. 

To build run 

`./buildmpi.sh`

To run. 

`mpirun -np 4 python mpitest.py`
