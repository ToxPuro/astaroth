#!/home/mvaisala/miniconda3/envs/acpy/bin/python

#from mpi4py import MPI
import mpitest

mpitest.mpiinit()

mpitest.setup(0)

mpitest.makeseries(1.0)

mpitest.barrier()
mpitest.print()

mpitest.barrier()
mpitest.copyval(0)

mpitest.barrier()
mpitest.print()


mpitest.mpifinalize()
