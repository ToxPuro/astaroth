from mpi4py import MPI
import mpitest

mpitest.init(0)

mpitest.makeseries(1.0)

mpitest.barrier()
mpitest.print()

mpitest.barrier()
mpitest.copyval(0)

mpitest.barrier()
mpitest.print()
