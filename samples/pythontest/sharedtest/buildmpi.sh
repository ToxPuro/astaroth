#IMPORTANT: Link the mpi4py with the MPI library you are otherwise using. Otherwise it will not work! 

python setupmpi.py build

rm mpitest.so

ln -s build/lib.linux-x86_64-3.8/mpitest.cpython-38-x86_64-linux-gnu.so mpitest.so
