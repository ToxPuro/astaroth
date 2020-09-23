python setupmpi.py build

rm mpitest.so

ln -s build/lib.linux-x86_64-3.7/mpitest.cpython-37m-x86_64-linux-gnu.so mpitest.so
