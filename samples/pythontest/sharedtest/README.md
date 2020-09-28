# Compiling OpenMPI 

Will only with of `--disable-dlopen` option is used when compiling OpenMPI

Compilation after downloading the source code
```
tar -xzvf openmpi_source_code.tar.gz 
cd openmpi-4.0.5
./configure --disable-dlopen --prefix=/path/to/openmpi/
make all install
```
In setupmpi.py set

```
mpi_compile_args = os.popen("/path/to/openmpi/bin/mpicc --showme:compile").read().strip().split(' ')
mpi_link_args    = os.popen("/path/to/openmpi/bin/mpicc --showme:link").read().strip().split(' ')
```
To build run 

`./buildmpi.sh`

`/path/to/openmpi/bin/mpirun -np 4 python mpitest.py`

Will give output like 

```
Initializing nprocs = 4, pid = 0, myglobal[0] = 0.000000 
Initializing nprocs = 4, pid = 2, myglobal[0] = 32.000000 
Initializing nprocs = 4, pid = 3, myglobal[0] = 48.000000 
Initializing nprocs = 4, pid = 1, myglobal[0] = 16.000000 
Array, pid = 0, [ 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0 ] 
pid 0, send_ind 1, recv_ind 3 
Array, pid = 2, [ 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0 ] 
pid 2, send_ind 3, recv_ind 1 
Array, pid = 1, [ 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0 ] 
pid 1, send_ind 2, recv_ind 0 
Array, pid = 3, [ 48.0, 49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0 ] 
pid 3, send_ind 0, recv_ind 2 
Array, pid = 0, [ 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 48.0 ] 
Array, pid = 3, [ 48.0, 49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 32.0 ] 
Array, pid = 1, [ 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 0.0 ] 
Array, pid = 2, [ 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 16.0 ] 

```

#How to do this with mpi4py

install mpi4py with 

`env MPICC=/path/to/mpicc pip install mpi4py`

E.g. 

`env MPICC=/usr/bin/mpicc pip install mpi4py`

Othewise libraries do not link correctly! Installing from mpi4py from anaconda can cause issues. 

To build run 

`./buildmpi.sh`

To run. 

`mpirun -np 4 python mpitest.py`
