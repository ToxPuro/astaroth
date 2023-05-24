Created by *Miikka Väisälä*, at 2023/5

# Purpose of this simulation setup 

Purpose if this simualtion sample was to play with more complicate multiple
boundary conditions. Heat flux enters from the bottom boundary on y direction
of the system and matter can exit from the other end. 

# What is a successful test

This is a boundary condition example. By default setting the simulation crashed
when run for longer. However, it serves as an example of how to set up boure
complicated boundary conditions via customizable taskgraph.  

**Please note that this is not a physics test, but a
test to see that the code is working as expected.**

# Required LSWITCHES 

At the moment you need to set the switches manually. Default master branch
configuration will have different LSWITCHES.

* In `../../../acc-runtime/samples/mhd_modular/mhdsolver.ac`

```
LDENSITY (1)
LHYDRO (1)
LMAGNETIC (0)
LENTROPY (1)
LTEMPERATURE (0)
LFORCING (0)
LUPWD (1)
LSINK (0)
LBFIELD (0)
LSHOCK (0)
```

# TaskGraph

This run uses the TaskGraph `Hydro_Heatduct_Solve` as listed in
`samples/standalone_mpi/simulation_taskgraphs.h`. At the moment of writing this
documentation, if you want to set up your own special boundary condition, make
a TaskGraph and call it following the examples int he code. 

# Setting up and compiling.

Run `./my_cmake.sh` or customize it to your system specific. 

# Running the simulation. 

Run e.g. `mpirun -n 4 ./ac_run_mpi --config astaroth.conf --init-condition HeatDuct` 
or however you particular system runs MPI. 

# Troubleshooting

On TIARA currently working on 

 1) cuda/11.3      2) gcc/9.1.0      3) mpich/3.3      4) cmake/3.22.1

OpenMPI/4.0.4 causes stability issues. 

It the case you get strange MPI errors, it might be that your particular system
has not been configured for GPUDirect RDMA. To run Astaroth without GPUDirect
RDMA, please set `-DUSE_CUDA_AWARE_MPI=OFF` in `my_cmake.sh`. 

On one machine we run into an issue that a wrong version of gcc was found. In
that case please either set flags `-DCMAKE_C_COMPILER=/path/to/gcc/`
`-DCMAKE_CXX_COMPILER=/path/to/gcc/` or set the environmental variables CC and
CXX with a correct path to your compiler. 
