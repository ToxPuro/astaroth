Created by *Miikka Väisälä*, at 2021/10

# Purpose of this simualtion setup 

This simulation was made to test that mpi, shock viscosity and task scheduling
system work together with periodic boundary conditions. The simulation runs
forced isothermal MHD turbulence in conditions where shocks are produced. 

# What is a successful test

A succesful test is that `standalone_mpi` runs properly on multiple GPUs
without the system crashing due to unnatural resons, and results looks sensible
without numerical garbage. **Please note that this is not a physics test, but a
test to see that the code is working as expected.**

# Required LSWITCHES 

At the moment you need to set the switches manually. Default master branch
configuration will have different LSWITCHES.

## In `acc/mhd_solver/stencil_kernel.ac`

```
LDENSITY (1)
LHYDRO (1)
LMAGNETIC (1)
LENTROPY (0)
LTEMPERATURE (0)
LFORCING (1)
LUPWD (1)
LSINK (0)
LBFIELD (1)
LSHOCK (1)
```

## In `samples/standalone_mpi/main.cc`

```
LSINK (0)
LFORCING (1)
LBFIELD (1)
LSHOCK (1)
```

# Setting up and compiling.

Run `./my_cmake.sh`

# Running the simulation. 

Run e.g. `mpirun -n 4 ./ac_run_mpi -c astaroth.conf` or however you particular
system runs MPI. 

# Troubleshooting

It the case you get strange MPI errors, it might be that your particular system
has not been configured for GPUDirect RDMA. To run Astaroth without GPUDirect
RDMA, please set `-DUSE_CUDA_AWARE_MPI=OFF` in `my_cmake.sh`. 
