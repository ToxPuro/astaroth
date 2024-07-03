Created by *Miikka Väisälä*, at 2021/10
Added Puhti and Mahti options by *Frederick Gent*, at 2024/05

# Purpose of this simulation setup

This simulation was made to test that mpi, shock viscosity and task scheduling
system work together with periodic boundary conditions. The simulation runs
forced isothermal MHD turbulence in conditions where shocks are produced.

# What is a successful test

A succesful test is that `standalone_mpi` runs properly on multiple GPUs
without the system crashing due to unnatural reasons, and results looks sensible
without numerical garbage. **Please note that this is not a physics test, but a
test to see that the code is working as expected.**

# Required LSWITCHES

It is convenient to identify the Astaroth parent directory for running the code
externally to the source files in a local run directory by executing the
following command, which could also be added to your .bashrc
export AC_HOME=/path/to/astaroth/parent/directory/ or by using
`source sourceme.sh` in the Astaroth parent directory.

The provided my_cmake.sh will then copy the files from
$AC_HOME/acc-runtime/samples/mhd_modular/
to a new directory ../DSL relative to the run directory and edit the default
settings in mhdsolver.ac to the following as required for the shocktest.

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

# Setting up and compiling.

Run `./my_cmake.sh`
Cases of successful application of the samples are included for puhti.csc.fi
and tiara.sinica.edu.tw


# Running the simulation.

Run e.g. `mpirun -np 4 ./ac_run_mpi --config astaroth.conf --init-condition ShockTurb`
or however your particular system runs MPI.

# Troubleshooting

On PUHTI currently working with

 1) cmake/3.23.1   2) gcc/11.3.0    3) openmpi/4.1.4-cuda  4) cuda/11.7.0

and on MAHTI currently working with

 1) cmake/3.21.4   2) gcc/11.2.0    3) openmpi/4.1.2-cuda  4) cuda/11.5.0

and on TIARA currently working with

 1) cmake/3.22.1   2) gcc/9.1.0     3) mpich/3.3           4) cuda/11.3

OpenMPI/4.0.4 causes stability issues. Prior to running `my_cmake.sh` the
necessary modules can be invoked by running `source moduleinfo_{MACHINE}'
if running on one of the MACHINES listed above.

In case you get strange MPI errors, it might be that your particular system
has not been configured for GPUDirect RDMA. To run Astaroth without GPUDirect
RDMA, please set `-DUSE_CUDA_AWARE_MPI=OFF` in `my_cmake.sh`.

On some machines MPI IO causes errors so Posix I/O can be enabled by using
`-DUSE_POSIX_IO=ON` in `my_cmake.sh`.

On one machine we run into an issue that a wrong version of gcc was found. In
that case please either set flags `-DCMAKE_C_COMPILER=/path/to/gcc/`
`-DCMAKE_CXX_COMPILER=/path/to/gcc/` or set the environmental variables CC and
CXX with a correct path to your compiler.
