# Astaroth TFM-MPI

## Overview

C++ implementation: `astaroth/samples/tfm-mpi/tfm.cc`

DSL implementation: `astaroth/samples/tfm/mhd/mhd.ac`. Note that the directory is `tfm`, not `tfm-mpi`

Config file: `astaroth/samples/tfm/mhd/mhd.ini`. Note also here the directory `tfm` instead of `tfm-mpi`.


## Building

### Modules
Puhti:
```bash
module load gcc/11.3.0 openmpi/4.1.4-cuda cuda cmake
```
Should have
```
Currently Loaded Modules:
  1) csc-tools (S)   3) gcc/11.3.0                  5) openmpi/4.1.4-cuda   7) cmake/3.23.1
  2) StdEnv          4) intel-oneapi-mkl/2022.1.0   6) cuda/11.7.0
```

LUMI:
```bash
module load PrgEnv-cray
module load craype-accel-amd-gfx90a
module load rocm
module load cray-python
module load cray-hdf5
module load LUMI/24.03 buildtools/24.03 # New
module load craype-accel-amd-gfx90a # New, must be loaded after LUMI/24.03
export MPICH_GPU_SUPPORT_ENABLED=1 # Note here
```
Should have
```bash
Currently Loaded Modules:
  1) craype-x86-rome                        7) ModuleLabel/label (S)  13) cray-libsci/24.03.0
  2) libfabric/1.15.2.0                     8) lumi-tools/24.05  (S)  14) PrgEnv-cray/8.5.0
  3) craype-network-ofi                     9) init-lumi/0.2     (S)  15) craype-accel-amd-gfx90a
  4) perftools-base/24.03.0                10) craype/2.7.31.11       16) rocm/6.0.3
  5) xpmem/2.8.2-1.0_5.1__g84a27a5.shasta  11) cray-dsmml/0.3.0       17) cray-python/3.11.7
  6) cce/17.0.1                            12) cray-mpich/8.1.29      18) cray-hdf5/1.12.2.11
```

### Commands
There's a build script in `astaroth/samples/tfm-mpi/build.sh`.

**IMPORTANT:** the build script relies on `$ASTAROTH` env variable being set to the base `astaroth` directory of the `2024-09-02-tfm-standalone` branch.

For example:
```bash
cd astaroth
export ASTAROTH=$(pwd) # Note here (can add also to .bashrc)
mkdir build && cd build
../samples/tfm-mpi/build.sh
```

The build system determines automatically whether it should compile for AMD or Nvidia.

## Setting up parameters

### Magnetic parameters

The default `AC_eta` and TFM-specific `AC_eta_tfm` are available in `astaroth/samples/tfm/mhd/mhd.ini`.

### B-field profile amplitude and wavenumber

See `AC_profile_amplitude` and `AC_profile_wavenumber` in `astaroth/samples/tfm/mhd/mhd.ini`.

### Simulation steps
See `AC_simulation_nsteps` and `AC_simulation_output_interval` in `astaroth/samples/tfm/mhd/mhd.ini`.

### Forcing

Forcing is currently always on and a new forcing vector generated at the start of each iteration. 

## Running

- The executable name is `tfm-mpi`.
- The program expects one (1) MPI process per GPU.
- There are eight (8) GPUs per node on LUMI.

LUMI
```bash
export SRUNMPI2="srun --account=<project number> -t 00:05:00 -p dev-g --gpus-per-node=1 --ntasks-per-node=1 --nodes=1" # 1 GPU
export SRUNMPI2="srun --account=<project number> -t 00:05:00 -p dev-g --gpus-per-node=2 --ntasks-per-node=2 --nodes=1" # 2 GPUs
export SRUNMPI4="srun --account=<project number> -t 00:05:00 -p dev-g --gpus-per-node=4 --ntasks-per-node=4 --nodes=1" # 4 GPUs
export SRUNMPI8="srun --account=<project number> -t 00:05:00 -p dev-g --gpus-per-node=8 --ntasks-per-node=8 --nodes=1" # 8 GPUs
export SRUNMPI16="srun --account=<project number> -t 00:05:00 -p small-g --gpus-per-node=8 --ntasks-per-node=8 --nodes=2" # 16 GPUs, 2 nodes, note small-g instead of dev-g

$SRUNMPI8 ./tfm-mpi
```

## Visualizing output

- Outputs are monolithic files holding the computational domain
- Halos are not included in the monolithic snapshots
- `astaroth/samples/tfm-mpi/visualize-debug.py` can be used as a starting point for visualizing the snapshots and profiles. Currently hacked together, so have to manually set mesh dimensions, and execute only specific jupyter cells in the script (does not likely run from the command line). I have used mainly the code blocks `# Plot collective` around line 40 and `# Profiles` around line 120.
- All files are written out as double-precision numbers in the ordering $x$-$y$-$z$ ($x$ fastest varying dimension). For example, the first 8 bytes correspond to the first double-precision value at index $(0, 0, 0)$ in the computational domain, the second corresponds to index $(1, 0, 0)$, etc.

Debug output is in the format
```bash
debug-step-000000000060-tfm-PROFILE_ucrossb21mean_x.profile # Monolithic profiles, step number 60, contains global_nz elements 
debug-step-000000000060-tfm-TF_a12_y.mesh # Monolithic field snapshots, step number 60, contains (global_nx, global_ny, global_nz) elements
proc-7-debug-step-000000000090-tfm-TF_a12_z.mesh # Distributed field snapshot of process 7 (includes halos but they are not updated before writing to disk so they can contain garbage). Contains (local_mx, local_my, local_mz) elements
```

Test field handle and output file names
```
// Test fields
Field TF_a11_x, TF_a11_y, TF_a11_z
Field TF_a12_x, TF_a12_y, TF_a12_z
Field TF_a21_x, TF_a21_y, TF_a21_z
Field TF_a22_x, TF_a22_y, TF_a22_z
```
defined in the beginning of tfm/mhd/mhd.ac.

Profile handle and output file names
```
// Mean-field profiles
Profile PROFILE_Umean_x, PROFILE_Umean_y, PROFILE_Umean_z

Profile PROFILE_ucrossb11mean_x, PROFILE_ucrossb11mean_y, PROFILE_ucrossb11mean_z
Profile PROFILE_ucrossb12mean_x, PROFILE_ucrossb12mean_y, PROFILE_ucrossb12mean_z
Profile PROFILE_ucrossb21mean_x, PROFILE_ucrossb21mean_y, PROFILE_ucrossb21mean_z
Profile PROFILE_ucrossb22mean_x, PROFILE_ucrossb22mean_y, PROFILE_ucrossb22mean_z

Profile PROFILE_B11mean_x, PROFILE_B11mean_y, PROFILE_B11mean_z
Profile PROFILE_B12mean_x, PROFILE_B12mean_y, PROFILE_B12mean_z
Profile PROFILE_B21mean_x, PROFILE_B21mean_y, PROFILE_B21mean_z
Profile PROFILE_B22mean_x, PROFILE_B22mean_y, PROFILE_B22mean_z
```

Profiles are named in format `Profile PROFILE_B12mean_x`, which corresponds to the $x$ component of the $\overline{B}^{12}$-field.
These correspond to "Eq. 15 in "Scale dependence of alpha effect and turbulent diffusivity", Brandenburg, Rädler, and Schrinner, 2018, https://arxiv.org/pdf/0801.1320. Here $B^{1c}$ (Brandenburg) corresponds to $B^{11}$ (Astaroth).
