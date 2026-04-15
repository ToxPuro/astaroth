# Astaroth standalone - Getting started

## Modules (LUMI, 2025-02-26)

```bash
module load PrgEnv-cray
module load craype-accel-amd-gfx90a
module load rocm
module load cray-python
module load cray-hdf5
module load LUMI/24.03 buildtools/24.03
module load craype-accel-amd-gfx90a # Must be loaded after LUMI/24.03

export MPICH_GPU_SUPPORT_ENABLED=1
```

Should have
```bash
...@uan01:~/astaroth/build> module list

Currently Loaded Modules:
  1) perftools-base/24.03.0       7) cray-dsmml/0.3.0     13) cray-hdf5/1.12.2.11                       19) LUMI/24.03              (S)
  2) cce/17.0.1                   8) cray-mpich/8.1.29    14) craype-x86-rome                           20) buildtools/24.03
  3) ModuleLabel/label      (S)   9) cray-libsci/24.03.0  15) libfabric/1.15.2.0                        21) craype-accel-amd-gfx90a (H)
  4) lumi-tools/24.05       (S)  10) PrgEnv-cray/8.5.0    16) craype-network-ofi
  5) init-lumi/0.2          (S)  11) rocm/6.0.3           17) xpmem/2.8.2-1.0_5.1__g84a27a5.shasta
  6) craype/2.7.31.11            12) cray-python/3.11.7   18) partition/L                          (S)
```

## Building (LUMI, 2025-02-26)
```bash
cd astaroth
mkdir build && cd build
cmake -DMPI_ENABLED=ON -DOPTIMIZE_MEM_ACCESSES=ON -DMAX_THREADS_PER_BLOCK=512 ..
cmake --build . --parallel
```

## Running (LUMI, 2025-02-26)

```bash
# Astaroth expect one process per GPU: 'ntasks-per-node' should match 'gpus-per-node'
# LUMI has 8 GPUs per node. 'ntasks-per-node' is therefore 8 at most.
# Use --nodes to control the number of nodes used
# Can allocate at most 8 GPUs on the dev-g partition. Use small-g for larger runs.
#
# More detailed instructions: https://docs.lumi-supercomputer.eu/runjobs/
#
# For quick tests, should modify the grid resolution in `astaroth/config/astaroth.conf`
# and setting, e.g.,
# AC_nx = 64 # Note: spaces matter here, may not be correctly parsed otherwise
# AC_ny = 64
# AC_nz = 64

# Run on 1 GPU
srun --account=<project id> -t 00:05:00 -p dev-g --gpus-per-node=1 --ntasks-per-node=1 --nodes=1 ./ac_run_mpi

# Run on 4 GPUs
srun --account=<project id> -t 00:05:00 -p dev-g --gpus-per-node=4 --ntasks-per-node=4 --nodes=1 ./ac_run_mpi

# Run on 16 GPUs
srun --account=<project id> -t 00:05:00 -p small-g --gpus-per-node=8 --ntasks-per-node=8 --nodes=2 ./ac_run_mpi
```

## Visualizing

The output is written in `output-snapshots`, `output-slices`, `timeseries.ts`.

There are visualization scripts in `astaroth/analysis`.

For visualizing, e.g., slices,
`../analysis/viz_tools/render_slices.py  --input output-slices/step_000000000100/* --write-png`

# More information
See `astaroth/README.md` and `astaroth/doc/`.
