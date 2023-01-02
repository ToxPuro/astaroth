#!/bin/bash
#SBATCH --account=project_462000120
#SBATCH --gres=gpu:8
#SBATCH --partition=dev-g
#SBATCH --ntasks=4096
#SBATCH --nodes=512
#SBATCH --time=00:59:00

module purge

module load CrayEnv
module load PrgEnv-cray
module load craype-accel-amd-gfx90a
module load rocm
module load buildtools
module load cray-python

export MPICH_GPU_SUPPORT_ENABLED=1

srun ./ac_run_mpi --config ./astaroth.conf --from-pc-varfile=/flash/project_462000120/pilot_experiments/data/var.4096.dat # From varfile
# srun ./ac_run_mpi --config ./astaroth.conf --from-snapshot # From snapshot
