#!/bin/bash
#SBATCH --account=project_462000120
#SBATCH --gres=gpu:8
#SBATCH --partition=dev-g
#SBATCH --ntasks-per-node=8
#SBATCH --nodes=8
#SBATCH --time=00:10:00

module purge

module load CrayEnv
module load PrgEnv-cray
module load craype-accel-amd-gfx90a
module load rocm
module load buildtools
module load cray-python

export MPICH_GPU_SUPPORT_ENABLED=1

srun ./ac_run_mpi --config ./astaroth.conf --from-pc-varfile=/scratch/project_462000120/jpekkila/mahti-512-varfile/var.dat # From varfile
# srun ./ac_run_mpi --config ./astaroth.conf --from-snapshot # From snapshot
