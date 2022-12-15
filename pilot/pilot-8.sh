#!/bin/bash
#SBATCH --account=project_462000120
#SBATCH --gres=gpu:8
#SBATCH --partition=pilot
#SBATCH --ntasks=8
#SBATCH --nodes=1
#SBATCH --time=00:10:00

module purge

module load CrayEnv
module load PrgEnv-cray
module load craype-accel-amd-gfx90a
module load rocm
module load buildtools
module load cray-python

srun ./ac_run_mpi --config ./astaroth.conf --from-pc-varfile=/scratch/project_462000120/jpekkila/mahti-512-varfile/var.dat
