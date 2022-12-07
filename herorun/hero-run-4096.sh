#!/bin/bash
#SBATCH --account=project_462000120
#SBATCH --gres=gpu:8
#SBATCH --partition=pilot
#SBATCH --ntasks=4096
#SBATCH --nodes=512
#SBATCH --time=00:59:59

module purge

module load CrayEnv
module load PrgEnv-cray
module load craype-accel-amd-gfx90a
module load rocm
module load buildtools
module load cray-python

srun ./ac_run_mpi --config /users/pekkila/astaroth/config/astaroth-hero-run.conf
