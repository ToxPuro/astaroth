#!/bin/bash
#SBATCH --account=project_462000120
#SBATCH --partition=dev-g
#SBATCH --nodes=512
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --time=00:59:00

module purge

module load CrayEnv
module load PrgEnv-cray
module load craype-accel-amd-gfx90a
module load rocm
module load buildtools
module load cray-python

export MPICH_GPU_SUPPORT_ENABLED=1

srun --cpu-bind=map_cpu:48,56,16,24,1,8,32,40 ./ac_run_mpi --config ./astaroth.conf --from-pc-varfile=/flash/project_462000120/pilot_experiments/data/var.4096.dat # From varfile
# srun --cpu-bind=map_cpu:48,56,16,24,1,8,32,40 ./ac_run_mpi --config ./astaroth.conf --from-snapshot # From snapshot
# srun --cpu-bind=map_cpu:48,56,16,24,1,8,32,40 ./mpitest 64 64 64 # Autotest
