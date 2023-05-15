#!/bin/bash
#SBATCH --account=project_462000190
#SBATCH --partition=dev-g
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --time=00:10:00

module purge
module load CrayEnv
module load PrgEnv-cray
module load rocm

export MPICH_GPU_SUPPORT_ENABLED=0 # Temporary workaround, remember to set USE_CUDA_AWARE_MPI=OFF when compiling

#srun --cpu-bind=map_cpu:48,56,16,24,1,8,32,40 ./ac_run_mpi --config ~/astaroth/pilot/astaroth-128.conf --run-init-kernel
srun --cpu-bind=map_cpu:48,56,16,24,1,8,32,40 ./ac_run_mpi --config ~/astaroth/pilot/astaroth-128.conf --from-snapshot
#srun --cpu-bind=map_cpu:48,56,16,24,1,8,32,40 ./ac_run_mpi --config ~/astaroth/pilot/astaroth.conf --run-init-kernel
# srun --cpu-bind=map_cpu:48,56,16,24,1,8,32,40 ./ac_run_mpi --config ./astaroth.conf --from-pc-varfile=/scratch/project_462000120/jpekkila/mahti-512-varfile/var.dat # From varfile
# srun --cpu-bind=map_cpu:48,56,16,24,1,8,32,40 ./ac_run_mpi --config ./astaroth.conf --from-snapshot # From snapshot
# srun --cpu-bind=map_cpu:48,56,16,24,1,8,32,40 ./mpitest 64 64 64 # Autotest
