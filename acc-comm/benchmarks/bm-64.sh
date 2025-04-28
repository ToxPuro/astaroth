#!/usr/bin/env bash
#SBATCH --account=project_462000613
#SBATCH -t 00:10:00
#SBATCH -p standard-g
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --nodes=8

module load PrgEnv-cray
module load craype-accel-amd-gfx90a
module load rocm
module load cray-python
module load cray-hdf5
module load LUMI/24.03 buildtools/24.03
module load craype-accel-amd-gfx90a # Must be loaded after LUMI/24.03

export MPICH_GPU_SUPPORT_ENABLED=1
export SRUN="srun --cpu-bind="map_cpu:49,57,17,25,1,9,33,41""

$SRUN ./benchmarks/bm_rank_reordering 1024 1024 1024 3 100 $SLURM_JOB_ID
$SRUN ./benchmarks/bm_rank_reordering 2048 1024 512 3 100 $SLURM_JOB_ID
#$SRUN ./benchmarks/bm_rank_reordering 1024 1024 512 3 100 $SLURM_JOB_ID
$SRUN ./benchmarks/bm_rank_reordering 512 1024 2048 3 100 $SLURM_JOB_ID
$SRUN ./benchmarks/bm_rank_reordering 4096 2048 128 3 100 $SLURM_JOB_ID

# $SRUN ./bm_pipelining 1024 1024 1024 3 1 100
# $SRUN ./bm_pipelining 1024 1024 1024 3 2 100
# $SRUN ./bm_pipelining 1024 1024 1024 3 4 100
# $SRUN ./bm_pipelining 1024 1024 1024 3 8 100
# $SRUN ./bm_pipelining 1024 1024 1024 3 16 100
# $SRUN ./bm_pipelining 1024 1024 1024 3 32 100

# Strong scaling
# ./tfm-mpi --config /users/pekkila/astaroth/samples/tfm/mhd/mhd.ini --global-nn-override 256,256,256 --job-id $SLURM_JOB_ID

# Weak scaling
# ./tfm-mpi --config /users/pekkila/astaroth/samples/tfm/mhd/mhd.ini --global-nn-override 1024,1024,1024 --job-id $SLURM_JOB_ID
