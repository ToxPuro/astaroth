#!/usr/bin/env bash
#SBATCH --account=project_462000613
#SBATCH -t 00:10:00
#SBATCH -p dev-g
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --nodes=1

module load PrgEnv-cray
module load craype-accel-amd-gfx90a
module load rocm
module load cray-python
module load cray-hdf5
module load LUMI/24.03 buildtools/24.03
module load craype-accel-amd-gfx90a # Must be loaded after LUMI/24.03

export MPICH_GPU_SUPPORT_ENABLED=1
export SRUN="srun --cpu-bind="map_cpu:49,57,17,25,1,9,33,41""

#$SRUN ./bm_rank_reordering 512 512 512 3 100 $SLURM_JOB_ID
#$SRUN ./bm_rank_reordering 512 512 256 3 100 $SLURM_JOB_ID
#$SRUN ./bm_rank_reordering 256 512 512 3 100 $SLURM_JOB_ID
#$SRUN ./bm_rank_reordering 1024 1024 64 3 100 $SLURM_JOB_ID

$SRUN ./bm_pipelining 512 512 512 3 1 100
$SRUN ./bm_pipelining 512 512 512 3 2 100
$SRUN ./bm_pipelining 512 512 512 3 4 100
$SRUN ./bm_pipelining 512 512 512 3 8 100
$SRUN ./bm_pipelining 512 512 512 3 16 100
#$SRUN ./bm_pipelining 512 512 512 3 32 100

# Strong scaling
$SRUN ./tfm-mpi --config /users/pekkila/astaroth/samples/tfm/mhd/mhd.ini --global-nn-override 256,256,256 --job-id $SLURM_JOB_ID

# Weak scaling
$SRUN ./tfm-mpi --config /users/pekkila/astaroth/samples/tfm/mhd/mhd.ini --global-nn-override 512,512,512 --job-id $SLURM_JOB_ID
