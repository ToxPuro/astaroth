#!/usr/bin/env bash
#SBATCH --account=project_462000613
#SBATCH -t 00:10:00
#SBATCH -p dev-g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1

module load PrgEnv-cray
module load craype-accel-amd-gfx90a
module load rocm
module load cray-python
module load cray-hdf5
module load LUMI/24.03 buildtools/24.03
module load craype-accel-amd-gfx90a # Must be loaded after LUMI/24.03

export MPICH_GPU_SUPPORT_ENABLED=1

./bm_pack 256 3 3 1 100 $SLURM_JOB_ID
./bm_pack 256 3 3 2 100 $SLURM_JOB_ID
./bm_pack 256 3 3 4 100 $SLURM_JOB_ID
./bm_pack 256 3 3 8 100 $SLURM_JOB_ID
./bm_pack 256 3 3 16 100 $SLURM_JOB_ID
#./bm_pack 256 3 3 32 100 $SLURM_JOB_ID

#./bm_rank_reordering 256 256 256 3 100 $SLURM_JOB_ID
#./bm_rank_reordering 256 256 128 3 100 $SLURM_JOB_ID
#./bm_rank_reordering 128 256 256 3 100 $SLURM_JOB_ID
#./bm_rank_reordering 512 512 32 3 100 $SLURM_JOB_ID

./bm_pipelining 256 256 256 3 1 100
./bm_pipelining 256 256 256 3 2 100
./bm_pipelining 256 256 256 3 4 100
./bm_pipelining 256 256 256 3 8 100
./bm_pipelining 256 256 256 3 16 100
# ./bm_pipelining 256 256 256 3 32 100

# Strong and weak scaling
./tfm-mpi --config /users/pekkila/astaroth/samples/tfm/mhd/mhd.ini --global-nn-override 256,256,256 --job-id $SLURM_JOB_ID
