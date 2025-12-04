#!/usr/bin/env bash
#SBATCH --account=project_462000987
#SBATCH -t 00:15:00
#SBATCH -p standard-g
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
module list
cmake -LAH >> system_info-$SLURM_JOB_ID.txt

CONFIG="/users/pekkila/astaroth/samples/tfm/mhd/mhd.ini"
cp $CONFIG config-$SLURM_JOB_ID.ini

export MPICH_GPU_SUPPORT_ENABLED=1

# Disabled for TFM tests (change to 'true' to enable)
if true; then
./benchmarks/bm_pack 128 3 3 1 100 $SLURM_JOB_ID
./benchmarks/bm_pack 128 3 3 2 100 $SLURM_JOB_ID
./benchmarks/bm_pack 128 3 3 4 100 $SLURM_JOB_ID
./benchmarks/bm_pack 128 3 3 8 100 $SLURM_JOB_ID
./benchmarks/bm_pack 128 3 3 16 100 $SLURM_JOB_ID
fi

if true; then
./benchmarks/bm_collective_comm 128 128 128 3 1 100
./benchmarks/bm_collective_comm 128 128 128 3 2 100
./benchmarks/bm_collective_comm 128 128 128 3 4 100
./benchmarks/bm_collective_comm 128 128 128 3 8 100
./benchmarks/bm_collective_comm 128 128 128 3 16 100
fi

if true; then
# NOTE: takes a very long time (512^3 grid size, should go for smaller)
./benchmarks/bm_rank_reordering 128 128 128 3 100 $SLURM_JOB_ID
./benchmarks/bm_rank_reordering 256 128 64 3 100 $SLURM_JOB_ID
./benchmarks/bm_rank_reordering 64 128 256 3 100 $SLURM_JOB_ID
./benchmarks/bm_rank_reordering 512 512 8 3 100 $SLURM_JOB_ID
fi