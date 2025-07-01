#!/usr/bin/env bash
#SBATCH --account=project_462000987
#SBATCH -t 00:10:00
#SBATCH -p dev-g
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=2
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

# Strong scaling
$SRUN ./tfm-mpi --config $CONFIG --global-nn-override 128,128,128 --job-id $SLURM_JOB_ID --benchmark 1 --benchmark-name "strong"

# Weak scaling
$SRUN ./tfm-mpi --config $CONFIG --global-nn-override 128,128,256 --job-id $SLURM_JOB_ID --benchmark 1
