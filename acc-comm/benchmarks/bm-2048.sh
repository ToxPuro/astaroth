#!/usr/bin/env bash
#SBATCH --account=project_462000987
#SBATCH -t 00:15:00
#SBATCH -p standard-g
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --nodes=256

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
export SRUN="srun --cpu-bind="map_cpu:49,57,17,25,1,9,33,41"" # Default mapping
# export SRUN="srun --cpu-bind="map_cpu:33,41,49,57,17,25,1,9"" # Hierarchical mapping (needs 6, 7, 0, 1, 2, 3, 4, 5 rank-device mapping)


# Weak scaling
$SRUN ./tfm-mpi --config $CONFIG --global-nn-override 1024,2048,2048 --job-id $SLURM_JOB_ID  --benchmark 1
