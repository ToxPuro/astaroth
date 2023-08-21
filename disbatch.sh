#!/bin/bash -l

#SBATCH --account=ituomine

#SBATCH -N 1

#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:v100:1
#SBATCH -p gputest

#SBATCH -t 00:05:00
#SBATCH --cpus-per-task=1
export OMP_NUM_THREADS=1
module load \

gcc/11.3.0 \

intel-oneapi-mkl/2022.1.0 \

openmpi/4.1.4 \

csc-tools \

StdEnv \

cuda/11.7.0
rm -rf output-slices
rm -rf output-snapshots
srun ./build/ac_run_mpi

