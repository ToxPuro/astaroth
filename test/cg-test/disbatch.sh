#!/bin/bash -l
#SBATCH --output=single-gpu
#SBATCH --output=test.out
#SBATCH --partition=dev-g  # Partition (queue) name
#SBATCH --nodes=1 # Total number of nodes 
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1       # Allocate one gpu per MPI rank
#SBATCH --time=00:05:00       # Run time (d-hh:mm:ss)
#SBATCH --account=project_462000759 # Project for billing
#SBATCH --cpus-per-task=7

srun build/cg-test


