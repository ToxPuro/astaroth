#!/bin/bash -l
#SBATCH --partition=dev-g# Partition (queue) name
#SBATCH --output=test.out
##SBATCH --partition=gpupilot # Partition (queue) name
##SBATCH --partition=gpularge # Partition (queue) name
#SBATCH --nodes=1 # Total number of nodes 
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:30:00       # Run time (d-hh:mm:ss)
#SBATCH --account=project_462001514
#SBATCH --cpus-per-task=7
#SBATCH --mem=0

srun $AC_HOME/PencilA-setups/Coala/build/ac_run_mpi --config $AC_HOME/PencilA-setups/Coala/config/astaroth.conf  --run-init-kernel randomize

