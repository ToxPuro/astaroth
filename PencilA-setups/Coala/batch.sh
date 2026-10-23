#!/bin/bash -l
#SBATCH --partition=gpumedium # Partition (queue) name
#SBATCH --output=test.out
#SBATCH --nodes=1 # Total number of nodes 
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:gh200:1
#SBATCH --time=00:30:00       # Run time (d-hh:mm:ss)
#SBATCH --account=project_2016901
#SBATCH --cpus-per-task=7
#SBATCH --mem=0

#Example of running init kernel from DSL
srun $AC_HOME/PencilA-setups/Coala/build/ac_run_mpi --config $AC_HOME/PencilA-setups/Coala/config/astaroth.conf  --run-init-kernel randomize
#Example of reading Pencil data in
#srun $AC_HOME/PencilA-setups/Coala/build/ac_run_mpi --config $AC_HOME/PencilA-setups/Coala/config/astaroth.conf  --from-snapshot ~/pencil-code/samples/0d-tests/coala/data/allprocs

