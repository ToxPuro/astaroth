#!/bin/bash
#SBATCH --account=project_2004753
#SBATCH --partition=gpumedium
#SBATCH --time=01:00:00
#SBATCH --nodes=2
#SBATCH --gres=gpu:a100:4
#SBATCH --ntasks-per-node=4
#SBATCH --output=slurm-astaroth-soma-%j.out
#SBATCH --error=slurm-astaroth-soma-%j.err

set -eu

#module purge

#module load openmpi/4.1.2-cuda
#module load cuda

echo "Starting SOMA Collectors..."
export SOMA_SERVER_ADDR_FILE=`pwd`/server.add
export SOMA_NODE_ADDR_FILE=`pwd`/node.add
export SOMA_NUM_SERVER_INSTANCES=1
export SOMA_NUM_SERVERS_PER_INSTANCE=4
export SOMA_SERVER_INSTANCE_ID=0

#Make sure number of SOMA servers is SOMA_NUM_SERVER_INSTANCES * SOMA_NUM_SERVERS_PER_INSTANCE
srun -n 4 -N 1 ../../../soma-collector-again/build/examples/example-server -a ofi+verbs:// &

sleep 10

echo "Starting Astaroth"

srun -n 4 -N 1 --label /scratch/project_2004753/lappiosk/soma_paper/astaroth/build/ac_run_mpi --config astaroth.conf --run-init-kernel
