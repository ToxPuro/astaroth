#!/bin/bash
#SBATCH --account=project_2004753
#SBATCH --partition=gputest
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:4
#SBATCH --ntasks-per-node=5
#SBATCH --output=slurm-astaroth-soma-%j.out
#SBATCH --error=slurm-astaroth-soma-%j.err

set -eu

echo "Starting SOMA Collectors..."
export SOMA_SERVER_ADDR_FILE=`pwd`/server.add
export SOMA_NODE_ADDR_FILE=`pwd`/node.add
export SOMA_NUM_SERVER_INSTANCES=1
export SOMA_NUM_SERVERS_PER_INSTANCE=$SLURM_NNODES
export SOMA_SERVER_INSTANCE_ID=0

#Make sure number of SOMA servers is SOMA_NUM_SERVER_INSTANCES * SOMA_NUM_SERVERS_PER_INSTANCE
SERVERS_PER_NODE=1
CLIENTS_PER_NODE=$(( SLURM_NTASKS_PER_NODE - 1 ))

srun ./wrapper.sh $SERVERS_PER_NODE $CLIENTS_PER_NODE

