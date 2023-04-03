#!/bin/bash

if [[ $# -ne 2 ]]; then
    printf "$0 expects two arguments, $# were given"
    exit 1
fi

SERVERS_PER_NODE=$1
CLIENTS_PER_NODE=$2
procid=$SLURM_PROCID

PROCS_PER_NODE=$(( SERVERS_PER_NODE + CLIENTS_PER_NODE ))

is_server(){
    pid=$1
    #In each block, first clients, then servers
    [[ $(( pid % PROCS_PER_NODE )) -ge CLIENTS_PER_NODE ]]
}


if is_server $procid ; then
    echo "Launching proc $procid, a server"
    ../../../soma-collector-again/build/examples/example-server -a ofi+verbs://
    #These all call MPI_Comm_split(color=900)
    echo "Server $procid exiting"
else
    echo "Launching proc $procid, a client"
    /scratch/project_2004753/lappiosk/soma_paper/astaroth/build/ac_run_mpi --config astaroth.conf --run-init-kernel
    #These all call MPI_Comm_split(color=666)
    echo "Client $procid exiting"
fi
