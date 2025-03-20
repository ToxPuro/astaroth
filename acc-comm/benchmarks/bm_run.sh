#!/usr/bin/env bash


max_nprocs=128
nprocs_per_node=8
for (( i = 1; i <= max_nprocs; i = 2*i ))
do
	(( ndevices = i <= nprocs_per_node ? i : nprocs_per_node ))
	(( nnodes = (i + nprocs_per_node - 1) / nprocs_per_node ))
	echo "np:       " ${i}
	echo "ndevices: " $ndevices
	echo "nnodes:   " $nnodes
	echo "-"
	mpirun --oversubscribe -n ${i} benchmarks/bm_scaling
done
