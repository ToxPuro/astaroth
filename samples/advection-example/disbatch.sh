#TP: for LUMI change for your cluster
#!/bin/bash -l
#SBATCH --output=test.out
#SBATCH --partition=dev-g  # Partition (queue) name
#SBATCH --nodes=1 # Total number of nodes 
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1       # Allocate one gpu per MPI rank
#SBATCH --time=00:10:00       # Run time (d-hh:mm:ss)
#SBATCH --account=project_462000613 # Project for billing
#SBATCH --mem=2g

export MPICH_GPU_SUPPORT_ENABLED=1
rm -rf *slices*
srun build/advection-example
ac_render_slices --input slices/* --write-png --write-movie --only-lines
