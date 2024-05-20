#!/bin/bash
#SBATCH	--account=YOUR_ACCOUNT_NUMBER                   # (-A) Account/project number
#SBATCH --job-name=shockturb_test   		        # (-J) Job name
#SBATCH	--partition=gp1d		                # (-p) Specific slurm partition
#SBATCH --nodes=1			                # (-N) Maximum number of nodes to be allocated
#SBATCH --ntasks-per-node=8		                # Maximum number of tasks on each node
#SBATCH --cpus-per-task=1                               # Number of CPUs per task
#SBATCH --gres=gpu:8                                    # Number of GPUs per node
#SBATCH --time=24:00:00	 	                        # (-t) Wall time limit (days-hrs:min:sec)
#SBATCH --output=output%j.log		                # (-o) Path to the standard output and error files relative to the working directory
#SBATCH --error=output%j.err		                # (-e) Path to the standard error ouput
#SBATCH --mail-type=ALL 				# Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=YOUR_EMAIL_ADDRESS 	# Where to send mail.  Set this to your email address

module purge
module load cmake/3.23.2 cuda/11.7 gcc9/9.3.1 mpich/4.1.2-mlnx5 

srun ./ac_run_mpi --config astaroth.conf --init-condition ShockTurb

