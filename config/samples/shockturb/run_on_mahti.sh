#!/bin/bash
#SBATCH -p gpumedium  #queue
#SBATCH -t 00:14:59  # run time (hh:mm:ss)
#SBATCH -J shocktest	     # job name
#SBATCH -o shocktest.o%j   # output and error file name (%j expands to jobID)
#SBATCH -N 2        # number of nodes
#SBATCH --ntasks-per-socket=2     #
#SBATCH --ntasks-per-node=4     #
#SBATCH --gres=gpu:a100:4     #
#SBATCH --mem=24000     #
#SBATCH --mail-user=fred.gent.ncl@gmail.com
#SBATCH --mail-type=begin  # email me when the job starts
#SBATCH --mail-type=end    # email me when the job finishes
#SBATCH -A ituomine    # project to be billed

LOGFILE="shocktest.out"

srun ./ac_run_mpi --config astaroth.conf --init-condition ShockTurb >>$LOGFILE 2>&1
