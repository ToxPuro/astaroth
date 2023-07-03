#!/bin/bash
#SBATCH --account=project_2000403
#SBATCH --partition=gputest
#SBATCH --nodes=2
#SBATCH --gres=gpu:v100:4
#SBATCH --ntasks-per-socket=2
#SBATCH -n 8
#SBATCH --exclusive
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=2

module purge
module load gcc cuda openmpi/4.1.4-cuda cmake python-data

export MPICH_GPU_SUPPORT_ENABLED=1

srun ./ac_run_mpi -c ../pilot/astaroth-128.conf &
main_pid=$!

sleep 5
if [[ $SLURM_PROCID -eq 0 ]]; then
    while :; do


        # Check if running. Note: assumes that $SLURM_JOB_ID.0 is the process we want to monitor
        #sync
        if sacct -j $SLURM_JOB_ID.0 -o "State" --noheader | grep -e RUNNING -e PENDING &> /dev/null; then
            num_dirs=$(find output-slices/* -type d | wc -l)
            while [[ $num_dirs -gt 1 ]]; do
                # Find the oldest directory
                dir=$(find output-slices/* -type d -printf '%p\n' | sort | head -n 1)
                $ASTAROTH/analysis/viz_tools/render_slices.py --input ${dir}/* --termcolor off --write-bin --no-write-png
                rm -rf ${dir}
                num_dirs=$(find output-slices/* -type d | wc -l)
            done
        else
            dirs=$(find output-slices/* -type d)
            for dir in $dirs; do
                $ASTAROTH/analysis/viz_tools/render_slices.py --input ${dir}/* --termcolor off --write-bin --no-write-png
                rm -rf ${dir}
            done
            break
        fi

        sleep 1
    done
fi

wait $main_pid
echo "Proc " $SLURM_PROCID " exiting"
