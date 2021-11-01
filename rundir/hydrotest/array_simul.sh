#!/bin/bash

#SBATCH --account=Project_2000403
#SBATCH --time 00:01:00
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=0
#SBATCH -p gpu
#SBATCH --array=1-9
#SBATCH --mail-type=BEGIN #uncomment to enable mail

FORCE=0.155
VISC=0.004

i=$SLURM_ARRAY_TASK_ID
if [[$i -gt 6]]
then
RELHEL=0.0
elif [[$i -gt 3]]
then
RELHEL=0.5
else
RELHEL=1.0
fi


srun -o task_id_$SLURM_ARRAY_TASK_ID.txt echo $RELHEL
#srun -o output_$SLURM_ARRAY_TASK_ID.txt python3 simul.py --forcing=$FORCE --visc=$VISC --hel=0.5 --ana_dir_name=array_submission_$SLURM_ARRAY_TASK_ID --fixedseed=500