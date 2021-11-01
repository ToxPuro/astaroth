#!/bin/bash

#SBATCH --account=Project_2000403
#SBATCH --time 00:06:00
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=0
#SBATCH -p gpu
##SBATCH --mail-type=BEGIN #uncomment to enable mail


set -e # just soldier on, some runs will fail...
SRC=/users/julianlagg/astaroth/rundir/hydrotest
REP=/users/julianlagg/analysis/report.txt
FORCE=0.155
VISC=0.0045
echo "type     i      j       time " > $REP
start_time=`date +%s`
j=1

for i in {1..30}
do
    #rm -rf /scratch/project_2000403/lyapunov/*
    srun -o output_$j.txt  python3 $SRC/simul.py --forcing=$FORCE --visc=$VISC --hel=0.5 --ana_dir_name=hel_half_A_$j --fixedseed=500$j
    current_time=`date +%s`
    echo $i $j `expr $current_time - $start_time` >> $REP
    sleep 1
    let "j+=1"
done
for i in {1..5}
do
    #rm -rf /scratch/project_2000403/lyapunov/*
    srun -o output_$j.txt $SRC/simul.py --forcing=$FORCE --visc=$VISC --hel=0.0 --ana_dir_name=hel_zero_A_$j --fixedseed=500$j
    current_time=`date +%s`
    echo $i $j `expr $current_time - $start_time` >> $REP
    sleep 1
    let "j+=1"
done
for i in {1..5}
do
    #rm -rf /scratch/project_2000403/lyapunov/*
    srun -o output_$j.txt python3 $SRC/simul.py --forcing=$FORCE --visc=$VISC --hel=1.0 --ana_dir_name=hel_one_A_$j --fixedseed=500$j
    current_time=`date +%s`
    echo $i $j `expr $current_time - $start_time` >> $REP
    sleep 1
    let "j+=1"
done
end_time=`date +%s`
echo execution time was `expr $end_time - $start_time` s.

