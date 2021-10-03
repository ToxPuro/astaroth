#!/bin/bash

set -e # just soldier on, some runs will fail...
SRC=/users/julianlagg/astaroth/rundir/hydrotest
REP=/users/julianlagg/analysis/report.txt
echo "i      j       time " > $REP
start_time=`date +%s`
j=2
for i in {1..19}
do
    rm -rf /scratch/project_2000403/lyapunov/*
    srun --account=Project_2000403 --gres=gpu:v100:4 --mem=24000 -t 00:15:00 -p gputest python3 $SRC/simul.py --forcing=0.06 --visc=0.0009 --hel=1.0 --ana_dir_name=detailed_A_$j --fixedseed=500$j
    current_time=`date +%s`
    echo $i $j `expr $current_time - $start_time` >> $REP
    sleep 1
    let "j+=1"
done
for i in {1..20}
do
    rm -rf /scratch/project_2000403/lyapunov/*
    srun --account=Project_2000403 --gres=gpu:v100:4 --mem=24000 -t 00:15:00 -p gputest python3 $SRC/simul.py --forcing=0.15 --visc=0.01 --hel=1.0 --ana_dir_name=detailed_B_$j --fixedseed=500$j
    current_time=`date +%s`
    echo $i $j `expr $current_time - $start_time` >> $REP
    sleep 1
    let "j+=1"
done
for i in {1..20}
do
    rm -rf /scratch/project_2000403/lyapunov/*
    srun --account=Project_2000403 --gres=gpu:v100:4 --mem=24000 -t 00:15:00 -p gputest python3 $SRC/simul.py --forcing=0.10 --visc=0.0005 --hel=1.0 --ana_dir_name=detailed_C_$j --fixedseed=500$j
    current_time=`date +%s`
    echo $i $j `expr $current_time - $start_time` >> $REP
    sleep 1
    let "j+=1"
done
for i in {1..20}
do
    rm -rf /scratch/project_2000403/lyapunov/*
    srun --account=Project_2000403 --gres=gpu:v100:4 --mem=24000 -t 00:15:00 -p gputest python3 $SRC/simul.py --forcing=0.025 --visc=0.02 --hel=1.0 --ana_dir_name=detailed_D_$j --fixedseed=500$j
    current_time=`date +%s`
    echo $i $j `expr $current_time - $start_time` >> $REP
    sleep 1
    let "j+=1"
done
end_time=`date +%s`
echo execution time was `expr $end_time - $start_time` s.

