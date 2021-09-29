#!/bin/bash

set -e # just soldier on, some runs will fail...
SRC=/users/julianlagg/astaroth/rundir/hydrotest

start_time=`date +%s`
j=0
for i in {1..20}
do
    for f in 0.01 0.02 0.04 0.05 0.06 0.09 0.1 0.11 0.12
    do
        for v in 0.002 0.003 0.005 0.007 0.008
        do
            rm -rf /scratch/project_2000403/lyapunov/*
            srun --account=Project_2000403 --gres=gpu:v100:4 --mem=24000 -t 00:15:00 -p gpu python3 $SRC/simul.py --forcing=$f --visc=$v --hel=1.0 --ana_dir_name=sample_$j --fixedseed=500$i
            sleep 1
            echo i=$i
            let "j+=1"
        done
    done
done

end_time=`date +%s`
echo execution time was `expr $end_time - $start_time` s.

