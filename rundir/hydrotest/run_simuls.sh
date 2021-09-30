#!/bin/bash

set -e # just soldier on, some runs will fail...
SRC=/users/julianlagg/astaroth/rundir/hydrotest

start_time=`date +%s`
j=1
for i in 1
do
    for f in $(python -c "import numpy as np; print(' '.join(list(map(str,np.linspace(0.01,0.2,15)))))") 
    do
        for v in $(python -c "import numpy as np; print(' '.join(list(map(str,np.linspace(0.0001,0.02,15)))))") 
        do
            rm -rf /scratch/project_2000403/lyapunov/*
            srun --account=Project_2000403 --gres=gpu:v100:4 --mem=24000 -t 00:05:00 -p gputest python3 $SRC/simul.py --forcing=$f --visc=$v --hel=1.0 --ana_dir_name=old_wave_$j --fixedseed=500$j
            sleep 1
            let "j+=1"
        done
    done
done

end_time=`date +%s`
echo execution time was `expr $end_time - $start_time` s.

