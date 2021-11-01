#!/bin/bash

set -e # just soldier on, some runs will fail...
SRC=/users/julianlagg/astaroth/rundir/hydrotest
REP=/users/julianlagg/analysis/report.txt

FORCE=0.08
VISC=0.0025

start_time=`date +%s`

j=0
echo "i  time" > $REP

for i in {1..40}
do
    for v in 0.0015 0.002 0.0025 0.003 0.0035 0.004 
    do
        for force in  100 105 110 115 120
        do
            for hel in 0
            do
                out_name=mass_simuls_helzero_fixed_pipes_init-$j
                srun --account=Project_2000403 --gres=gpu:v100:1 --mem=0 -t 00:10:00 -p gpu python3 $SRC/simul.py --forcing=0.$force --visc=$v --hel=0.$hel --ana_dir_name=$out_name --fixedseed=119$j 1>$out_name.out 2>$out_name.err &
                let "j+=1"
            done
        done
    done
    wait
    echo $i ";" $(date) >> $REP
    python3 $SRC/clean_out_files.py
done

end_time=`date +%s`
echo execution time was `expr $end_time - $start_time` s.

