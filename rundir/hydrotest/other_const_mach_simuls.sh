#!/bin/bash

set -e # just soldier on, some runs will fail...
SRC=/users/julianlagg/astaroth/rundir/hydrotest
REP=/users/julianlagg/analysis/report.txt


start_time=`date +%s`

j=0

for i in {1..20}
do
    for v in  0.001 0.00125 0.0015 0.00175 0.002 0.00225 0.0025 0.00275 0.003 0.00325 0.0035 0.00375 0.004 0.00425 0.0045 0.00475 0.005
    do
        for offset in 0.080 0.833333 0.0866666 0.090
        do
            f=$(echo "11*$v+$offset" | bc -l)
            out_name=other_const_mach_simuls-$j
            srun --account=Project_2000403 --gres=gpu:v100:1 --mem=0 -t 00:10:00 -p gpu python3 $SRC/simul.py --forcing=$f --visc=$v --hel=0.0 --ana_dir_name=$out_name --fixedseed=1889$j 1>$out_name.out 2>$out_name.err &
            let j+=1
        done
    done
    wait
    echo $i " (other);" $(date) >> $REP
    python3 $SRC/clean_out_files.py
done

end_time=`date +%s`
echo execution time was `expr $end_time - $start_time` s.

