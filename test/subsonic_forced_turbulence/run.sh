mpiexec -n 1 build/ac_run_mpi --run-init-kernel randomize --config ./astaroth.conf
rm -f timeseries.ts
mpiexec -n 1 build/ac_run_mpi --run-init-kernel randomize --config ./astaroth.conf
python3 $AC_HOME/analysis/test_tools/verify.py reference.ts timeseries.ts
