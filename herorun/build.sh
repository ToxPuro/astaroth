#!/usr/bin/bash

#ASTAROTH_DIR="$(dirname "$0")/.."
#echo $ASTAROTH_DIR

#if [[ -z "$AC_HOME" ]]; then
#    echo "AC_HOME not defined. Need to run `source path/to/astaroth/sourceme.sh`."
#    exit 1
#fi

#echo $AC_HOME
#cmake -DUSE_HIP=ON -DMPI_ENABLED=ON -DSINGLEPASS_INTEGRATION=OFF -DOPTIMIZE_MEM_ACCESSES=ON $AC_HOME && make -j

# ENSURE $ASTAROTH is set!
cmake -DUSE_HIP=ON -DMPI_ENABLED=ON -DUSE_CUDA_AWARE_MPI=OFF -DSINGLEPASS_INTEGRATION=OFF -DOPTIMIZE_MEM_ACCESSES=ON $ASTAROTH && make -j
./ac_run_mpi --config $CONFIGDIR