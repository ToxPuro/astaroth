#!/usr/bin/bash
if [[ -z "${ASTAROTH}" ]]; then
    echo "Environment variable ASTAROTH was not set"
    echo "Please 'export ASTAROTH=<path to Astaroth dir>' and run again"
    exit 1
fi

BUILD_OPTIONS="\
-DOPTIMIZE_MEM_ACCESSES=ON \
-DMPI_ENABLED=ON \
-DSINGLEPASS_INTEGRATION=ON \
-DBUILD_STANDALONE=OFF \
-DBUILD_MHD_SAMPLES=OFF \
-DBUILD_SAMPLES=OFF \
-DDSL_MODULE_DIR=$ASTAROTH/samples/tfm/mhd \
-DPROGRAM_MODULE_DIR=$ASTAROTH/samples/tfm-mpi \
-DBUILD_ACC_RUNTIME_LIBRARY=ON \
-DMAX_THREADS_PER_BLOCK=512 \
" # NOTE CUDA_ARCHITECTURE to build for the work machine
# -DCUDA_ARCHITECTURES=61 \
# -DCUB_PATH=${CUB_PATH} \
# -DTHRUST_PATH=${THRUST_PATH} \

cmake $BUILD_OPTIONS $@ $ASTAROTH && make -j # Build Astaroth
#cmake $BUILD_OPTIONS $@ $ASTAROTH/acc-runtime && make -j # Build ACC runtime only
