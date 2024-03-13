#!/usr/bin/bash
if [[ -z "${ASTAROTH}" ]]; then
    echo "Environment variable ASTAROTH was not set"
    echo "Please 'export ASTAROTH=<path to Astaroth dir>' and run again"
    exit 1
fi

BUILD_OPTIONS="\
-DOPTIMIZE_MEM_ACCESSES=ON \
-DMPI_ENABLED=ON \
-DSINGLEPASS_INTEGRATION=OFF \
-DBUILD_STANDALONE=OFF \
-DBUILD_MHD_SAMPLES=OFF \
-DBUILD_SAMPLES=OFF \
-DDSL_MODULE_DIR=$ASTAROTH/samples/tfm/mhd \
-DPROGRAM_MODULE_DIR=$ASTAROTH/samples/tfm
"

cmake $BUILD_OPTIONS $@ $ASTAROTH && make -j