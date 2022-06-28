#!/bin/bash
# Run as `. ./build.sh` to run in the current environment
CURR_DIR=$(pwd)
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ACC_RUNTIME_DIR=${SCRIPT_DIR}/..
PARAMS=${@:1:$#-1}

mkdir -p tmpbuild
cd tmpbuild
cmake $PARAMS $ACC_RUNTIME_DIR
echo "static int stencils_accessed[NUM_KERNELS][NUM_FIELDS][NUM_STENCILS] = {{{0}}};" > api/stencil_accesses.h
make -j
sed -i 's/__global__//g' api/user_kernels.h
sed -i 's/__device__//g' api/user_kernels.h
sed -i 's/#define \([a-z_0-9]*\)(field) (processed_stencils\[(field)\]\[stencil_[a-z_0-9]*\])/auto \1=[\&](const auto field){stencils_accessed\[field\][stencil_\1]=true;return AcReal(1.0);};/g' api/user_kernels.h
cd ..

gcc -Wfatal-errors ${ACC_RUNTIME_DIR}/stencil-access-generator/main.cpp -I tmpbuild/api -I ${ACC_RUNTIME_DIR}/api -lm -o stencil-access-generator
./stencil-access-generator stencil_accesses.h
mkdir -p acc-runtime/api
mv stencil_accesses.h acc-runtime/api
cmake $@