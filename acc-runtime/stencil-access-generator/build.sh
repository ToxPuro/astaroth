#!/bin/bash
# Run as `. ./build.sh` to run in the current environment
mkdir build
cd build
rm api/user_kernels.h
cmake ../../
echo "static int stencils_accessed[NUM_KERNELS][NUM_FIELDS][NUM_STENCILS] = {{{0}}};" > api/stencil_accesses.h
make -j
#sed -i 's/(processed_stencils\[(field)\]\[stencil_\([a-z_0-9]*\)\])/(stencils_accessed\[(field)\]\[stencil_\1\] = true\)/g' api/user_kernels.h
sed -i 's/__global__//g' api/user_kernels.h
sed -i 's/__device__//g' api/user_kernels.h
sed -i 's/#define \([a-z_0-9]*\)(field) (processed_stencils\[(field)\]\[stencil_[a-z_0-9]*\])/auto \1=[\&](const auto field){stencils_accessed\[field\][stencil_\1]=true;return AcReal(1.0);};/g' api/user_kernels.h
cd ..

gcc -Wfatal-errors  main.cpp -I build/api -I ../api -lm && ./a.out
cp stencil_accesses.h ../../build/acc-runtime/api