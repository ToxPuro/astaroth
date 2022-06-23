#!/bin/bash
mkdir build
cd build
rm api/user_kernels.h
cmake ../../
make -j
sed -i 's/(processed_stencils\[(field)\]\[stencil_\([a-z_0-9]*\)\])/(stencils_accessed\[(field)\]\[stencil_\1\] = true\)/g' api/user_kernels.h
cd ..

gcc main.cpp -I build/api -I ../api -lm && ./a.out