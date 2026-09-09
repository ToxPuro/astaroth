#!/bin/bash
#
cmake -B build -S $AC_HOME -DCMAKE_BUILD_TYPE=Debug -DMPI_ENABLED=ON -DOPTIMIZE_MEM_ACCESSES=ON -DFUSE_KERNELS=ON -DBUILD_TESTS=ON -DDSL_MODULE_DIR=../DSL && cmake --build build -t kernel-fusion-test -j
