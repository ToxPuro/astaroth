#!/bin/bash
#
target=$(basename "${PWD}")
cmake -B build -S $AC_HOME -DCMAKE_CXX_FLAGS="-O0 -Og" -DCMAKE_BUILD_TYPE=Debug -DMPI_ENABLED=ON -DOPTIMIZE_MEM_ACCESSES=ON -DBUILD_TESTS=ON -DDSL_MODULE_DIR=../DSL && cmake --build build -t $target -j
