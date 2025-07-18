#!/bin/bash
#
target=$(basename "${PWD}")
mkdir -p build && cd build && cmake -DCMAKE_CXX_FLAGS="-O0 -Og" -DCMAKE_BUILD_TYPE=Debug -DMPI_ENABLED=ON -DOPTIMIZE_MEM_ACCESSES=ON -DBUILD_TESTS=ON -DDSL_MODULE_DIR=../DSL $AC_HOME && make $target -j
