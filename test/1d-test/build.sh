#!/bin/bash
#
mkdir -p build && cd build && cmake -DCMAKE_BUILD_TYPE=Debug -DMPI_ENABLED=ON -DOPTIMIZE_MEM_ACCESSES=ON -DRUNTIME_COMPILATION=on -DBUILD_TESTS=ON -DDSL_MODULE_DIR=../DSL $AC_HOME && make 1d-test -j
