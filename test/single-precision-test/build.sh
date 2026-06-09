#!/bin/bash
#
cmake -B build -S $AC_HOME -DCMAKE_BUILD_TYPE=Debug -DMPI_ENABLED=ON -DOPTIMIZE_MEM_ACCESSES=ON -DBUILD_TESTS=ON -DRUNTIME_COMPILATION=OFF -DDSL_MODULE_DIR=../DSL && cmake --build build -t single-precision-test -j
