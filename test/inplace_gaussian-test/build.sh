#!/bin/bash
#
cmake -B build -S $AC_HOME -DCMAKE_BUILD_TYPE=Debug -DMPI_ENABLED=ON -DOPTIMIZE_MEM_ACCESSES=ON -DRUNTIME_COMPILATION=off -DBUILD_TESTS=ON -DDSL_MODULE_DIR=../DSL && cmake --build build -t inplace_gaussian-test -j

