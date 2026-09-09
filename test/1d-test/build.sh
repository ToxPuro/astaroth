#!/bin/bash
#
cmake -B build -S $AC_HOME -DCMAKE_BUILD_TYPE=Debug -DMPI_ENABLED=ON -DOPTIMIZE_MEM_ACCESSES=ON -DRUNTIME_COMPILATION=on -DBUILD_TESTS=ON -DDSL_MODULE_DIR=test/1d-test/DSL && cmake --build build -t 1d-test -j
