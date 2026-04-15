#!/bin/bash
#
mkdir -p build && cd build && cmake -DMPI_ENABLED=ON -DOPTIMIZE_MEM_ACCESSES=ON -DBUILD_TESTS=ON -DRUNTIME_COMPILATION=ON -DDSL_MODULE_DIR=../DSL $AC_HOME && make poisson-test -j
