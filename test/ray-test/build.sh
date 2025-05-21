#!/bin/bash
#
mkdir -p build && cd build && cmake -DMAX_THREADS_PER_BLOCK=512 -DMPI_ENABLED=ON -DOPTIMIZE_MEM_ACCESSES=ON -DBUILD_TESTS=ON -DDSL_MODULE_DIR=../DSL $AC_HOME && make ray-test -j
