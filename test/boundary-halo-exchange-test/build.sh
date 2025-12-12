#!/bin/bash
#
mkdir -p build && cd build && cmake -DUSE_HEFFTE=OFF -DMPI_ENABLED=ON -DFFT_ENABLED=ON -DOPTIMIZE_MEM_ACCESSES=ON -DBUILD_TESTS=ON -DRUNTIME_COMPILATION=OFF -DDSL_MODULE_DIR=../DSL $AC_HOME && make boundary-halo-exchange-test -j
