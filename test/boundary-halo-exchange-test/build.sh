#!/bin/bash
#
cmake -B build -S $AC_HOME -DUSE_HEFFTE=OFF -DMPI_ENABLED=ON -DFFT_ENABLED=ON -DOPTIMIZE_MEM_ACCESSES=ON -DBUILD_TESTS=ON -DRUNTIME_COMPILATION=OFF -DDSL_MODULE_DIR=test/boundary-halo-exchange-test/DSL && cmake --build build -t boundary-halo-exchange-test -j
