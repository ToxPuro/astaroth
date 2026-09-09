#!/bin/bash
#
cmake -B build -S $AC_HOME -DCMAKE_BUILD_TYPE=Debug -DMPI_ENABLED=ON -DOPTIMIZE_MEM_ACCESSES=ON -D2D=OFF -DBUILD_TESTS=ON -DDSL_MODULE_DIR=../DSL && cmake --build build -t uneven-grid-test -j
