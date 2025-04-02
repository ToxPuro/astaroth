#!/bin/bash
#
mkdir -p build && cd build && cmake -DMPI_ENABLED=ON -DOPTIMIZE_MEM_ACCESSES=ON -D2D=ON -DBUILD_TESTS=ON -DDSL_MODULE_DIR=../DSL $AC_HOME && make 2d-test -j
