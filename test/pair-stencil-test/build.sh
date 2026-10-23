#!/bin/bash
#
cmake -B build -S $AC_HOME -DMPI_ENABLED=ON -DOPTIMIZE_MEM_ACCESSES=ON -DBUILD_TESTS=ON -DRUNTIME_COMPILATION=OFF -DDSL_MODULE_DIR=test/pair-stencil-test/DSL && cmake --build build -t pair-stencil-test -j
