#!/bin/bash
#
cmake -B build -S $AC_HOME -DMPI_ENABLED=ON -DOPTIMIZE_MEM_ACCESSES=ON -DBUILD_TESTS=ON -DDSL_MODULE_DIR=test/reduce-sum-add-test/DSL && cmake --build build -t reduce-sum-add-test -j
