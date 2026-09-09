#!/bin/bash
#
cmake -B build -S $AC_HOME -DCMAKE_BUILD_TYPE=Debug -DMPI_ENABLED=ON -DOPTIMIZE_MEM_ACCESSES=ON -DBUILD_TESTS=ON -DDSL_MODULE_DIR=test/field-arr-test/DSL && cmake --build build -t field-arr-test -j
