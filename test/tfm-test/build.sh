#!/bin/bash

# http://redsymbol.net/articles/unofficial-bash-strict-mode/
set -euo pipefail
IFS=$'\n\t'

cmake -S $AC_HOME -B build -DCMAKE_BUILD_TYPE=Debug -DMPI_ENABLED=ON -DOPTIMIZE_MEM_ACCESSES=ON -DRUNTIME_COMPILATION=ON -DBUILD_TESTS=ON -DDSL_MODULE_DIR=../DSL
cmake --build build -t tfm-test -j
