#!/bin/bash

# http://redsymbol.net/articles/unofficial-bash-strict-mode/
set -euo pipefail
IFS=$'\n\t'

cmake -S $AC_HOME -B build -DBUILD_INTEGRATOR=ON -DCMAKE_BUILD_TYPE=Release -DMPI_ENABLED=ON -DOPTIMIZE_MEM_ACCESSES=ON -DRUNTIME_COMPILATION=OFF -DALLOW_DEAD_VARIABLES=ON -DBUILD_TESTS=OFF -DDSL_MODULE_DIR=../DSL -DDSL_MODULE_FILE=solver.ac
cmake --build build -t integrator_standalone -j
