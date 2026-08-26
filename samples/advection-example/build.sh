#!/bin/bash

# http://redsymbol.net/articles/unofficial-bash-strict-mode/
set -euo pipefail
IFS=$'\n\t'

cmake -S $AC_HOME -B build -DMPI_ENABLED=ON -DOPTIMIZE_MEM_ACCESSES=ON -DBUILD_ADVECTION_EXAMPLE=ON -DDSL_MODULE_DIR=../DSL
cmake --build build -t advection-example -j
