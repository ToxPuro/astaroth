#!/bin/bash

# http://redsymbol.net/articles/unofficial-bash-strict-mode/
set -euo pipefail
IFS=$'\n\t'

cmake -S $AC_HOME -B build -DBUILD_SAMPLE=integrator
cmake --build build -t integrator_standalone
