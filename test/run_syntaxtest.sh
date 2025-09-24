#!/bin/bash
mkdir -p acc-runtime/build
cd acc-runtime/build
cmake -DOPTIMIZE_MEM_ACCESSES=ON -DUSE_HIP=OFF -DBUILD_ACC_RUNTIME_LIBRARY=ON .. && make -j
cd acc
$AC_HOME/acc-runtime/tests/syntaxtest.sh
