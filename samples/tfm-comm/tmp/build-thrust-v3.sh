#!/usr/bin/env bash
#clang++ -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP -std=c++17 -x c++ ../tmp/thrust-test-v3.cu -I/Users/pekkilj1/repositories/cccl-main/thrust
clang++ -std=c++17 -x c++ -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP -I${THRUST_INCLUDE_DIR} -I${CUDACXX_INCLUDE_DIR} -I${CUB_INCLUDE_DIR} ../tmp/thrust-test-v3.cu
