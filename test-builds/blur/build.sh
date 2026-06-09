#!/bin/bash
#
cmake -B build -S $AC_HOME -DOPTIMIZE_MEM_ACCESSES=OFF -DMPI_ENABLED=ON -DBUILD_STANDALONE=OFF -DBUILD_MHD_SAMPLES=OFF -DPROGRAM_MODULE_DIR=${AC_HOME}/samples/blur -DDSL_MODULE_DIR=${AC_HOME}/acc-runtime/samples/blur && cmake --build build -j
