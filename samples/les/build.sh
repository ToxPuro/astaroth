#!/bin/bash
cmake -DBUILD_SAMPLES=OFF -DPROGRAM_MODULE_DIR=samples/les -DDSL_MODULE_DIR=../samples/les .. && make -j