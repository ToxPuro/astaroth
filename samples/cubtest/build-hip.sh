#!/usr/bin/bash
hipcc -DUSE_HIP=1 -std=c++14 --offload-arch=gfx90a cubtest.cu -I$ASTAROTH_PATH/include