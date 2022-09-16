#!/bin/python3
import os

import socket
hostname = socket.gethostname()

if "mahti" in hostname or "puhti" in hostname:
    build_dir='/users/pekkila/astaroth/build'
    cmake='/users/pekkila/cmake/build/bin/cmake'
elif "triton" in hostname:
    build_dir='/m/home/home6/61/pekkilj1/unix/repositories/astaroth/build'
    cmake='/m/home/home6/61/pekkilj1/unix/repositories/cmake/build/bin/cmake'
else:
    print("Could not recognize the system")
    exit(-1)

cwd = os.getcwd()
if cwd != build_dir:
    print(f"Invalid dir {cwd}. Should be {build_dir}")
    exit(-1)

exit(1)

os.system("rm bwtest-benchmark.csv")

num_fields = 8
num_stencils = 10
num_points_per_stencil = 55
working_set_size = num_fields * num_stencils * num_points_per_stencil

#for r in range(0,working_set_size):
r = 0
while r <= working_set_size:
    os.system(f"sed -i 's/#define HALO ((int)[0-9]*)/#define HALO ((int){r})/g' ../samples/bwtest/bwtest-benchmark.cu")
    os.system(f"{cmake} .. && make -j && ./bwtest-benchmark")
    if r == 0:
        r = 1
    else:
        r = 2 * r
