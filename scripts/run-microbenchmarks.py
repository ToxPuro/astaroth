#!/bin/python3
import os


#build_dir='/m/home/home6/61/pekkilj1/unix/repositories/astaroth/build'
build_dir='/home/pekkilj1/astaroth/build'
#cmake='/m/home/home6/61/pekkilj1/unix/repositories/cmake/build/bin/cmake'
cmake='/home/pekkilj1/cmake/build/bin/cmake'

cwd = os.getcwd()
if cwd != build_dir:
    print(f"Invalid dir {cwd}. Should be {build_dir}")
    exit(-1)


# Build and remove previous results
os.system(f'{cmake} .. && make -j')
os.system('echo "problemsize,workingsetsize,milliseconds,bandwidth" > bwtest-benchmark.csv')


max_problem_size = 1 * 1024**3    # 1 GiB
max_working_set_size = 8200 #512 * 1024 # 512 KiB

problem_size = 8
working_set_size = 8

# Variable problem size
while problem_size <= max_problem_size:
    os.system(f'./bwtest-benchmark {problem_size} {working_set_size}')
    problem_size *= 2
os.system('mv bwtest-benchmark.csv problem-size.csv')

# Variable working set size
problem_size = 256 * 1024**2       # 256 MiB
while working_set_size <= max_working_set_size:
    os.system(f'./bwtest-benchmark {problem_size} {working_set_size}')
    working_set_size *= 2
os.system('mv bwtest-benchmark.csv working-set-size.csv')



'''
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
'''