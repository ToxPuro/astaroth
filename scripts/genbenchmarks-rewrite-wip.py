#!/usr/bin/env python3
import os
import argparse
import socket
import math
from contextlib import redirect_stdout

# Parse arguments
parser = argparse.ArgumentParser(description='A tool for generating benchmarks')
parser.add_argument('--task-type', type=str, nargs='+', choices=['preprocess', 'build', 'run', 'postprocess'], help='The type of the task performed with this script', required=True)
## Build arguments
parser.add_argument('--implementations', type=str, nargs='+', choices=['implicit', 'explicit'], default=['implicit', 'explicit'], help='The list of implementations used in testing')
parser.add_argument('--io-implementations', type=str, nargs='+', choices=['collective', 'distributed'], default=['collective', 'distributed'], help='The list of IO implementations used in testing')
parser.add_argument('--launch-bounds-range', type=int, nargs=2, default=[0, 1024], help='The range for the maximum number of threads per block applied to launch bounds in testing (inclusive)')
parser.add_argument('--cmakelistdir', type=str, default='.', help='Directory containing the project CMakeLists.txt')
parser.add_argument('--use-hip', action='store_true', help='Compile with HIP support')
## Run arguments
parser.add_argument('--account', type=str, help='The account used in tests')
parser.add_argument('--partition', type=str, help='The partition used for running the tests')
parser.add_argument('--run-scripts', type=str, nargs='+', help='A list of job scripts to run the tests')
parser.add_argument('--dims', type=int, default=[64, 64, 64], nargs=3, help='The dimensions of the computational domain')
parser.add_argument('--run-dirs', type=str, nargs='+', help='A list of directories to run the tests in')
parser.add_argument('--num-devices', type=int, nargs=2, default=[1, 4096], help='The range for the number of devices used in testing (inclusive)')
parser.add_argument('--dryrun', action='store_true', help='Do a dryrun without compiling or running. Prints os commands to stdout.')
args = parser.parse_args()

benchmark_dir = 'benchmark-data'
scripts_dir    = f'{benchmark_dir}/scripts'
builds_dir     = f'{benchmark_dir}/builds'
output_dir     = f'{benchmark_dir}/output'

# Set system account
if args.account:
    system.account = args.account

# Set system partition
if args.partition:
    system.partition = args.partition

# Set problem size
nx = args.dims[0]
ny = args.dims[1]
nz = args.dims[2]

# Set device counts
min_devices = args.num_devices[0]
max_devices = args.num_devices[1]

# Print some information before starting
print(args.run_dirs)
print(args.run_scripts)

def syscall(cmd):
    if (args.dryrun):
        print(cmd)
    else:
        os.system(cmd)

# System
class System:
    
    def __init__(self, id, account, partition, ngpus_per_node, modules, use_hip, gres='', additional_commands='', optimal_implementation=1, optimal_tpb=0):
        self.id = id
        self.account = account
        self.partition = partition
        self.ngpus_per_node = ngpus_per_node
        self.modules = modules
        self.use_hip = use_hip
        self.gres = gres
        self.additional_commands = additional_commands
        self.optimal_implementation = optimal_implementation
        self.optimal_tpb = optimal_tpb

    def load_modules(self):
        syscall(f'module purge')
        syscall(self.modules)
        
    def print_sbatch_header(self, ntasks, ngpus=-1):
        if ngpus < 0:
            ngpus = ntasks

        time = '00:14:59'

        gpualloc_per_node = min(ngpus, self.ngpus_per_node)
        nodes = int(math.ceil(ngpus / self.ngpus_per_node))
        if nodes > 1 and ntasks != ngpus:
            print(f'ERROR: Insufficient ntasks ({ntasks}). Asked for {ngpus} devices but there are only {self.ngpus_per_node} devices per node.')
            assert(nodes == 1 or ntasks == ngpus)

        print('#!/bin/bash')
        if self.account:
            print(f'#SBATCH --account={self.account}')
        if self.gres:
            print(f'#SBATCH --gres={self.gres}:{gpualloc_per_node}')
        print(f'#SBATCH --partition={self.partition}')
        print(f'#SBATCH --ntasks={ntasks}')
        print(f'#SBATCH --nodes={nodes}')
        print(f'#SBATCH --time={time}')
        #print('#SBATCH --accel-bind=g') # bind tasks to closest GPU
        #print('#SBATCH --hint=memory_bound') # one core per socket
        #print(f'#SBATCH --ntasks-per-socket={min(ntasks, ngpus/self.nsockets)}')
        #print('#SBATCH --cpu-bind=sockets')
        print(self.additional_commands)
    
        # Load modules
        print(f'module purge')
        print(self.modules)

mahti = System(id='a100', account='project_2000403', partition='gpusmall', ngpus_per_node=4, gres='gpu:a100',
               modules='module load gcc/9.4.0 openmpi/4.1.2-cuda cuda cmake', use_hip=False, optimal_implementation='1', optimal_tpb='0')
puhti = System(id='v100', account='project_2000403', partition='gpu', ngpus_per_node=4,
               gres='gpu:v100', modules='module load gcc cuda openmpi cmake', use_hip=False,
               additional_commands='''
export UCX_RNDV_THRESH=16384
export UCX_RNDV_SCHEME=get_zcopy
export UCX_MAX_RNDV_RAILS=1''', optimal_implementation='1', optimal_tpb='0')
triton = System(id='mi100', account='', partition='gpu-amd', ngpus_per_node=1, gres='',
                modules='module load gcc bison flex cmake openmpi', use_hip=True, optimal_implementation='1', optimal_tpb='512')
lumi = System(id='mi250x', account='project_462000120', partition='pilot', ngpus_per_node=8, gres='gpu', modules='''
        module load CrayEnv
        module load PrgEnv-cray
        module load craype-accel-amd-gfx90a
        module load rocm
        module load buildtools
        module load cray-python
        ''', use_hip=True, optimal_implementation='1', optimal_tpb='512')

# Select system
hostname = socket.gethostname()
if 'mahti' in hostname:
    system = mahti
elif 'puhti' in hostname:
    system = puhti
elif 'uan' in hostname:
    system = lumi
elif 'triton' in hostname:
    system = triton
else:
    print(f'Unknown system {hostname}')
    exit(-1)
system.load_modules()

# Microbenchmarks
def gen_microbenchmarks(system):
    with open(f'{scripts_dir}/microbenchmark.sh', 'w') as f:
        with redirect_stdout(f):
            # Create the batch script
            system.print_sbatch_header(ntasks=1)

            # Bandwidth
            problem_size     = 8 # Bytes
            working_set_size = 8 # Bytes
            max_problem_size = 1 * 1024**3    # 1 GiB
            while problem_size <= max_problem_size:
                print(f'srun ./bwtest-benchmark {problem_size} {working_set_size}')
                problem_size *= 2

            # Working set
            problem_size     = 256 * 1024**2 # Bytes, 256 MiB
            working_set_size = 8         # Bytes
            max_working_set_size = 8200  # r = 512, (512 * 2 + 1) * 8 bytes = 8200 bytes
            while working_set_size <= max_working_set_size:
                print(f'srun ./bwtest-benchmark {problem_size} {working_set_size}')
                working_set_size *= 2

# Device benchmarks
def gen_devicebenchmarks(system, nx, ny, nz):
    with open(f'{scripts_dir}/device-benchmark.sh', 'w') as f:
        with redirect_stdout(f):
            system.print_sbatch_header(1)
            print(f'srun ./benchmark-device {nx} {ny} {nz}')

# Intra-node benchmarks
def gen_nodebenchmarks(system, nx, ny, nz, min_devices, max_devices):
    devices = min_devices
    while devices <= min(system.ngpus_per_node, max_devices):
        with open(f'{scripts_dir}/node-benchmark-{devices}.sh', 'w') as f:
            with redirect_stdout(f):
                system.print_sbatch_header(1, devices)
                print(f'srun ./benchmark-node {nx} {ny} {nz}')
        devices *= 2

# Strong scaling
def gen_strongscalingbenchmarks(system, nx, ny, nz, min_devices, max_devices):
    devices = min_devices
    while devices <= max_devices:
        with open(f'{scripts_dir}/strong-scaling-benchmark-{devices}.sh', 'w') as f:
            with redirect_stdout(f):
                system.print_sbatch_header(devices)
                print(f'srun ./benchmark {nx} {ny} {nz}')
        devices *= 2

# Weak scaling
def gen_weakscalingbenchmarks(system, nx, ny, nz, min_devices, max_devices):
    # Weak scaling
    devices = min_devices
    while devices <= max_devices:
        with open(f'{scripts_dir}/weak-scaling-benchmark-{devices}.sh', 'w') as f:
            with redirect_stdout(f):
                system.print_sbatch_header(devices)
                print(f'srun ./benchmark {nx} {ny} {nz}')
        devices *= 2
        if nz < ny:
            nz *= 2
        elif ny < nx:
            ny *= 2
        else:
            nx *= 2

# IO benchmarks
def gen_iobenchmarks(system, nx, ny, nz, min_devices, max_devices):
    devices = min_devices
    while devices <= max_devices:
        with open(f'{scripts_dir}/io-scaling-benchmark-{devices}.sh', 'w') as f:
            with redirect_stdout(f):
                system.print_sbatch_header(devices)
                print(f'srun ./mpi-io {nx} {ny} {nz}')
        devices *= 2

# Generate run scripts
if 'preprocess' in args.task_type:
    syscall(f'mkdir -p {scripts_dir}')

    gen_microbenchmarks(system)

    gen_devicebenchmarks(system, nx, ny, nz)
    gen_nodebenchmarks(system, nx, ny, nz, min_devices, max_devices)

    gen_strongscalingbenchmarks(system, nx, ny, nz, min_devices, max_devices)
    gen_weakscalingbenchmarks(system, nx, ny, nz, min_devices, max_devices)
    gen_iobenchmarks(system, nx, ny, nz, min_devices, max_devices)

# Build and run
if 'preprocess' in args.task_type or 'build' in args.task_type or 'run' in args.task_type:
    syscall(f'mkdir -p {builds_dir}')
    
    for implementation in args.implementations:
        for io_implementation in args.io_implementations:
            tpb = args.launch_bounds_range[0]
            while tpb <= args.launch_bounds_range[1]:

                impl_id     = 1 if implementation == 'implicit' else 2
                use_smem    = implementation == 'explicit'
                distributed = io_implementation == 'distributed'

                build_dir = f'{builds_dir}/implementation{impl_id}_maxthreadsperblock{tpb}_distributed{distributed}'
                syscall(f'mkdir -p {build_dir}')

                # Build
                if 'build' in args.task_type or 'run' in args.task_type:
                    flags = f'''-DMPI_ENABLED=ON -DSINGLEPASS_INTEGRATION=ON -DUSE_HIP={system.use_hip} -DIMPLEMENTATION={impl_id} -DUSE_SMEM={use_smem} -DUSE_DISTRIBUTED_IO={distributed}'''
                    syscall(f'cmake {flags} -S {args.cmakelistdir} -B {build_dir}')
                    syscall(f'make --directory={build_dir} -j')

                # Run
                if 'run' in args.task_type:
                    if args.run_dirs and build_dir in args.run_dirs:
                        for script in args.run_scripts:
                            syscall(f'sbatch --chdir="{build_dir}" {script}')

                tpb = 1 if tpb == 0 else 2*tpb

import pandas as pd
def postprocess(system, nx, ny, nz):
    syscall(f'mkdir -p {output_dir}')

    with open(f'{output_dir}/microbenchmark.csv', 'w') as f:
        with redirect_stdout(f):
            print('usesmem,maxthreadsperblock,problemsize,workingsetsize,milliseconds,bandwidth')
    syscall(f'cat {builds_dir}/*/microbenchmark.csv >> {output_dir}/microbenchmark.csv')

    df = pd.read_csv(f'{output_dir}/microbenchmark.csv', comment='#')
    df = df.loc[(df['usesmem'] == 0) & (df['maxthreadsperblock'] == 0) & (df['workingsetsize'] == 8)]
    df = df.drop_duplicates(subset=['problemsize'])
    df = df.sort_values(by=['problemsize'])
    df.to_csv(f'{output_dir}/bandwidth-{system.id}.csv', index=False)

    df = pd.read_csv(f'{output_dir}/microbenchmark.csv', comment='#')
    df = df.loc[(df['usesmem'] == 1) & (df['maxthreadsperblock'] == 0) & (df['workingsetsize'] == 8)]
    df = df.drop_duplicates(subset=['problemsize'])
    df = df.sort_values(by=['problemsize'])
    df.to_csv(f'{output_dir}/bandwidth-smem-{system.id}.csv', index=False)

    df = pd.read_csv(f'{output_dir}/microbenchmark.csv', comment='#')
    df = df.loc[(df['usesmem'] == 0) & (df['maxthreadsperblock'] == 0) & (df['problemsize'] == 268435456)]
    df = df.drop_duplicates(subset=['workingsetsize'])
    df = df.sort_values(by=['workingsetsize'])
    df.to_csv(f'{output_dir}/workingset-{system.id}.csv', index=False)

    df = pd.read_csv(f'{output_dir}/microbenchmark.csv', comment='#')
    df = df.loc[(df['usesmem'] == 1) & (df['maxthreadsperblock'] == 0) & (df['problemsize'] == 268435456)]
    df = df.drop_duplicates(subset=['workingsetsize'])
    df = df.sort_values(by=['workingsetsize'])
    df.to_csv(f'{output_dir}/workingset-smem-{system.id}.csv', index=False)

    # Device
    with open(f'{output_dir}/device-benchmark.csv', 'w') as f:
        with redirect_stdout(f):
            print('implementation,maxthreadsperblock,milliseconds,nx,ny,nz,devices')
    syscall(f'cat {builds_dir}/*/device-benchmark.csv >> {output_dir}/device-benchmark.csv')

    df = pd.read_csv(f'{output_dir}/device-benchmark.csv', comment='#')
    df = df.loc[(df['implementation'] == 1)]
    df = df.sort_values(by=['maxthreadsperblock'])
    #df = df.drop_duplicates(subset=['workingsetsize'])
    df.to_csv(f'{output_dir}/implicit-{system.id}.csv', index=False)

    df = pd.read_csv(f'{output_dir}/device-benchmark.csv', comment='#')
    df = df.loc[(df['implementation'] == 2)]
    df = df.sort_values(by=['maxthreadsperblock'])
    #df = df.drop_duplicates(subset=['workingsetsize'])
    df.to_csv(f'{output_dir}/explicit-{system.id}.csv', index=False)

    # Node
    with open(f'{output_dir}/node-benchmark.csv', 'w') as f:
        with redirect_stdout(f):
            print('implementation,maxthreadsperblock,milliseconds,nx,ny,nz,devices')
    syscall(f'cat {builds_dir}/*/node-benchmark.csv >> {output_dir}/node-benchmark.csv')

    df = pd.read_csv(f'{output_dir}/node-benchmark.csv', comment='#')
    df = df.loc[(df['implementation'] == 1)]
    df = df.sort_values(by=['maxthreadsperblock'])
    #df = df.drop_duplicates(subset=['workingsetsize'])
    df.to_csv(f'{output_dir}/node-implicit-{system.id}.csv', index=False)

    df = pd.read_csv(f'{output_dir}/node-benchmark.csv', comment='#')
    df = df.loc[(df['implementation'] == 2)]
    df = df.sort_values(by=['maxthreadsperblock'])
    #df = df.drop_duplicates(subset=['workingsetsize'])
    df.to_csv(f'{output_dir}/node-explicit-{system.id}.csv', index=False)

    # Scaling
    with open(f'{output_dir}/scaling-benchmark.csv', 'w') as f:
        with redirect_stdout(f):
            print('devices,millisecondsmin,milliseconds50thpercentile,milliseconds90thpercentile,millisecondsmax,usedistributedcommunication,nx,ny,nz,dostrongscaling')
    syscall(f'cat {builds_dir}/*/scaling-benchmark.csv >> {output_dir}/scaling-benchmark.csv')

    df = pd.read_csv(f'{output_dir}/scaling-benchmark.csv', comment='#')
    df = df.loc[(df['nx'] == nx) & (df['ny'] == ny) & (df['nz'] == nz)]
    df = df.sort_values(by=['devices'])
    df = df.drop_duplicates(subset=['devices', 'nx', 'ny', 'nz'])
    df.to_csv(f'{output_dir}/scaling-strong-{system.id}.csv', index=False)

    nn = nx * ny * nz
    df = pd.read_csv(f'{output_dir}/scaling-benchmark.csv', comment='#')
    df = df.loc[(df['nx'] * df['ny'] * df['nz']) / df['devices'] == nn]
    df = df.sort_values(by=['devices'])
    df = df.drop_duplicates(subset=['devices', 'nx', 'ny', 'nz'])
    df.to_csv(f'{output_dir}/scaling-weak-{system.id}.csv', index=False)

    # IO scaling
    with open(f'{output_dir}/scaling-io-benchmark.csv', 'w') as f:
        with redirect_stdout(f):
            print(f'devices,writemilliseconds,writebandwidth,readmilliseconds,readbandwidth,usedistributedio,nx,ny,nz')
    syscall(f'cat {builds_dir}/*/scaling-io-benchmark.csv >> {output_dir}/scaling-io-benchmark.csv')

    # Collective
    df = pd.read_csv(f'{output_dir}/scaling-io-benchmark.csv', comment='#')
    df = df.loc[(df['usedistributedio'] == 0)]
    df = df.sort_values(by=['devices'])
    df = df.drop_duplicates(subset=['devices', 'nx', 'ny', 'nz'])
    df.to_csv(f'{output_dir}/scaling-io-collective-{system.id}.csv', index=False)

    # Distributed
    df = pd.read_csv(f'{output_dir}/scaling-io-benchmark.csv', comment='#')
    df = df.loc[(df['usedistributedio'] == 1)]
    df = df.sort_values(by=['devices'])
    df = df.drop_duplicates(subset=['devices', 'nx', 'ny', 'nz'])
    df.to_csv(f'{output_dir}/scaling-io-distributed-{system.id}.csv', index=False)

# Postprocess
if 'postprocess' in args.task_type:
    postprocess(system, nx, ny, nz)
    