#!/bin/python3
from contextlib import redirect_stdout
from contextlib import contextmanager
import os
import socket
import math
import sys

dryrun=False

build_benchmarks=True
run_benchmarks=True

class System:

    def __init__(self, id, account, partition, ngpus_per_node, modules, use_hip, gres='', additional_commands=''):
        self.id = id
        self.account = account
        self.partition = partition
        self.ngpus_per_node = ngpus_per_node
        self.modules = modules
        self.use_hip = use_hip
        self.gres = gres
        self.additional_commands = additional_commands
        
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

    def build(self, build_flags, cmakelistdir):
        # Load modules
        if dryrun:
            print(f'module purge')
            print(self.modules)
            print(f'cmake {build_flags} {cmakelistdir}')
        else:
            os.system(f'module purge')
            os.system(self.modules)
            os.system(f'cmake {build_flags} {cmakelistdir}')
            os.system('make -j')


mahti = System(id='a100', account='project_2000403', partition='gpumedium', ngpus_per_node=4, gres='gpu:a100',
               modules='module load gcc/9.4.0 openmpi/4.1.2-cuda cuda cmake', use_hip=False)
puhti = System(id='v100', account='project_2000403', partition='gpu', ngpus_per_node=4,
               gres='gpu:v100', modules='module load gcc cuda openmpi cmake', use_hip=False,
               additional_commands='''
export UCX_RNDV_THRESH=16384
export UCX_RNDV_SCHEME=get_zcopy
export UCX_MAX_RNDV_RAILS=1''')
triton = System(id='mi100', account='', partition='gpu-amd', ngpus_per_node=1, gres='',
                modules='module load gcc bison flex cmake openmpi', use_hip=True)
lumi = System(id='mi250x', account='project_462000120', partition='pilot', ngpus_per_node=8, gres='gpu', modules='''
        module load CrayEnv
        module load PrgEnv-cray
        module load craype-accel-amd-gfx90a
        module load rocm
        module load buildtools
        module load cray-python
        ''', use_hip=True)

class FileStructure:

    def __init__(self, cmakelistdir='.'):

        initial_dir = os.getcwd()

        # Record the CMakeLists.txt dir
        os.chdir(cmakelistdir)
        if not os.path.isfile('CMakeLists.txt'):
            print(f'Could not find CMakeLists.txt in {os.getcwd()}. Please run the script in the dir containing the project CMakeLists.txt or give the directory as a parameter.')
            exit(-1)

        self.cmakelistdir = os.getcwd()
        os.chdir(initial_dir)

        # Create a new dir for the benchmark data
        os.system(f'mkdir -p benchmark-data')
        os.chdir('benchmark-data')
        self.base_dir = os.getcwd()

        os.system(f'mkdir -p builds')
        os.chdir('builds')
        self.build_dir = os.getcwd()

        os.chdir(self.base_dir)
        os.system(f'mkdir -p scripts')
        os.chdir('scripts')
        self.script_dir = os.getcwd()

        os.chdir(initial_dir)


def genbenchmarks(system, fs):

    # Create batch scripts
    os.chdir(fs.script_dir)

    # Microbenchmarks
    with open('microbenchmark.sh', 'w') as f:
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
    nn = 64
    nx = ny = nz = nn
    with open('device-benchmark.sh', 'w') as f:
        with redirect_stdout(f):
            system.print_sbatch_header(1)
            print(f'srun ./benchmark-device {nx} {ny} {nz}')

    # Intra-node benchmarks
    devices = 1
    while devices <= system.ngpus_per_node:
        with open(f'node-benchmark-{devices}.sh', 'w') as f:
            with redirect_stdout(f):
                system.print_sbatch_header(1, devices)
                print(f'srun ./benchmark-node {nx} {ny} {nz}')
        devices *= 2

    # Strong scaling
    max_devices = 4096
    devices = 1
    while devices <= max_devices:
        with open(f'strong-scaling-benchmark-{devices}.sh', 'w') as f:
            with redirect_stdout(f):
                system.print_sbatch_header(devices)
                print(f'srun ./benchmark {nx} {ny} {nz}')
        devices *= 2

    # Weak scaling
    devices = 1
    while devices <= max_devices:
        with open(f'weak-scaling-benchmark-{devices}.sh', 'w') as f:
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
    devices = 1
    while devices <= max_devices:
        with open(f'io-scaling-benchmark-{devices}.sh', 'w') as f:
            with redirect_stdout(f):
                system.print_sbatch_header(devices)
                print(f'srun ./mpi-io {nx} {ny} {nz}')
        devices *= 2

    # Create build dirs
    num_implementations = 2
    max_threads_per_block = 1024
    for implementation in range(1, num_implementations+1):
        tpb = 0
        while tpb <= max_threads_per_block:

            os.chdir(fs.build_dir)
            dir = f'implementation{implementation}_maxthreadsperblock{tpb}'
            os.system(f'mkdir -p {dir}')
            os.chdir(dir)

            # Build
            if build_benchmarks:
                use_smem = implementation == 2 # tmp hack, note depends on implementation enum
                build_flags = f'-DUSE_HIP={system.use_hip} -DMPI_ENABLED=ON -DIMPLEMENTATION={implementation} -DMAX_THREADS_PER_BLOCK={tpb} -DUSE_SMEM={use_smem}'
                system.build(build_flags, fs.cmakelistdir)

            # Run
            if run_benchmarks:
                # Run microbenchmarks
                if dryrun:
                    print(f'sbatch {fs.script_dir}/microbenchmark.sh')
                else:
                    os.system(f'sbatch {fs.script_dir}/microbenchmark.sh')

                '''
                # Run device benchmarks
                if dryrun:
                    print(f'sbatch {fs.script_dir}/device-benchmark.sh')
                else:
                    os.system(f'sbatch {fs.script_dir}/device-benchmark.sh')

                # Run node benchmarks
                if dryrun:
                    print(f'sbatch {fs.script_dir}/node-benchmark-2.sh')
                else:
                    os.system(f'sbatch {fs.script_dir}/node-benchmark-2.sh')

                # Run scaling benchmarks
                # TODO
                '''
        
            if tpb == 0:
                tpb = 32
            else:
                tpb *= 2

# pip3 install --user pandas numpy
import pandas as pd
def postprocess(system, fs):
    os.chdir(fs.base_dir)

    with open(f'microbenchmark.csv', 'w') as f:
        with redirect_stdout(f):
            print('usesmem,maxthreadsperblock,problemsize,workingsetsize,milliseconds,bandwidth')
    os.system(f'cat {fs.build_dir}/*/microbenchmark.csv >> microbenchmark.csv')

    df = pd.read_csv('microbenchmark.csv', comment='#')
    df = df.loc[(df['usesmem'] == 0) & (df['maxthreadsperblock'] == 0) & (df['workingsetsize'] == 8)]
    df = df.drop_duplicates(subset=['problemsize'])
    df.to_csv(f'bandwidth-{system.id}.csv', index=False)

    df = pd.read_csv('microbenchmark.csv', comment='#')
    df = df.loc[(df['usesmem'] == 1) & (df['maxthreadsperblock'] == 0) & (df['workingsetsize'] == 8)]
    df = df.drop_duplicates(subset=['problemsize'])
    df.to_csv(f'bandwidth-smem-{system.id}.csv', index=False)

    df = pd.read_csv('microbenchmark.csv', comment='#')
    df = df.loc[(df['usesmem'] == 0) & (df['maxthreadsperblock'] == 0) & (df['problemsize'] == 268435456)]
    df = df.drop_duplicates(subset=['workingsetsize'])
    df.to_csv(f'workingset-{system.id}.csv', index=False)

    df = pd.read_csv('microbenchmark.csv', comment='#')
    df = df.loc[(df['usesmem'] == 1) & (df['maxthreadsperblock'] == 0) & (df['problemsize'] == 268435456)]
    df = df.drop_duplicates(subset=['workingsetsize'])
    df.to_csv(f'workingset-smem-{system.id}.csv', index=False)

    with open(f'device-benchmark.csv', 'w') as f:
        with redirect_stdout(f):
            print('implementation,maxthreadsperblock,milliseconds,nx,ny,nz,devices')
    os.system(f'cat {fs.build_dir}/*/device-benchmark.csv >> device-benchmark.csv')

    df = pd.read_csv('device-benchmark.csv', comment='#')
    df = df.loc[(df['implementation'] == 1)]
    #df = df.drop_duplicates(subset=['workingsetsize'])
    df.to_csv(f'implicit-{system.id}.csv', index=False)

    df = pd.read_csv('device-benchmark.csv', comment='#')
    df = df.loc[(df['implementation'] == 2)]
    #df = df.drop_duplicates(subset=['workingsetsize'])
    df.to_csv(f'explicit-{system.id}.csv', index=False)

    with open(f'node-benchmark.csv', 'w') as f:
        with redirect_stdout(f):
            print('implementation,maxthreadsperblock,milliseconds,nx,ny,nz,devices')
    os.system(f'cat {fs.build_dir}/*/node-benchmark.csv >> node-benchmark.csv')

# Generate the filestructure
if len(sys.argv) > 1:
    fs = FileStructure(sys.argv[1])
else:
    fs = FileStructure()

# Select system
system = puhti

# Generate and run the benchmarks
#genbenchmarks(system, fs)

# Postprocess
postprocess(system, fs)
