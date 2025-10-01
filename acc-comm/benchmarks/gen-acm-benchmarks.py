#!/usr/bin/env python3
import numpy as np
from contextlib import redirect_stdout

def gen_srun_command(cpu_bind = ""):
    return f'srun --cpu-bind="{cpu_bind}"' if cpu_bind else "srun"

def gen_run_information():
    return """
module list
cmake -LAH >> system_info-$SLURM_JOB_ID.txt
"""

class System:
    def __init__(self, name, account, partition, devices_per_node, sockets_per_node, gres_type, modules, env_vars, cpu_bind):
        self.name = name
        self.account = account
        self.partition = partition
        self.devices_per_node = devices_per_node
        self.sockets_per_node = sockets_per_node
        self.gres_type = gres_type
        self.modules = modules
        self.env_vars = env_vars
        self.cpu_bind = cpu_bind

    def gen_sbatch_preamble(self, nprocs, time_limit):
        ntasks_per_node = min(nprocs, self.devices_per_node)
        nnodes = nprocs // ntasks_per_node
        assert(ntasks_per_node * nnodes == nprocs)

        preamble = f"""#!/usr/bin/env bash
#SBATCH --account={self.account}
#SBATCH --time={time_limit}
#SBATCH --partition={self.partition}
#SBATCH --nodes={nnodes}
#SBATCH --ntasks-per-node={ntasks_per_node}
"""
        if self.gres_type:
            preamble += f"#SBATCH --gres={self.gres_type}:{ntasks_per_node}\n"
        else:
            preamble += f"#SBATCH --gpus-per-node={ntasks_per_node}\n"

        if self.sockets_per_node:
            ntasks_per_socket = max(ntasks_per_node // self.sockets_per_node, 1)
            preamble += f"#SBATCH --ntasks-per-socket={ntasks_per_socket}\n"
        return preamble

    def gen_preamble(self, nprocs, time_limit):
        return self.gen_sbatch_preamble(nprocs, time_limit) + \
               self.modules + \
               self.env_vars + \
               gen_run_information()
    
    def gen_srun_command(self, nprocs):
        if nprocs >= self.devices_per_node:
            return gen_srun_command(self.cpu_bind)
        else:
            return gen_srun_command()


lumi = System(
    name = "lumi",
    account = "project_462001062",
    #partition = "dev-g",
    partition = "standard-g",
    devices_per_node = 8,
    sockets_per_node = None,
    gres_type = None,
    modules = """
# Modules
module load PrgEnv-cray
module load craype-accel-amd-gfx90a
module load rocm
module load cray-python
module load cray-hdf5
module load LUMI/24.03 buildtools/24.03
module load craype-accel-amd-gfx90a # Must be loaded after LUMI/24.03
""",
    env_vars = """
# Environment variables
export MPICH_GPU_SUPPORT_ENABLED=1
""",
    cpu_bind = "map_cpu:33,41,49,57,17,25,1,9",
)

puhti = System(
    name = "puhti",
    account = "project_2000403",
    #partition = "gputest",
    partition = "gpu",
    devices_per_node = 4,
    sockets_per_node = 2,
    gres_type = "gpu:v100",
    modules = """
# Modules
module load gcc/11.3.0 openmpi/4.1.4-cuda cuda cmake
""",
    env_vars = "",
    cpu_bind = "",
)

mahti = System(
    name = "mahti",
    account = "project_2000403",
    #partition = "gputest",
    partition = "gpumedium",
    devices_per_node = 4,
    sockets_per_node = 2,
    gres_type = "gpu:a100",
    modules = """
# Modules
module load python-data
module load gcc/10.4.0 openmpi/4.1.5-cuda cuda
""",
    env_vars = "",
    cpu_bind = "",
)

# def gen_tfm_benchmark(system, nprocs, time_limit, config, nn, label):
#     return system.gen_preamble(nprocs, time_limit) + \
#            gen_run_information_tfm(config, label) + \
#            f'{system.gen_srun_command(nprocs)} {gen_run_command_tfm(config, nn, label)}'

# def gen_weak_scaling_benchmarks(system, max_nprocs):
    
#     nn = [256,256,256]
#     axis = len(nn) - 1
#     time_limit = "00:30:00"
#     config = "/users/pekkila/astaroth/samples/tfm/cases/laplace-nonsoca.ini"
#     label = "weak"

#     nprocs = 1
#     while nprocs <= max_nprocs:
#         with open(f'bm-{label}-{nprocs}.sh', 'w') as f:
#             print(gen_tfm_benchmark(system, nprocs, time_limit, config, nn, label), file=f)
        
#         nprocs *= 2
#         nn[axis] *= 2
#         axis = (axis + len(nn) - 1) % len(nn)
        
# def gen_strong_scaling_benchmarks(system, max_nprocs):
#     nn = [256,256,256]
#     time_limit = "00:15:00"
#     config = "/users/pekkila/astaroth/samples/tfm/cases/laplace-nonsoca.ini"
#     label = "strong"

#     nprocs = 2
#     while nprocs <= max_nprocs:
#         with open(f'bm-{label}-{nprocs}.sh', 'w') as f:
#             print(gen_tfm_benchmark(system, nprocs, time_limit, config, nn, label), file=f)
#         nprocs *= 2

# gen_weak_scaling_benchmarks(lumi, 4096)
# gen_strong_scaling_benchmarks(lumi, 512)

def gen_pack_benchmarks(system):
    nn = 256**3
    radius = 4
    nprocs = 1
    nsamples = 100
    time_limit = "00:15:00"
    jobid = "$SLURM_JOB_ID"

    with open(f'bm-{system.name}-pack-nd.sh', 'w') as f:
        with redirect_stdout(f):
            print(system.gen_preamble(nprocs, time_limit))

            # ND
            jobname = "\"nd\""
            npack = 1
            for ndims in range(1, 6):
                dim = int(np.round(nn**(1/ndims)))
                print(f'time ./benchmarks/bm_pack {dim} {radius} {ndims} {npack} {nsamples} {jobid} {jobname}')
            print()

            jobname = "\"aggr\""
            ndims = 3
            for npack in [1, 2, 4, 8, 16]:
                dim = int(np.round(nn**(1/ndims)))
                print(f'time ./benchmarks/bm_pack {dim} {radius} {ndims} {npack} {nsamples} {jobid} {jobname}')

def gen_comm_benchmarks(system):

    nn = 256**3
    radius = 4
    npack = 1
    nsamples = 100
    time_limit = "00:15:00"
    jobid = "$SLURM_JOB_ID"

    jobname = "\"scaling-strong\""
    ndims = 3
    dim = int(np.round(nn**(1/ndims)))
    for nprocs in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
        with open(f'bm-{system.name}-comm-scaling-strong-{nprocs}.sh', 'w') as f:
            with redirect_stdout(f):
                print(system.gen_preamble(nprocs, time_limit))
                print(f'time ./benchmarks/bm_collective_comm {dim} {radius} {ndims} {npack} {nsamples} {jobid} {jobname}')

    jobname = "\"nd\""
    nprocs = 64
    npack = 1
    with open(f'bm-{system.name}-comm-nd-{nprocs}.sh', 'w') as f:
        with redirect_stdout(f):
            print(system.gen_preamble(nprocs, time_limit))
            for ndims in range(1, 6):
                dim = int(np.round(nn**(1/ndims)))
                print(f'time ./benchmarks/bm_collective_comm {dim} {radius} {ndims} {npack} {nsamples} {jobid} {jobname}')

    jobname = "\"aggr\""
    nprocs = 64
    ndims = 3
    dim = int(np.round(nn**(1/ndims)))
    with open(f'bm-{system.name}-comm-aggr-{nprocs}.sh', 'w') as f:
        with redirect_stdout(f):
            print(system.gen_preamble(nprocs, time_limit))
            for npack in [1, 2, 4, 8, 16]:
                print(f'time ./benchmarks/bm_collective_comm {dim} {radius} {ndims} {npack} {nsamples} {jobid} {jobname}')

def gen_rank_reordering_benchmarks(system):
    nsamples = 100
    time_limit = "00:15:00"
    jobid = "$SLURM_JOB_ID"
    radius = 4

    cases = np.array((
        [256, 256, 256],
        [512, 256, 128], 
        [512, 512, 64],
        [128, 256, 512],
        [64, 512, 512], 
    ))
    ndims = len(cases[0])
    assert(ndims == 3)


    for nprocs in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
        with open(f'bm-{system.name}-rank-reordering-scaling-strong-{nprocs}.sh', 'w') as f:
            with redirect_stdout(f):
                print(system.gen_preamble(nprocs, time_limit))

                for i, nn in enumerate(cases):
                    #jobname = "\"scaling-strong-(" + ",".join([str(x) for x in case]) + ")\""
                    jobname = f"\"scaling-strong-case-{i}\""
                    print(f'time ./benchmarks/bm_rank_reordering {nn[0]} {nn[1]} {nn[2]} {radius} {nsamples} {jobid} {jobname} 0') # Non-hiearchical
                    print(f'time {gen_srun_command(system.cpu_bind)} ./benchmarks/bm_rank_reordering {nn[0]} {nn[1]} {nn[2]} {radius} {nsamples} {jobid} {jobname} 1') # Hierarchical


    # NOTE: modifies cases
    # max_nprocs = 256
    # nprocs = 1
    # axis = ndims-1
    # while nprocs <= max_nprocs:
    #     with open(f'bm-{system.name}-rank-reordering-scaling-weak-{nprocs}.sh', 'w') as f:
    #         with redirect_stdout(f):
    #             print(system.gen_preamble(nprocs, time_limit))

    #             for i, nn in enumerate(cases):
    #                 # print(f'nn: {nn}')
    #                 # print(f'nprocs: {nprocs}')
    #                 # print(f'axis: {axis}')
    #                 jobname = f"\"scaling-weak-case-{i}\""
    #                 print(f'time ./benchmarks/bm_rank_reordering {nn[0]} {nn[1]} {nn[2]} {radius} {nsamples} {jobid} {jobname} 0') # Non-hiearchical
    #                 print(f'time {gen_srun_command(system.cpu_bind)} ./benchmarks/bm_rank_reordering {nn[0]} {nn[1]} {nn[2]} {radius} {nsamples} {jobid} {jobname} 1') # Hierarchical
    #                 nn[axis] *=2
                    
    #             nprocs *= 2
    #             axis = (axis + len(nn) - 1) % len(nn)



systems = [lumi, puhti, mahti]

for system in systems:
    gen_pack_benchmarks(system)
    gen_comm_benchmarks(system)
    gen_rank_reordering_benchmarks(system)