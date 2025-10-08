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

def get_global_dim(nprocs, dim, ndims):
    return int(np.round((nprocs*dim**ndims)**(1/ndims)))

def gen_pack_benchmarks(system):
    radius = 4
    nsamples = 100          # NOTE: Can increase for production
    time_limit = "00:15:00" # NOTE: Can increase for production
    jobid = "$SLURM_JOB_ID"

    #for dim in [32, 128, 512]:
    dims =  [16, 64, 256] # takes 5 mins 100 samples
    nprocs = 1  
    for dim in dims:
    
        # OK
        jobname = f"pack-nd-dim-{dim}"
        npack = 1
        with open(f'bm-{system.name}-{jobname}.sh', 'w') as f:
            with redirect_stdout(f):
                print(system.gen_preamble(nprocs, time_limit))

                # ND
                for ndims in range(1, 6):
                    if dim**ndims > dims[-1]**3: # Skip way too large dims
                        continue
                    print(f'time srun ./benchmarks/bm_pack {dim} {radius} {ndims} {npack} {nsamples} {jobid} {jobname}')
                print()

        # OK
        jobname = f"pack-aggr-dim-{dim}"
        ndims = 3
        with open(f'bm-{system.name}-{jobname}.sh', 'w') as f:
            with redirect_stdout(f):
                print(system.gen_preamble(nprocs, time_limit))

                for npack in [1, 2, 4, 8, 16]:
                    print(f'time srun ./benchmarks/bm_pack {dim} {radius} {ndims} {npack} {nsamples} {jobid} {jobname}')

    # OK
    ndims = 3
    npack = 1
    for nprocs in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
        for dim in dims:

            jobname = f"comm-scaling-strong-dim-{dim}"
            with open(f'bm-{system.name}-{jobname}-{nprocs}.sh', 'w') as f:
                with redirect_stdout(f):
                    print(system.gen_preamble(nprocs, time_limit))
                    print(f'time srun ./benchmarks/bm_collective_comm {dim} {radius} {ndims} {npack} {nsamples} {jobid} {jobname}')
                

    # OK
    ndims = 3
    nprocs = 64
    for dim in dims:
        
        global_dim = get_global_dim(nprocs, dim, ndims)
        assert(global_dim == 4*dim)
        jobname = f"comm-aggr-dim-{global_dim}"

        with open(f'bm-{system.name}-{jobname}.sh', 'w') as f:
            with redirect_stdout(f):
                print(system.gen_preamble(nprocs, time_limit))
                for npack in [1, 2, 4, 8, 16]:
                    print(f'time srun ./benchmarks/bm_collective_comm {global_dim} {radius} {ndims} {npack} {nsamples} {jobid} {jobname}')

    # OK
    nprocs = 64
    for dim in dims:
        jobname = f"comm-nd-dim-{dim}"

        
        with open(f'bm-{system.name}-{jobname}.sh', 'w') as f:
            with redirect_stdout(f):
                print(system.gen_preamble(nprocs, time_limit))

                for ndims in range(1, 4):
                    global_dim = get_global_dim(nprocs, dim, ndims)
                    print(f'time srun ./benchmarks/bm_collective_comm {global_dim} {radius} {ndims} {npack} {nsamples} {jobid} {jobname}')

    # OK
    nprocs = 64
    r = 4
    dims = np.array([
        [1024, 1024, 1024],
        [2048, 1024,  512],
        [2048, 2048,  256],
        [ 512, 1024, 2048],
        [ 256, 2048, 2048]
    ])

    for dim in dims:

        jobname = f"rank-reorder-{dim[0]}-{dim[1]}-{dim[2]}"
        
        with open(f'bm-{system.name}-{jobname}.sh', 'w') as f:
            with redirect_stdout(f):
                print(system.gen_preamble(nprocs, time_limit))

                print(f'time srun ./benchmarks/bm_rank_reordering {dim[0]} {dim[1]} {dim[2]} {radius} {nsamples} {jobid} {jobname} 0') # Non-hiearchical
                print(f'time {gen_srun_command(system.cpu_bind)} ./benchmarks/bm_rank_reordering {dim[0]} {dim[1]} {dim[2]} {radius} {nsamples} {jobid} {jobname} 1') # Hierarchical

    # Placeholder
    # for dim in dims:
        
    #     for ndims in range(6):
    #         jobname = f"comm-nd-dim-{dim}"

    #         global_dim = get_global_dim(nprocs, dim, ndims)
    #         with open(f'bm-{system.name}-{jobname}.sh', 'w') as f:
    #             with redirect_stdout(f):
    #                 print(system.gen_preamble(nprocs, time_limit))
    #                     print(f'time srun ./benchmarks/bm_collective_comm {global_dim} {radius} {ndims} {npack} {nsamples} {jobid} {jobname}')

    # dims =  [64, 256, 1024]
    # nprocs = 64  
    # for dim in dims:
    #     jobname = f"comm-nd-dim-{dim}"
    #     npack = 1
    #     with open(f'bm-{system.name}-{jobname}.sh', 'w') as f:
    #         with redirect_stdout(f):
    #             print(system.gen_preamble(nprocs, time_limit))

    #             # ND
    #             for ndims in range(1, 6):
    #                 if dim**ndims > dims[-1]**3: # Skip way too large dims
    #                     continue
    #                 print(f'time srun ./benchmarks/bm_collective_comm {dim} {radius} {ndims} {npack} {nsamples} {jobid} {jobname}')
    #             print()

    #     jobname = f"comm-aggr-dim-{dim}"
    #     ndims = 3
    #     with open(f'bm-{system.name}-{jobname}.sh', 'w') as f:
    #         with redirect_stdout(f):
    #             print(system.gen_preamble(nprocs, time_limit))

    #             for npack in [1, 2, 4, 8, 16]:
    #                 print(f'time srun ./benchmarks/bm_collective_comm {dim} {radius} {ndims} {npack} {nsamples} {jobid} {jobname}')

def gen_comm_benchmarks_old(system):

    nn = 128**3
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
                print(f'time srun ./benchmarks/bm_collective_comm {dim} {radius} {ndims} {npack} {nsamples} {jobid} {jobname}')

    jobname = "\"nd\""
    nprocs = 64
    npack = 1
    with open(f'bm-{system.name}-comm-nd-{nprocs}.sh', 'w') as f:
        with redirect_stdout(f):
            print(system.gen_preamble(nprocs, time_limit))
            for ndims in range(1, 6):
                dim = int(np.round(nn**(1/ndims)))
                print(f'time srun ./benchmarks/bm_collective_comm {dim} {radius} {ndims} {npack} {nsamples} {jobid} {jobname}')

    jobname = "\"aggr\""
    nprocs = 64
    ndims = 3
    dim = int(np.round(nn**(1/ndims)))
    with open(f'bm-{system.name}-comm-aggr-{nprocs}.sh', 'w') as f:
        with redirect_stdout(f):
            print(system.gen_preamble(nprocs, time_limit))
            for npack in [1, 2, 4, 8, 16]:
                print(f'time srun ./benchmarks/bm_collective_comm {dim} {radius} {ndims} {npack} {nsamples} {jobid} {jobname}')

def gen_rank_reordering_benchmarks_old(system):
    nsamples = 100
    time_limit = "00:15:00"
    jobid = "$SLURM_JOB_ID"
    radius = 4

    cases = np.array((
        np.array([256, 256, 256])//2,
        np.array([512, 256, 128])//2,
        np.array([512, 512, 64])//2,
        np.array([128, 256, 512])//2,
        np.array([64, 512, 512])//2,
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
                    print(f'time srun ./benchmarks/bm_rank_reordering {nn[0]} {nn[1]} {nn[2]} {radius} {nsamples} {jobid} {jobname} 0') # Non-hiearchical
                    print(f'time {gen_srun_command(system.cpu_bind)} ./benchmarks/bm_rank_reordering {nn[0]} {nn[1]} {nn[2]} {radius} {nsamples} {jobid} {jobname} 1') # Hierarchical


    # NOTE: modifies cases
    max_nprocs = 256
    nprocs = 1
    axis = ndims-1
    while nprocs <= max_nprocs:
        with open(f'bm-{system.name}-rank-reordering-scaling-weak-{nprocs}.sh', 'w') as f:
            with redirect_stdout(f):
                print(system.gen_preamble(nprocs, time_limit))

                for i, nn in enumerate(cases):
                    # print(f'nn: {nn}')
                    # print(f'nprocs: {nprocs}')
                    # print(f'axis: {axis}')
                    jobname = f"\"scaling-weak-case-{i}\""
                    print(f'time srun ./benchmarks/bm_rank_reordering {nn[0]} {nn[1]} {nn[2]} {radius} {nsamples} {jobid} {jobname} 0') # Non-hiearchical
                    print(f'time {gen_srun_command(system.cpu_bind)} ./benchmarks/bm_rank_reordering {nn[0]} {nn[1]} {nn[2]} {radius} {nsamples} {jobid} {jobname} 1') # Hierarchical
                    nn[axis] *=2
                    
                nprocs *= 2
                axis = (axis + len(nn) - 1) % len(nn)



systems = [lumi]

for system in systems:
    gen_pack_benchmarks(system)
    # gen_comm_benchmarks(system)
    # gen_rank_reordering_benchmarks(system)