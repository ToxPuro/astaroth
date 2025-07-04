#!/usr/bin/env python3

def gen_sbatch_preamble(nprocs, devices_per_node, account, time_limit, partition):
    ntasks_per_node = min(nprocs, devices_per_node)
    nnodes = nprocs // ntasks_per_node
    assert(ntasks_per_node * nnodes == nprocs)

    return f"""#!/usr/bin/env bash
#SBATCH --account={account}
#SBATCH --time={time_limit}
#SBATCH --partition={partition}
#SBATCH --gpus-per-node={ntasks_per_node}
#SBATCH --ntasks-per-node={ntasks_per_node}
#SBATCH --nodes={nnodes}
"""

def gen_srun_command(cpu_bind = ""):
    return f'srun --cpu-bind="{cpu_bind}"' if cpu_bind else "srun"

def gen_run_information():
    return """
module list
cmake -LAH >> system_info-$SLURM_JOB_ID.txt
"""

def gen_run_information_tfm(config, label):
    return f"cp {config} config-$SLURM_JOB_ID-{label}.ini\n"

def gen_run_command_tfm(config, global_nn_override, label):
    formatted_global_nn_override = ",".join(map(str, global_nn_override))

    return f"""./tfm-mpi \\
          --config {config} \\
          --global-nn-override {formatted_global_nn_override} \\
          --job-id $SLURM_JOB_ID \\
          --benchmark 1 \\
          --benchmark-name {label}
"""


class System:
    def __init__(self, account, partition, devices_per_node, modules, env_vars, cpu_bind):
        self.account = account
        self.partition = partition
        self.devices_per_node = devices_per_node
        self.modules = modules
        self.env_vars = env_vars
        self.cpu_bind = cpu_bind

    def gen_preamble(self, nprocs, time_limit):
        return gen_sbatch_preamble(nprocs, self.devices_per_node, self.account, time_limit, self.partition) + \
               self.modules + \
               self.env_vars + \
               gen_run_information()
    
    def gen_srun_command(self, nprocs):
        if nprocs >= self.devices_per_node:
            return gen_srun_command(self.cpu_bind)
        else:
            return gen_srun_command()


lumi = System(
    account = "project_462000987",
    #partition = "dev-g",
    partition = "standard-g",
    devices_per_node = 8,
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

def gen_tfm_benchmark(system, nprocs, time_limit, config, nn, label):
    return system.gen_preamble(nprocs, time_limit) + \
           gen_run_information_tfm(config, label) + \
           f'{system.gen_srun_command(nprocs)} {gen_run_command_tfm(config, nn, label)}'

def gen_weak_scaling_benchmarks(system, max_nprocs):
    
    nn = [256,256,256]
    axis = len(nn) - 1
    time_limit = "00:15:00"
    config = "/users/pekkila/astaroth/samples/tfm/mhd/mhd.ini"
    label = "weak"

    nprocs = 1
    while nprocs <= max_nprocs:
        with open(f'bm-{label}-{nprocs}.sh', 'w') as f:
            print(gen_tfm_benchmark(system, nprocs, time_limit, config, nn, label), file=f)
        
        nprocs *= 2
        nn[axis] *= 2
        axis = (axis + len(nn) - 1) % len(nn)
        
def gen_strong_scaling_benchmarks(system, max_nprocs):
    nn = [256,256,256]
    time_limit = "00:15:00"
    config = "/users/pekkila/astaroth/samples/tfm/mhd/mhd.ini"
    label = "strong"

    nprocs = 2
    while nprocs <= max_nprocs:
        with open(f'bm-{label}-{nprocs}.sh', 'w') as f:
            print(gen_tfm_benchmark(system, nprocs, time_limit, config, nn, label), file=f)
        nprocs *= 2

gen_weak_scaling_benchmarks(lumi, 4096)
gen_strong_scaling_benchmarks(lumi, 256)


# import numpy as np
# import copy

# def gen_weak_scaling_grid_dim_map(nn, max_nprocs):
#     dims = [copy.deepcopy(nn)]

#     axis = len(nn) - 1
    
#     nprocs = 1
#     while nprocs <= max_nprocs:
#         nprocs *= 2
#         nn[axis] *= 2
#         dims += copy.deepcopy([nn])
#         axis = (axis + len(nn) - 1) % len(nn)
    
#     return dims


# def gen_trace_profiler_command_rocm(rocprof_wrapper):
#     return f"{rocprof_wrapper} --hip-trace --trace-start off"

# def gen_tfm_profile_benchmark(system, nprocs, time_limit, config, nn, label, rocprof_wrapper):
#     return system.gen_preamble(nprocs, time_limit) + \
#            gen_run_information_tfm(config, label) + \
#            f'{lumi.gen_srun_command(nprocs)} {gen_trace_profiler_command_rocm(rocprof_wrapper)} {gen_run_command_tfm(config, nn, label)}'

# def gen_profiling_benchmarks(system, proc_counts):
#     initial_nn = [256,256,256]
#     time_limit = "00:15:00"
#     config = "/users/pekkila/astaroth/samples/tfm/mhd/mhd.ini"
#     rocprof_wrapper = "/users/pekkila/astaroth/samples/tfm-mpi/rocprof-wrapper.sh"
#     label = "prof"

#     for nprocs in proc_counts:
#         nn = initial_nn
#         axis = len(nn) - 1
#         if np.prod(initial_nn) < np.prod(nn):
#             nn[axis] *= 2
#             axis = (axis + len(nn) - 1) % len(nn)
#         assert(np.prod(initial_nn) == np.prod(nn))

#         with open(f'bm-{label}-{nprocs}.sh', 'w') as f:
#            print(gen_tfm_profile_benchmark(system, nprocs, time_limit, config, nn, label, rocprof_wrapper), file=f)
#         nprocs *= 2
        

# gen_profiling_benchmarks(lumi, [1, 8])
# print(gen_weak_scaling_grid_dim_map([128,128,128], 64))

# # %%
# import numpy as np
# print("hello")

# max_power = 13
# proc_counts = [2**i for i in range(max_power)]
# print(proc_counts)

# for nprocs in proc_counts:


# #while nprocs <= max_nprocs:


# /users/pekkila/astaroth/samples/tfm-mpi/rocprof-wrapper.sh --hip-trace --trace-start off