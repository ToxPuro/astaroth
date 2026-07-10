# ⚠️ WARNING ⚠️
# **This is a LLM based analysis.**
# **THE BRANCH SHOULD NEVER BE MERGED TO master OR develop BRANCHES!**

# LLM Information
- **Model:** Qwen3.6-35B-A3B

# Overview
The `genbenchmarkscripts` is a C utility that generates SLURM batch scripts (`benchmark_*.sh`) for running Astaroth benchmarks on HPC clusters. It produces 6 scripts for processor counts of 1, 2, 4, 8, 16, 32, and 64. Each script contains SLURM directives, module loading commands, and a sequence of 22 benchmark subdirectory runs. It is a code-generation tool rather than a runtime benchmark itself — the generated scripts invoke the `./benchmark` executables from the benchmark subdirectories.

# Directory Structure & File Descriptions

| File | Description |
| :--- | :--- |
| `CMakeLists.txt` | Build configuration; compiles `main.c` into `genbenchmarkscripts` executable and adds a `POST_BUILD` custom command that automatically runs the binary after compilation, generating scripts in `${PROJECT_BINARY_DIR}`. |
| `main.c` | Script generator: iterates over powers of 2 from 1 to 64 MPI ranks, writes SLURM batch files with hardware directives, module loads, and 22 benchmark run commands. |

# Precondition

| Condition | Macro/Requirement | Description |
| :--- | :--- | :--- |
| C compiler available | (CMake default) | Required for build |
| HPC SLURM scheduler | SLURM environment | Generated scripts target SLURM (not PBS, SGE, etc.) |
| V100 GPU nodes | `--gres=gpu:v100:*` | Scripts hardcode NVIDIA V100 GPUs |
| Puhti cluster (or similar) | Misconfigured node exclusion list | Scripts exclude specific known-bad nodes on the Puhti cluster |

# Program Flow

The program generates one `.sh` file per processor count (powers of 2, 1–64):

1. **Loop over `nprocs`**: 1, 2, 4, 8, 16, 32, 64.
2. **Compute resource allocation**:
   - `gpus_per_node = min(nprocs, 4)` — max 4 GPUs per node.
   - `nodes = ceil(nprocs / 4)` — how many nodes needed.
3. **Write SLURM header** (boilerplate):
   - Job name: `astaroth`
   - Account: `project_2000403`
   - Wall time: 4 hours
   - Partition: `gpu`
   - Exclusive allocation
   - 10 CPUs per task
   - GPU resource: `v100` × `gpus_per_node`
   - Task/node count: `-n nprocs`, `-N nodes`
   - Output file: `benchmark-%d-<jobid>.out`
   - **Exclusion list**: hardcoded list of misconfigured Puhti nodes (r04g05-06, r02g02, r14g04, etc.)
4. **Write module loading**: loads `gcc/8.3.0`, `cuda/10.1.168`, `cmake`, `openmpi/4.0.3-cuda`, `nccl`.
5. **Write benchmark run commands** (22 benchmarks):
   - For each benchmark subdirectory, `cd` into it, run `srun ./benchmark <nn> <nn> <nn>`, clean core dumps, `cd ..`.
   - On multi-node runs (`nodes >= 2`): sets UCX environment variables for GPU direct (`UCX_RNDV_THRESH=16384`, `UCX_RNDV_SCHEME=get_zcopy`, `UCX_MAX_RNDV_RAILS=1`) and adds `--kill-on-bad-exit=0`.
   - Single-node runs: no UCX overrides, just `srun` with `--kill-on-bad-exit=0`.

# Generated File Names

| `nprocs` | Output File |
| :--- | :--- |
| 1 | `benchmark_1.sh` |
| 2 | `benchmark_2.sh` |
| 4 | `benchmark_4.sh` |
| 8 | `benchmark_8.sh` |
| 16 | `benchmark_16.sh` |
| 32 | `benchmark_32.sh` |
| 64 | `benchmark_64.sh` |

# Benchmark Subdirectories Invoked

The generated scripts run 22 benchmarks in order (each from its own subdirectory):

| # | Subdirectory | Grid Size | Description |
|---|-------------|-----------|-------------|
| 1 | `benchmark_decomp_1D` | 256³ | 1D domain decomposition benchmark |
| 2 | `benchmark_decomp_2D` | 256³ | 2D domain decomposition benchmark |
| 3 | `benchmark_decomp_3D` | 256³ | 3D domain decomposition benchmark |
| 4 | `benchmark_decomp_1D_comm` | 256³ | 1D decomp with communication timing |
| 5 | `benchmark_decomp_2D_comm` | 256³ | 2D decomp with communication timing |
| 6 | `benchmark_decomp_3D_comm` | 256³ | 3D decomp with communication timing |
| 7 | `benchmark_meshsize_256` | 256³ | Mesh size scaling baseline |
| 8 | `benchmark_meshsize_512` | 512³ | Mesh size scaling — medium |
| 9 | `benchmark_meshsize_1024` | 1024³ | Mesh size scaling — large |
| 10 | `benchmark_meshsize_2048` | 2048³ | Mesh size scaling — very large |
| 11 | `benchmark_stencilord_2` | 256³ | Stencil order 2 |
| 12 | `benchmark_stencilord_4` | 256³ | Stencil order 4 |
| 13 | `benchmark_stencilord_6` | 256³ | Stencil order 6 |
| 14 | `benchmark_stencilord_8` | 256³ | Stencil order 8 |
| 15 | `benchmark_timings_control` | 256³ | Control timings |
| 16 | `benchmark_timings_comp` | 256³ | Compute-bound timings |
| 17 | `benchmark_timings_comm` | 256³ | Communication-bound timings |
| 18 | `benchmark_timings_default` | 256³ | Default timings |
| 19 | `benchmark_timings_corners` | 256³ | Corner case timings |
| 20 | `benchmark_weak_128` | 128³ | Weak scaling benchmark at 128³ |
| 21 | `benchmark_weak_256` | 256³ | Weak scaling benchmark at 256³ |
| 22 | `benchmark_weak_512` | 512³ | Weak scaling benchmark at 512³ |

# Key Environment Variables & SLURM Directives

| Variable/Directive | Value | Purpose |
|--------------------|-------|---------|
| `#SBATCH --gres=gpu:v100:<N>` | N = 1–4 | GPU resource allocation |
| `#SBATCH -x` | Node exclusion list | Exclude misconfigured nodes |
| `UCX_RNDV_THRESH` | 16384 | UCX rendezvous threshold (multi-node only) |
| `UCX_RNDV_SCHEME` | `get_zcopy` | UCX rendezvous scheme (multi-node only) |
| `UCX_MAX_RNDV_RAILS` | 1 | Limit rendezvous rails (multi-node only) |
| `srun --kill-on-bad-exit=0` | 0 | Do not kill other tasks on individual failure |

# Notable Observations

1. **Code generation, not runtime**: This tool generates shell scripts — it does not perform any simulation or benchmarking itself. The actual benchmark executables live in separate subdirectories (e.g., `samples/genbenchmarkscripts/benchmark_decomp_1D/`).

2. **Hardcoded cluster assumptions**: The scripts are tuned for the **Puhti** HPC cluster at CSC (Finland) with specific node exclusion lists, account IDs, and module versions. They may not work on other clusters without modification.

3. **CUDA 10.1 / OpenMPI 4.0.3**: The module loading uses relatively old versions (gcc 8.3.0, cuda 10.1.168, openmpi 4.0.3-cuda), reflecting the cluster's configuration at the time the generator was written.

4. **UCX GPU direct tuning**: On multi-node runs, UCX is tuned for GPU direct communication (`gdr_copy`, `cuda_ipc`). The commented-out Fredrik tunings (`UCX_RNDV_THRESH=16384`) suggest these are actively experimented with.

5. **GPU cap per node**: Limits to 4 GPUs per node — reflects hardware topology (likely 4-way GPU nodes on Puhti).

6. **Power-of-2 scaling**: Tests only powers of 2 (1–64), suggesting the benchmarks focus on strong/weak scaling analysis rather than fine-grained process counts.

7. **Core dump cleanup**: Each benchmark run includes `rm -f core.*` to prevent core dumps from filling up disk space on shared filesystems.

8. **Commented-out profiling**: The nvprof profiling code (lines 60–71) is commented out, suggesting profiling was used during development but removed from the generated scripts.

9. **Auto-generate on build**: The `POST_BUILD` custom command in CMakeLists.txt ensures scripts are regenerated every time the project is rebuilt, so they stay in sync with the latest binary.
