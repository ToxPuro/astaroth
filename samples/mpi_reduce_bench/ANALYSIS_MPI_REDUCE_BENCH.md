# ⚠️ WARNING ⚠️
# **This is a LLM based analysis.**
# **THE BRANCH SHOULD NEVER BE MERGED TO master OR develop BRANCHES!**

# LLM Information
- **Model:** Qwen3.6-35B-A3B

# Overview
The `mpi_reduce_bench` sample is an MPI reduction operation benchmark for Astaroth's GPU-accelerated grid infrastructure. It measures the latency of scalar and vector reduction operations across an MPI-distributed grid, using the 90th percentile of 100 iterations as the reported metric. Scalar reductions operate on a single vertex buffer field (VTXBUF_UUX), while vector reductions operate on three fields simultaneously (VTXBUF_UUX, VTXBUF_UUY, VTXBUF_UUZ). Five reduction types are benchmarked: MAX, MIN, SUM, RMS, and RMS_EXP. Results are appended to a CSV file (`mpi_reduction_benchmark.csv`) with fields: benchmark label, test label, process count, and measured latency in milliseconds. The benchmark is designed to profile how reduction costs scale with the number of MPI processes.

# Directory Structure & File Descriptions

| File | Description |
| :--- | :--- |
| `CMakeLists.txt` | Build configuration; compiles `main.cc` into the `mpi_reduce_bench` executable, linked against `astaroth_core` and `astaroth_utils`. Minimal — just `add_executable` and `target_link_libraries`. |
| `main.cc` | Reduction benchmark driver (177 lines): initializes MPI and Astaroth grid, defines scalar and vector test case arrays, runs 100 iterations per test with CUDA stream synchronization, collects timing, reports 90th percentile, and appends results to CSV. |
| `mpibench.sh` | SLURM batch submission script (84 lines): wraps `mpi_reduce_bench` for HPC cluster execution, configurable process count, partition, and benchmark tag. Default is 8 processes across 2 nodes on the `gpu` partition. |

# Compile-Time Requirements

The code has a single preprocessor guard:

| Guard | Behavior if not defined |
| :--- | :--- |
| `AC_MPI_ENABLED` | Prints error message, returns `EXIT_FAILURE`. Astaroth must be built with `cmake -DMPI_ENABLED=ON`. |

# Compile-Time Options

None defined in the source. The benchmark uses fixed constants for iterations (`num_iters = 100`) and the percentile threshold (`nth_percentile = 0.90`).

# Input Parameters / Command-Line Interface

| Argument | Position | Default | Description |
| :--- | :--- | :--- | :--- |
| `benchmark_label` | argv[1] | ISO timestamp of `std::chrono::system_clock::now()` | A label prepended to the CSV output. When run via `mpibench.sh`, this is the short git HEAD hash (from `git rev-parse --short HEAD`). |

Usage: `mpirun -np <num_processes> ./mpi_reduce_bench [benchmark_label]`

# Program Flow

1. **MPI Init**: `MPI_Init`, get rank (`pid`) and total processes (`nprocs`).

2. **Benchmark Label**: If `argv[1]` is provided, use it as `benchmark_label`. Otherwise, generate an ISO timestamp from the current system clock.

3. **Mesh Configuration**: Load `AC_DEFAULT_CONFIG` into `AcMeshInfo info` via `acLoadConfig`. This provides the default field definitions and grid configuration from Astaroth's DSL-generated config.

4. **Test Case Definitions**:
   - **Scalar reductions** (`std::vector<AcScalReductionTestCase>`): 5 test cases, all operating on `VTXBUF_UUX` with reduction types MAX, MIN, RMS, RMS_EXP, and SUM.
   - **Vector reductions** (`std::vector<AcVecReductionTestCase>`): 5 test cases, operating on three fields (`VTXBUF_UUX`, `VTXBUF_UUY`, `VTXBUF_UUZ`) with the same reduction types.

5. **Grid Init**: `acGridInit(info)` — initializes the Astaroth grid infrastructure, allocating GPU memory for all vertex buffers.

6. **Scalar Reduction Benchmarking**:
   For each of the 5 scalar test cases:
   a. Loop 100 times:
      - `acGridSynchronizeStream(STREAM_ALL)` — ensure all GPU streams are idle.
      - `timer_reset(&t)` — reset the high-resolution timer.
      - `acGridSynchronizeStream(STREAM_ALL)` — re-sync (double barrier to ensure accurate timing).
      - `acGridReduceScal(STREAM_DEFAULT, rtype, vtxbuf, &candidate)` — launch the scalar reduction on `STREAM_DEFAULT` (STREAM_0).
      - `acGridSynchronizeStream(STREAM_ALL)` — wait for reduction to complete.
      - Record elapsed milliseconds (`timer_diff_nsec(t) / 1e6`).
   b. Sort results and select the 90th percentile value.
   c. Rank 0 prints the percentile to stdout and appends to `mpi_reduction_benchmark.csv`.

7. **Vector Reduction Benchmarking**:
   Same pattern as scalar, but using `acGridReduceVec(STREAM_DEFAULT, rtype, vtxbuf_a, vtxbuf_b, vtxbuf_c, &candidate)` with 3 vertex buffers.

8. **Cleanup**: `acGridQuit()`, `MPI_Finalize()`, return `EXIT_SUCCESS`.

# CSV Output Format

```
"benchmark_label","test_label", nprocs, latency_ms
```

Example rows:
```
"2026-07-10 12:00:00","Scalar MAX", 8, 1.23456
"2026-07-10 12:00:00","Vector SUM", 8, 2.34567
"a1b2c3d","Scalar RMS", 32, 3.45678
```

The file is opened in append mode (`"a"`), so repeated runs accumulate data. Only rank 0 writes.

# Astaroth Grid APIs Used

| Function | Description |
| :--- | :--- |
| `acGridInit(info)` | Initialize grid infrastructure with mesh config from `AC_DEFAULT_CONFIG`. |
| `acGridQuit()` | Shutdown grid infrastructure, free GPU memory. |
| `acGridSynchronizeStream(stream)` | Block until all operations on the specified stream(s) have completed. Used heavily for accurate timing. |
| `acGridReduceScal(stream, reduction, vtxbuf_handle, result)` | **Core scalar reduction**: Compute a single-value reduction (MAX/MIN/SUM/RMS/RMS_EXP) across all ranks for a single vertex buffer field. Result is a single `AcReal` value broadcast to all ranks. |
| `acGridReduceVec(stream, reduction, a, b, c, result)` | **Core vector reduction**: Compute a single-value reduction across all ranks for three vertex buffer fields simultaneously (e.g., 3D velocity magnitude). Result is a single `AcReal` value broadcast to all ranks. |

# Astaroth Host APIs Used

| Function | Description |
| :--- | :--- |
| `acLoadConfig(config, &info)` | Load mesh configuration from `AC_DEFAULT_CONFIG`. Defines field layouts, grid topology, and initialization conditions. |

# Reduction Types

Defined in `stdlib/reduction.h` as `const AcReduction` structs:

| Constant | Reduction Op | Post-processing | Single-field map | Three-field map |
| :--- | :--- | :--- | :--- | :--- |
| `RTYPE_MAX` | `AC_REDUCE_OP_MAX` | None | `AC_MAP_VTXBUF` | `AC_MAP_VTXBUF3_NORM` |
| `RTYPE_MIN` | `AC_REDUCE_OP_MIN` | None | `AC_MAP_VTXBUF` | `AC_MAP_VTXBUF3_NORM` |
| `RTYPE_SUM` | `AC_REDUCE_OP_SUM` | None | `AC_MAP_VTXBUF` | `AC_MAP_VTXBUF3_NORM` |
| `RTYPE_RMS` | `AC_REDUCE_OP_SUM` | RMS (sqrt of mean of squares) | `AC_MAP_VTXBUF_SQUARE` | `AC_MAP_VTXBUF3_SQUARE` |
| `RTYPE_RMS_EXP` | `AC_REDUCE_OP_SUM` | RMS (with exponential squaring) | `AC_MAP_VTXBUF_EXP_SQUARE` | `AC_MAP_VTXBUF3_EXP_SQUARE` |

# MPI APIs Used

| Function | Description |
| :--- | :--- |
| `MPI_Init(NULL, NULL)` | Initialize MPI. |
| `MPI_Comm_rank(MPI_COMM_WORLD, &pid)` | Get rank ID. |
| `MPI_Comm_size(MPI_COMM_WORLD, &nprocs)` | Get total number of processes. |
| `MPI_Finalize()` | Shutdown MPI. |

# Timer Utilities

| Function | Description |
| :--- | :--- |
| `timer_reset(&t)` | Reset a high-resolution timer. From `timer_hires.h`. |
| `timer_diff_nsec(t)` | Return elapsed time in nanoseconds since last reset. |

# Preprocessor Constants / Definitions

| Constant | Description |
| :--- | :--- |
| `STREAM_DEFAULT` | Defaults to `STREAM_0` (defined in `astaroth_base.h`). |
| `STREAM_ALL` | Sentinel value meaning "all streams" for synchronization. |
| `VTXBUF_UUX` | Vertex buffer handle for x-component of velocity (used as the single-field input for scalar reductions). |
| `VTXBUF_UUY` | Vertex buffer handle for y-component (used in vector reductions). |
| `VTXBUF_UUZ` | Vertex buffer handle for z-component (used in vector reductions). |
| `num_iters` | 100 — number of iterations per test case. |
| `nth_percentile` | 0.90 — the 90th percentile is used as the reported metric. |

# Notable Observations

1. **90th percentile metric**: The benchmark reports the 90th percentile of 100 iterations rather than the mean or median. This is a common choice for low-latency benchmarking — it reduces the impact of OS scheduling jitter and GC pauses while still capturing typical (not best-case) performance.

2. **Double synchronization per iteration**: `acGridSynchronizeStream(STREAM_ALL)` is called immediately before and after `timer_reset(&t)`. This ensures that any residual GPU work from previous iterations is fully complete before timing begins, and that the timer starts from a clean slate. However, the sync before `acGridReduceScal`/`acGridReduceVec` already provides this guarantee — the extra sync pair around `timer_reset` may be redundant.

3. **Stream 0 for reductions**: All reductions are launched on `STREAM_DEFAULT` (STREAM_0). This means reductions execute sequentially across test cases. There's no overlap — each test case fully completes (including synchronization) before the next one begins.

4. **Per-iteration synchronization overhead**: Each iteration includes three `acGridSynchronizeStream(STREAM_ALL)` calls (before timer, after timer, after reduction). This means the measured latency includes GPU synchronization overhead, not just the raw reduction time. The synchronization is necessary for correctness but inflates the reported numbers.

5. **Only rank 0 writes CSV**: The `if (!pid)` guard ensures only rank 0 writes results. This is appropriate for a benchmark that outputs to a single file. All ranks participate in the reduction computation, but only one process handles file I/O.

6. **Append mode, no file reset**: The CSV is opened in append mode (`"a"`). Running the benchmark multiple times accumulates data, which is convenient for comparing different configurations but means stale results from previous runs remain unless manually cleared.

7. **No grid size variation**: The grid configuration comes entirely from `AC_DEFAULT_CONFIG` — there's no way to change the grid dimensions at runtime. The only variable is the number of MPI processes. This means the benchmark measures reduction latency as a function of process count for a fixed grid size, which is useful for scaling analysis but not for studying the effect of grid size on reduction cost.

8. **Scalar and vector reductions use the same 5 ops**: Both scalar and vector test cases benchmark the exact same five reduction types (MAX, MIN, SUM, RMS, RMS_EXP). The distinction is that scalar operates on one field and vector operates on three fields simultaneously. The three-field versions likely add per-element computation (e.g., normalizing 3D vectors for MIN/MAX, or computing the squared magnitude for RMS), which may affect GPU occupancy and memory bandwidth utilization.

9. **`VTXBUF_UUX` as the canonical field**: Scalar reductions always operate on `VTXBUF_UUX`. This is the x-component of the velocity field, which is a reasonable default for a single-field benchmark but doesn't represent all field types (e.g., it doesn't test scalar fields like pressure or density).

10. **No error checking on MPI calls**: `MPI_Init`, `MPI_Comm_rank`, `MPI_Comm_size`, and `MPI_Finalize` have no error checking. An MPI failure would produce undefined behavior rather than a clear error message. This is consistent with the `mpi-io` sample's approach.

11. **Uses `acLoadConfig` rather than explicit mesh setup**: The grid is configured entirely through Astaroth's config system (`AC_DEFAULT_CONFIG`). This is a black-box approach — the actual grid dimensions, number of fields, and initialization conditions depend on the DSL-generated config. To know the exact mesh size being benchmarked, one would need to inspect the config file.

12. **CUDA-aware MPI implied**: `acGridReduceScal` and `acGridReduceVec` operate on GPU-resident vertex buffers. This implies the Astaroth library uses CUDA-aware MPI (or an equivalent HIP-aware mechanism) to perform the reduction directly on GPU memory without staging data through host memory. This is critical for performance.

13. **100 iterations may be insufficient**: For operations that complete in microseconds, 100 iterations provides only 100 samples. Given that the 90th percentile of 100 values is the 90th sorted value, the resolution of the percentile estimate is coarse (±10%). For high-precision benchmarking, more iterations (e.g., 1000+) would be preferable.

14. **No per-process output**: Unlike the `mpi-io` sample, `mpi_reduce_bench` produces a single shared CSV file (`mpi_reduction_benchmark.csv`) written by rank 0. If multiple runs of `mpi_reduce_bench` execute concurrently on different nodes, they will contend for the same file unless they use different filenames.

15. **SLURM script hardcodes V100s**: `mpibench.sh` requests `--gres=gpu:v100:4`, tying the benchmark to NVIDIA V100 GPUs with 4 GPUs per node. This limits portability to other GPU types (e.g., A100, MI250x).

16. **Process-to-node calculation**: `num_nodes = 1 + (num_procs - 1) / 4` assumes 4 GPUs per node. Combined with the V100 reservation, this reflects a design assumption that the target hardware has exactly 4 GPUs per node.

17. **Benchmark label flexibility**: The `benchmark_label` supports human-readable timestamps (default) or arbitrary strings (e.g., git commit hashes from `mpibench.sh`). This enables tracking benchmark results against specific code revisions in the CSV output.

18. **No vector/scalar comparison analysis**: The benchmark measures scalar and vector reductions separately but does not include any analysis comparing the two. A useful extension would be to compute the ratio `vector_time / scalar_time` to quantify the overhead of processing three fields simultaneously versus one.

19. **No scaling curve fitting**: The CSV output is raw data — there's no built-in analysis for plotting scaling curves, fitting strong/weak scaling models, or extracting communication cost parameters (e.g., MPI Allreduce latency and bandwidth constants).
