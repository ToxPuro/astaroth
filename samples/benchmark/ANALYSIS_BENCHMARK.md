# ⚠️ WARNING ⚠️
# **This is a LLM based analysis.**
# **THE BRANCH SHOULD NEVER BE MERGED TO master OR develop BRANCHES!**

# LLM Information
- **Model:** Qwen3.6-35B-A3B

# Overview
The `benchmark` sample is Astaroth's core performance benchmarking tool. It measures integration step execution time across varying problem sizes and processor counts to evaluate both strong and weak scaling characteristics. The benchmark can also perform numerical verification against a reference CPU implementation to ensure correctness of the GPU-accelerated integration.

# Directory Structure & File Descriptions

| File | Description |
| :--- | :--- |
| `CMakeLists.txt` | Build configuration; compiles `main.cc` into the `benchmark` executable with position-independent code, linked against `astaroth_core` and `astaroth_utils`. |
| `main.cc` | The benchmark implementation: MPI-based parallel execution, argument parsing for test type and mesh dimensions, integration step timing, percentile statistics, optional verification, and per-kernel sanity checks. |

# Precondition Checks

The executable requires two conditions to run, checked at compile time via preprocessor guards:

| Condition | Macro | Error Message |
| :--- | :--- | :--- |
| MPI support | `AC_MPI_ENABLED` | "The library was built without MPI support, cannot run." |
| Integration DSL | `AC_INTEGRATION_ENABLED` | "The library was built without AC_INTEGRATION_ENABLED..." |

# Command-Line Interface

| Argument | Description |
| :--- | :--- |
| `-t <type>` | Test type: `strong` for strong scaling, `weak` for weak scaling. |
| `<nx> <ny> <nz>` | Base grid dimensions (required positional arguments). |
| `<verify>` (optional) | Set to `1` to enable CPU-vs-GPU verification; `0` skips it. |

Usage: `./benchmark -t strong <nx> <ny> <nz> <verify>`

# Program Flow

The benchmark follows this execution path:

## 1. Initialization
1. `MPI_Init` — Start MPI.
2. `acProfilerStop` — Ensure profiler is disabled during benchmarking.
3. `acInitInfo()` / `acLoadConfig(AC_DEFAULT_CONFIG, &info)` / `acHostUpdateParams(&info)` — Set up the mesh info object.
4. Configure decomposition and communication strategies:
   - `AC_PROC_MAPPING_STRATEGY_MORTON` — Morton curve mapping.
   - `AC_DECOMPOSE_STRATEGY_MORTON` — Morton curve decomposition.
   - `AC_MPI_COMM_STRATEGY_DUP_WORLD` — Duplicate MPI_COMM_WORLD.
5. (Optional) Runtime compilation and dynamic library loading if `AC_RUNTIME_COMPILATION` is defined.

## 2. Grid Setup
Depends on the test type:

**Strong Scaling** — Fixed total grid size, decreasing local work per process:
- `AC_ngrid = (nx, ny, nz)` — Total domain size.
- `AC_nlocal = (nx/decomp.x, ny/decomp.y, nz/decomp.z)` — Per-process subdomain.

**Weak Scaling** — Fixed local grid size, growing total domain with processes:
- `AC_nlocal = (nx, ny, nz)` — Per-process subdomain (constant).
- `AC_ngrid = (decomp.x*nx, decomp.y*ny, decomp.z*nz)` — Total domain scales with `nprocs`.

## 3. Device Setup
- `acGridInit(info)` — Allocate device buffers, setup MPI communicators.
- `acGridRandomize()` — Initialize the mesh with random field data.

## 4. Dry Run
One integration step executed without timing to ensure all GPU kernels are compiled and launched.

## 5. Optional Verification
If `verify` is set:
1. Two host-side meshes (`model` and `candidate`) are created and randomly initialized.
2. `model` is loaded to the device and periodic boundary conditions are applied.
3. 10 integration steps are performed in parallel:
   - GPU: `integrate(dt)` via task graph execution.
   - CPU: `acHostIntegrateStep(model, dt)` on host mesh with periodic bounds.
4. Device mesh is stored back to `candidate` via `acGridStoreMesh`.
5. `acVerifyMesh("Integration", model, candidate)` compares host and GPU results.
6. If verification fails (`AC_SUCCESS` not returned), the benchmark aborts.

## 6. Timing Benchmark
- **Warmup**: 5 integration steps discarded.
- **Measurement**: 100 integration steps timed individually, results stored as milliseconds.
- Each step is wrapped with `acGridSynchronizeStream(STREAM_ALL)` for accurate GPU-side timing.

## 7. Statistics & Output
Rank 0 computes and prints:
- **50th percentile** (median) integration step time.
- **90th percentile** integration step time.

Results are appended to `scaling-benchmark.csv` with the format:
```
nprocs, min, 50th_percentile, 90th_percentile, max, use_distributed_io, nx, ny, nz, is_strong_scaling
```

## 8. Sanity Performance Checks
Individual kernel timings are measured:
- `acGridPeriodicBoundconds` — Periodic boundary condition application.
- `acGridIntegrate` — Full integration step.
- `acGridReduceScal` — Scalar field sum reduction.
- `acGridReduceVec` — Vector field sum reduction across fields 0, 1, 2.

If enabled, `acProfilerStart()`/`acProfilerStop()` wraps the integrate call for profiling output.

## 9. Cleanup
- `acGridQuit()` — Destroy device resources.
- `MPI_Finalize()` — Terminate MPI.

# The Integrate Function

The core benchmarked operation is encapsulated in the `integrate()` function:

```cpp
void integrate(const AcReal dt)
{
    acDeviceSetInput(acGridGetDevice(), AC_dt, dt);
    for (int substep = 0; substep < 3; ++substep) {
        acDeviceSetInput(acGridGetDevice(), AC_SUBSTEP, (AC_SUBSTEP_NUMBER)substep);
        AcTaskGraph* dsl_graph = acGetOptimizedDSLTaskGraph(AC_rhs_substep);
        const AcReal start = MPI_Wtime();
        acGridExecuteTaskGraph(dsl_graph, 1);
        const AcReal end = MPI_Wtime();
        fprintf(stderr, "Substep took %.14e\n", end - start);
    }
}
```

Key details:
- The integration consists of **3 substeps** per step (`AC_rhs_substep` kernel).
- `AC_dt` and the current `AC_SUBSTEP` are set per-substep via device inputs.
- Each substep's timing is logged to stderr with microsecond precision.
- A constant timestep `dt = FLT_EPSILON` is used (the actual timestep value is irrelevant since only throughput is measured).

# Key Dependencies
- `astaroth.h` — Core Astaroth API (mesh, device, task graph operations).
- `astaroth_utils.h` — Utility functions (info management, config loading, host-side mesh ops).
- `astaroth_cuda_wrappers.h` — CUDA-specific wrapper functions.
- `../../stdlib/reduction.h` — Reduction operation definitions (`RTYPE_SUM`, etc.).
- `errchk.h` — Error-checking macros.
- `timer_hires.h` — High-resolution timer utilities.
- `MPI` — Parallel process management and timing (`MPI_Wtime`).

# Usage Notes

- The benchmark requires MPI-enabled builds (`-DMPI_ENABLED=ON`).
- The benchmark requires integration DSL code to be compiled into the library (`AC_INTEGRATION_ENABLED`).
- Only rank 0 writes to stdout and the CSV output file.
- CUDA synchronization (`acGridSynchronizeStream`) around every timed call ensures accurate GPU-side measurements but adds overhead to each iteration.
- The CSV file uses append mode (`"a"`) so results accumulate across runs.
