# ⚠️ WARNING ⚠️
# **This is a LLM based analysis.**
# **THE BRANCH SHOULD NEVER BE MERGED TO master OR develop BRANCHES!**

# LLM Information
- **Model:** Qwen3.6-35B-A3B

# Overview
The `taskgraph_trace` directory provides an MPI-based performance tracing utility that records per-kernel execution timing data from Astaroth's task graph execution. It sets up two randomized host meshes (model and candidate), loads one to GPU, performs 100 warm-up iterations via `acGridIntegrate()` to trigger JIT compilation and auto-optimizations, then enables tracing and executes one measured iteration. The trace output is written to per-process files (`trace_pid_<rank>.txt`) and captures fine-grained timing information for each task and kernel in the default task graph, making it useful for performance profiling and bottleneck identification.

# Directory Structure & File Descriptions

| File | Description |
| :--- | :--- |
| `CMakeLists.txt` | Build config: creates `taskgraph_trace` executable from `main.cc`, links `astaroth_core` and `astaroth_utils`. |
| `main.cc` | MPI-based task graph performance tracer. Initializes MPI, creates two randomized host meshes, loads one to GPU, warms up kernels with 100 iterations, enables tracing, executes one measured iteration, and shuts down. |

# Compile-Time Requirements

| Setting | Value | Description |
| :--- | :--- | :--- |
| MPI | `REQUIRED` | Program requires MPI; falls back to error message if built without MPI support. |
| `AC_MPI_ENABLED` | Config-dependent | If disabled, program prints error and exits. |

Compile options: Inherited from `astaroth_core` (typically `-Wall -Wextra -Werror -Wdouble-promotion -Wfloat-conversion -Wshadow`).

# Compile-Time Options

| Macro | Default | Description |
| :--- | :--- | :--- |
| `AC_MPI_ENABLED` | Config-dependent | If disabled, program prints error and exits. |

# Input Parameters / Command-Line Interface

No command-line arguments. The program uses `AC_DEFAULT_CONFIG` via `acLoadConfig()` for mesh configuration.

Usage: `mpirun -np <num_processes> ./taskgraph_trace`

# Program Flow

## 1. Setup
- `MPI_Init(NULL, NULL)`
- `MPI_Comm_size(MPI_COMM_WORLD, &nprocs)` — get process count.
- `MPI_Comm_rank(MPI_COMM_WORLD, &pid)` — get process rank.
- `srand(321654987)` — seed the pseudo-random number generator with a fixed value for reproducibility.

## 2. Configuration & Host Mesh Setup (rank 0 only)
- `acLoadConfig(AC_DEFAULT_CONFIG, &info)` — load mesh configuration.
- `if (pid == 0)` — only rank 0 creates meshes.
- `acHostMeshCreate(info, &model)` — allocate first CPU-side `AcMesh`.
- `acHostMeshCreate(info, &candidate)` — allocate second CPU-side `AcMesh`.
- `acHostMeshRandomize(&model)` — randomize first mesh.
- `acHostMeshRandomize(&candidate)` — randomize second mesh.

## 3. GPU Initialization & Mesh Load
- `acGridInit(info)` — initialize GPU subsystem, create default task graph.
- `acGridLoadMesh(STREAM_DEFAULT, model)` — transfer the **model** mesh to GPU. The **candidate** mesh remains on CPU (never loaded).

## 4. Trace Setup
- `acGridGetDefaultTaskGraph()` — retrieve the default task graph.
- Construct filename: `"dependencies_pid_" + std::to_string(pid)` — per-process dependency file.
- `acGraphWriteDependencies(dependencies_file_path.c_str(), default_tasks)` — write dependency structure (same as `taskgraph_print`).
- **Warm-up loop:** 100 iterations of `acGridIntegrate(STREAM_DEFAULT, FLT_EPSILON)` to trigger JIT compilation and auto-optimizations.
- Construct trace filename: `"trace_pid_" + std::to_string(pid)` — per-process trace output file.
- `acGraphEnableTrace(trace_file_path.c_str(), default_tasks)` — enable kernel timing trace for the task graph.

## 5. Measured Execution
- `acGridIntegrate(STREAM_DEFAULT, FLT_EPSILON)` — execute one iteration with tracing enabled. Kernel execution times are recorded to the trace file.

## 6. Cleanup
- `acGridQuit()` — shutdown GPU subsystem.
- `MPI_Finalize()` — shutdown MPI.
- `return EXIT_SUCCESS` — clean exit.

# API Functions Used

## GPU/Grid API

| Function | Description |
| :--- | :--- |
| `acGridInit(info)` | Initialize GPU subsystem, create global grid and default task graph. |
| `acGridQuit()` | Shutdown GPU subsystem. |
| `acGridLoadMesh(stream, mesh)` | Transfer mesh from host to GPU. |
| `acGridGetDefaultTaskGraph()` | Retrieve the default task graph created during grid initialization. |
| `acGridIntegrate(stream, dt)` | Execute the default task graph for one timestep with the given dt parameter. |

## Debug/Introspection API

| Function | Description |
| :--- | :--- |
| `acGraphWriteDependencies(path, graph)` | Write task graph dependency information to a text file. |
| `acGraphEnableTrace(path, graph)` | Enable kernel execution timing trace, writing results to a file. |

## Host API

| Function | Description |
| :--- | :--- |
| `acLoadConfig(path, &info)` | Load configuration from file. |
| `acHostMeshCreate(info, &mesh)` | Allocate CPU-side `AcMesh`. |
| `acHostMeshRandomize(&mesh)` | Fill with pseudo-random values. |

# Warm-up Strategy

## 100 Iteration Warm-up (lines 42-44)
The program executes 100 warm-up iterations before tracing:
```cpp
for (int i = 0; i < 100; i++) {
    acGridIntegrate(STREAM_DEFAULT, FLT_EPSILON);
}
```
This serves several purposes:
1. **JIT compilation:** GPU kernels are compiled on first use; warm-up ensures compiled versions exist.
2. **Auto-optimization:** The library may tune kernel parameters (workgroup sizes, tile dimensions) during initial runs.
3. **Memory allocation:** GPU memory is allocated and pinned during the first few iterations.
4. **Warm-up overhead excluded:** The 100 warm-up iterations are not traced, so JIT compilation and memory allocation overhead are excluded from the trace data.

# Trace Output

## `acGraphEnableTrace`
This function enables fine-grained timing tracing for the task graph. It writes per-kernel execution times to a file named `trace_pid_<rank>.txt` for each MPI process. The trace likely includes:
- Kernel start/end timestamps.
- Per-task execution duration.
- Possibly GPU kernel launch latencies.
- Possibly halo exchange and boundary condition timings.
- Inter-task synchronization delays.

The per-process file naming allows correlation with the corresponding `dependencies_pid_<rank>.txt` files from step 4.

# Notable Observations

1. **Performance profiling focus:** Unlike the other taskgraph_* samples (which are introspection/debugging utilities), this sample is specifically designed for performance measurement — it traces kernel execution times to identify bottlenecks.

2. **Two meshes but one loaded:** The program creates two randomized meshes (`model` and `candidate`) but only loads `model` to GPU. The `candidate` mesh sits on the CPU unused. This may be scaffolding for a future feature (e.g., model comparison or parameter sweep).

3. **Fixed random seed:** `srand(321654987)` on line 19 seeds the RNG with a hardcoded value, ensuring reproducible randomized initial conditions. This is good practice for performance testing where results need to be comparable across runs.

4. **Unused `nprocs` variable:** The `nprocs` variable is populated by `MPI_Comm_size` but never used — same pattern as `taskgraph_test`.

5. **100 warm-up iterations:** This is a significant number of warm-up iterations. While JIT compilation typically completes on the first kernel launch, 100 iterations ensures all auto-tuning and memory optimizations are fully settled before tracing begins. This is conservative but thorough.

6. **`acGridIntegrate` vs `acGridExecuteTaskGraph`:** This sample uses `acGridIntegrate(STREAM_DEFAULT, FLT_EPSILON)` which is a higher-level wrapper that executes the default task graph for one timestep. Other samples use `acGridExecuteTaskGraph(graph, n)` for explicit task graph control. `acGridIntegrate` abstracts away the task graph reference.

7. **DT set to FLT_EPSILON:** Consistent with testing/verification samples, `acGridIntegrate(STREAM_DEFAULT, FLT_EPSILON)` sets the timestep to machine epsilon. The trace measures kernel execution times, not physical simulation behavior.

8. **Two trace files per process:** Each rank produces two files: `dependencies_pid_<rank>.txt` (task graph structure) and `trace_pid_<rank>.txt` (execution timing). This allows correlating structure with performance.

9. **No host mesh cleanup:** Neither `model` nor `candidate` is destroyed with `acHostMeshDestroy()`. Both are leaked on process termination.

10. **Trace is one-shot:** `acGraphEnableTrace()` is called once, then one measured iteration runs, then the program exits. There is no mechanism to enable/disable tracing mid-execution or to compare trace data across multiple runs.

11. **`#endif` typo:** Line 65 has `#endif // AC_MPI_ENABLES` (missing trailing `D`), consistent with `taskgraph_print` and `taskgraph_test`.

12. **Copy-paste error message:** Line 61 says "cannot run mpitest" — the same copy-paste artifact from other samples.

13. **CMakeLists.txt comment describes three-step integration:** Line 1 of CMakeLists.txt says "TaskGraph trace of one iteration of a three step integration," suggesting the default task graph contains 3 compute steps (likely halo exchange → boundary condition → solve, matching `taskgraph_example`).

14. **No stdout output during execution:** Unlike `taskgraph_test` which prints a progress message, this sample produces no stdout output. All results go to files, making it suitable for batch execution but not interactive feedback.

15. **`acGraphEnableTrace` replaces `acGraphPrintDependencies`:** This sample does both — it writes dependencies to a file first, then enables tracing. It is more comprehensive than either `taskgraph_print` or `taskgraph_test` individually.

16. **`STREAM_DEFAULT` used consistently:** Mesh load and integrate both use `STREAM_DEFAULT`, consistent with the single-stream pattern in other samples.

17. **Trace granularity unknown:** The exact content of the trace file is not documented. It could include GPU kernel start/end times, CPU-side kernel launch times, or both. The granularity determines how useful the trace is for identifying specific bottlenecks.

18. **MPI rank used for file naming:** Per-process files use `pid` (MPI rank) in the filename, enabling correlation between MPI processes and their local task graph performance. This is essential for multi-process performance analysis.

19. **Potential for automated performance regression testing:** The combination of fixed seed, fixed warm-up count, fixed trace file output, and per-process files makes this sample potentially useful for automated performance testing — trace output could be compared across commits or hardware changes.

20. **`errchk.h` included but unused:** Same pattern as other samples — the header is included but no error checking is performed. Consistent with the minimal error handling throughout the taskgraph_* sample family.
