# ⚠️ WARNING ⚠️
# **This is a LLM based analysis.**
# **THE BRANCH SHOULD NEVER BE MERGED TO master OR develop BRANCHES!**

# LLM Information
- **Model:** Qwen3.6-35B-A3B

# Overview
The `mpitest` sample is a comprehensive MPI integration test for Astaroth's GPU-accelerated grid infrastructure. It verifies the correctness of core Astaroth subsystems — mesh load/store, disk I/O, boundary conditions, task graph execution, integration, and reductions — by comparing GPU (device) results against CPU (host) reference implementations. The test is driven by six command-line parameters controlling grid dimensions, integration step count, and numerical tolerance. It uses `dt = FLT_EPSILON` for integration to isolate numerical discrepancies from physical integration drift. The sample also supports runtime compilation (`AC_RUNTIME_COMPILATION`), loading generated DSL code at runtime from shared libraries.

# Directory Structure & File Descriptions

| File | Description |
| :--- | :--- |
| `CMakeLists.txt` | Build configuration; compiles `main.cc` into the `mpitest` executable, linked against `astaroth_core` and `astaroth_utils`. Includes commented-out OpenMP support for potential multi-core parallelization. Enables position-independent code. |
| `main.cc` | MPI integration test (423 lines): initializes MPI and Astaroth grid with configurable mesh decomposition, verifies mesh load/store, disk write/read, periodic BCs, task-graph execution, GPU vs CPU integration, and all three reduction types (scalar, vector, Alfven). |
| `README.txt` | Build and run instructions: requires `cmake -DMPI_ENABLED=ON -DCMAKE_CXX_COMPILER=$(which mpicxx)` and execution via `mpirun` or `srun`. |

# Compile-Time Requirements

The code has a single preprocessor guard:

| Guard | Behavior if not defined |
| :--- | :--- |
| `AC_MPI_ENABLED` | Prints error message, returns `EXIT_FAILURE`. Astaroth must be built with `cmake -DMPI_ENABLED=ON`. |

# Compile-Time Options

| Option | Default | Description |
| :--- | :--- | :--- |
| `AC_RUNTIME_COMPILATION` | Off | If defined, enables runtime compilation of DSL kernels, loading them from dynamically compiled shared libraries. Also writes compilation log to `ac_compilation_log`. |

# Input Parameters / Command-Line Interface

| Parameter | Position | Default | Description |
| :--- | :--- | :--- | :--- |
| `nx` | argv[1] | 18 (`2*9`) | Grid dimension in x |
| `ny` | argv[2] | 22 (`2*11`) | Grid dimension in y |
| `nz` | argv[3] | 28 (`4*7`) | Grid dimension in z |
| `NUM_INTEGRATION_STEPS` | argv[4] | 100 | Number of integration steps to run for correctness verification |
| `max_ulp_error` | argv[5] | 5 | Maximum allowed error in ULPs (units in last place) for mesh comparison |

Usage: `mpirun -np <num_processes> ./mpitest [nx] [ny] [nz] [num_steps] [max_ulp_error]`

# Program Flow

## 1. MPI Init
`MPI_Init`, get rank (`pid`) and total processes (`nprocs`).

## 2. Mesh Configuration
a. `acInitInfo()` — create default `AcMeshInfo`.
b. `acLoadConfig(AC_DEFAULT_CONFIG, &info)` — load field definitions and defaults.
c. `info.comm->handle = MPI_COMM_WORLD` — inject the MPI communicator.
d. `acPushToConfig(info, AC_proc_mapping_strategy, AC_PROC_MAPPING_STRATEGY_LINEAR)` — linear process mapping.
e. `acPushToConfig(info, AC_decompose_strategy, AC_DECOMPOSE_STRATEGY_MORTON)` — Morton-curve decomposition.
f. `acPushToConfig(info, AC_MPI_comm_strategy, AC_MPI_COMM_STRATEGY_DUP_WORLD)` — duplicate `MPI_COMM_WORLD`.
g. Parse command-line grid dimensions (`nx`, `ny`, `nz`).
h. `acDecompose(nprocs, info)` — compute domain decomposition across ranks.
i. `acSetGridMeshDims(nx, ny, nz, &info)` — set global grid dimensions.
j. `acSetLocalMeshDims(nx/decomp.x, ny/decomp.y, nz/decomp.z, &info)` — set per-rank local dimensions.

## 3. Runtime Compilation (if `AC_RUNTIME_COMPILATION`)
a. Set runtime constants: spherical coords flag, `AC_runtime_real`, `AC_runtime_real3`, `AC_runtime_int`, `AC_runtime_int3`, arrays for real/int/bool.
b. `acCompile(build_str, info)` — compile DSL kernels from build string.
c. `acLoadLibrary(stdout, info)` — load compiled shared library.
d. `acLoadUtils(stdout, info)` — load utility functions from compiled library.
e. Write compilation log to `ac_compilation_log`.

## 4. Process Limit Check
`max_devices = 2 * 2 * 4 = 32`. If `nprocs > 32`, abort. This is hardcoded — the test assumes a maximum of 32 GPU processes.

## 5. Host Mesh Creation (rank 0 only)
a. Create `model` and `candidate` host meshes via `acHostGridMeshCreate`.
b. Randomize both via `acHostGridMeshRandomize`.
c. `srand(321654987)` — fixed seed for reproducibility.

## 6. GPU Memory Allocation & Grid Init
a. Allocate `gmem_arr` (100 `AcReal` elements) for GPU real array parameter: `info[AC_real_gmem_arr] = gmem_arr`.
b. `acGridInit(info)` — initialize GPU grid infrastructure.
c. Rank 0 saves local device config to `mpitest.conf` via `acStoreConfig`.

## 7. Load/Store Verification
a. `acGridLoadMesh(STREAM_DEFAULT, model)` — load model to device.
b. `acGridStoreMesh(STREAM_DEFAULT, &candidate)` — store device to candidate.
c. Rank 0: `acVerifyMesh("Load/Store", model, candidate)` — compare. Uses `WARNCHK_ALWAYS` (warning, not fatal).

## 8. Write/Read Disk I/O Verification
a. Rank 0: `acHostGridMeshRandomize(&candidate)` — scramble candidate data.
b. `acGridWriteMeshToDiskLaunch("snapshot", "0")` — launch async mesh write.
c. `acGridDiskAccessSync()` — wait for async I/O to complete.
d. Build field list: iterate `NUM_VTXBUF_HANDLES` fields, push each to `std::vector<Field> io_fields`.
e. Read all fields: `acGridAccessMeshOnDiskSynchronous` for each field from `"snapshot"`.
f. `acGridPeriodicBoundconds(STREAM_DEFAULT)` — apply periodic BCs on device.
g. `acGridStoreMesh(STREAM_DEFAULT, &candidate)` — store device to candidate.
h. Rank 0: `acHostMeshApplyPeriodicBounds(&model)` + `acVerifyMesh("Write/Read", model, candidate)`.

## 9. Periodic Boundary Conditions Verification (legacy API)
a. Rank 0: `acHostGridMeshRandomize(&model)` (appears twice — likely a copy-paste artifact).
b. `acGridHaloExchange()` — exchange halo data between ranks.
c. `acGridLoadMesh(STREAM_DEFAULT, model)` — load model to device.
d. `acGridPeriodicBoundconds(STREAM_DEFAULT)` — apply periodic BCs on device.
e. `acGridStoreMesh(STREAM_DEFAULT, &candidate)` — store device to candidate.
f. Rank 0: `acHostMeshApplyPeriodicBounds(&model)` + `acVerifyMesh("Periodic boundconds", model, candidate)`.

## 10. DSL Task Graph Periodic BCs Verification
a. Rank 0: `acHostGridMeshRandomize(&model)`.
b. `acGetOptimizedDSLTaskGraph(boundconds)` — get optimized task graph for boundary conditions.
c. `acGridLoadMesh(STREAM_DEFAULT, model)` — load model to device.
d. `acGridExecuteTaskGraphBase(periodic, 1, true)` — execute the task graph once with all operations.
e. `acGridStoreMesh(STREAM_DEFAULT, &candidate)` — store device to candidate.
f. Rank 0: `acHostMeshApplyPeriodicBounds(&model)` + `acVerifyMesh("DSL Periodic boundconds", model, candidate)`.

## 11. GPU Integration Verification (legacy `acGridIntegrate`)
a. `dt = FLT_EPSILON` — minimal time step.
b. Rank 0: `acHostGridMeshRandomize(&model)`.
c. `acGridLoadMesh(STREAM_DEFAULT, model)` — load model to device.
d. `acGridPeriodicBoundconds(STREAM_DEFAULT)` — apply periodic BCs.
e. Device integration loop: `NUM_INTEGRATION_STEPS` iterations of `acGridIntegrate(STREAM_DEFAULT, dt)`.
f. Periodic BCs after integration, `acGridStoreMesh(STREAM_DEFAULT, &candidate)`.
g. Rank 0: Host integration via `acHostIntegrateStep(model, dt)` for same number of steps, then `acVerifyMeshWithMaximumError("Integration", model, candidate, max_ulp_error)`.

## 12. Dryrun: Task Graph Substeps
a. `acDeviceSetInput(acGridGetDevice(), AC_SUBSTEP, substep)` for substep 0, 1, 2.
b. `acGetOptimizedDSLTaskGraph(AC_rhs_substep)` — get RHS substep task graph.
c. `acGridExecuteTaskGraph(dsl_graph, 1)` — execute once.
d. This runs 3 substeps but does NOT compare against CPU — it's a smoke test that the call doesn't crash.

## 13. GPU Integration Verification (DSL Task Graph)
a. Rank 0: `acHostGridMeshRandomize(&model)`.
b. `acGridLoadMesh(STREAM_DEFAULT, model)`, periodic BCs.
c. Device integration loop: `NUM_INTEGRATION_STEPS` iterations, each containing 3 substeps:
   - `acDeviceSetInput(AC_SUBSTEP, substep)` — set substep number.
   - `acGetOptimizedDSLTaskGraph(AC_rhs_substep)` — get task graph.
   - `acGridExecuteTaskGraph(dsl_graph, 1)` — execute.
   - Time each substep with `MPI_Wtime()`, print to stderr.
d. Periodic BCs, `acGridStoreMesh(STREAM_DEFAULT, &candidate)`.
e. Rank 0: Host integration via `acHostIntegrateStep` for same steps, then `acVerifyMeshWithMaximumError("DSL ComputeSteps", model, candidate, max_ulp_error)`.

## 14. Scalar Reductions Verification
a. Rank 0: `acHostGridMeshRandomize(&model)`, apply periodic BCs on host.
b. `acGridLoadMesh(STREAM_DEFAULT, model)`, periodic BCs on device.
c. Loop over 5 reductions: `RTYPE_MAX`, `RTYPE_MIN`, `RTYPE_SUM`, `RTYPE_RMS`, `RTYPE_RMS_EXP`.
d. For each:
   - `acGridReduceScal(STREAM_DEFAULT, reduction, (VertexBufferHandle)0, &candval)` — GPU reduction.
   - Rank 0: `acHostReduceScal(model, reduction, v0)` — CPU reference.
   - `acGetError(modelval, candval)` — compute error struct.
   - Set `error.maximum_magnitude` and `error.minimum_magnitude` from host reductions.
   - `acEvalErrorWithMaximumError(reduction.name, error, max_ulp_error)` — check tolerance.

## 15. Vector Reductions Verification
a. Loop over same 5 reductions.
b. Fields: `v0 = VTXBUF_UUX (handle 0)`, `v1 = VTXBUF_UUY (handle 1)`, `v2 = VTXBUF_UUZ (handle 2)`.
c. For each:
   - `acGridReduceVec(STREAM_DEFAULT, reduction, v0, v1, v2, &candval)` — GPU reduction.
   - Rank 0: `acHostReduceVec(model, reduction, v0, v1, v2)` — CPU reference.
   - Error comparison with `acEvalErrorWithMaximumError`.

## 16. Alfven Reductions Verification
a. Loop over 3 reductions: `RTYPE_ALFVEN_MAX`, `RTYPE_ALFVEN_MIN`, `RTYPE_ALFVEN_RMS`.
b. Fields: `v0`, `v1`, `v2`, `v3` (handles 0-3).
c. For each:
   - `acGridReduceVecScal(STREAM_DEFAULT, reduction, v0, v1, v2, v3, &candval)` — GPU reduction.
   - Rank 0: `acHostReduceVecScal(model, reduction, v0, v1, v2, v3)` — CPU reference.
   - Error comparison with `acEvalErrorWithMaximumError`.

## 17. Cleanup
a. Rank 0: `acHostMeshDestroy(&model)`, `acHostMeshDestroy(&candidate)`.
b. `acGridQuit()` — shutdown grid.
c. `ac_MPI_Finalize()` — use Astaroth's MPI finalize wrapper (not raw `MPI_Finalize`).
d. `finalized = true` — set flag to prevent `acAbort` from double-aborting.
e. Rank 0: print completion status to stderr.

# Astaroth Grid APIs Used

| Function | Description |
| :--- | :--- |
| `acGridInit(info)` | Initialize GPU grid infrastructure with mesh config. |
| `acGridQuit()` | Shutdown grid, free GPU memory. |
| `acGridLoadMesh(stream, mesh)` | Load mesh from host to device. |
| `acGridStoreMesh(stream, &mesh)` | Store mesh from device to host. |
| `acGridIntegrate(stream, dt)` | Perform one GPU integration step (legacy API). |
| `acGridPeriodicBoundconds(stream)` | Apply periodic boundary conditions on device. |
| `acGridSynchronizeStream(stream)` | Block until stream operations complete. |
| `acGridSwapBuffers()` | Swap all input/output buffer pairs. |
| `acGridHaloExchange()` | Exchange halo data between MPI ranks. |
| `acGridWriteMeshToDiskLaunch(dir, label)` | Launch asynchronous mesh write to disk. |
| `acGridDiskAccessSync()` | Wait for pending async disk I/O to complete. |
| `acGridAccessMeshOnDiskSynchronous(field, dir, label, ACCESS_READ/WRITE)` | Synchronous disk I/O for a single field. |
| `acGridReduceScal(stream, reduction, vtxbuf, result)` | Compute scalar reduction across all ranks for one field. |
| `acGridReduceVec(stream, reduction, a, b, c, result)` | Compute scalar reduction across all ranks for three fields. |
| `acGridReduceVecScal(stream, reduction, a, b, c, d, result)` | Compute scalar reduction across all ranks for three vector fields + one scalar field (Alfven). |
| `acGridExecuteTaskGraph(graph, n_iters)` | Execute a task graph for `n_iters` iterations. |
| `acGridExecuteTaskGraphBase(graph, n_iters, include_all)` | Execute task graph with option to include all operations. |
| `acDeviceSetInput(device, param, value)` | Set a runtime parameter for device kernel execution. |

# Astaroth Host Mesh APIs Used

| Function | Description |
| :--- | :--- |
| `acHostGridMeshCreate(info, &mesh)` | Create a host-side mesh matching the grid config. |
| `acHostGridMeshRandomize(&mesh)` | Randomize all field values in the host mesh. |
| `acHostMeshApplyPeriodicBounds(&mesh)` | Apply periodic boundary conditions on host. |
| `acHostMeshDestroy(&mesh)` | Free host mesh memory. |
| `acHostIntegrateStep(mesh, dt)` | CPU-side integration step (reference implementation). |

# Astaroth Host Reduction APIs (CPU Reference)

| Function | Description |
| :--- | :--- |
| `acHostReduceScal(mesh, reduction, a)` | CPU scalar reduction for one field. |
| `acHostReduceVec(mesh, reduction, a, b, c)` | CPU scalar reduction for three fields. |
| `acHostReduceVecScal(mesh, reduction, a, b, c, d)` | CPU scalar reduction for three vector fields + one scalar (Alfven). |

# Astaroth Utility / DSL APIs Used

| Function | Description |
| :--- | :--- |
| `acInitInfo()` | Create a default-initialized `AcMeshInfo`. |
| `acLoadConfig(config, &info)` | Load mesh config from `AC_DEFAULT_CONFIG`. |
| `acPushToConfig(info, param, value)` | Set a parameter in the mesh config. |
| `acSetGridMeshDims(nx, ny, nz, &info)` | Set global grid dimensions. |
| `acSetLocalMeshDims(nx, ny, nz, &info)` | Set per-rank local grid dimensions. |
| `acDecompose(nprocs, info)` | Compute domain decomposition (returns `int3` with x/y/z process counts). |
| `acGetOptimizedDSLTaskGraph(op)` | Get optimized task graph for an operation (`boundconds`, `AC_rhs_substep`). |
| `acCompile(build_str, info)` | Compile DSL kernels at runtime (if `AC_RUNTIME_COMPILATION`). |
| `acLoadLibrary(stream, info)` | Load compiled DSL shared library. |
| `acLoadUtils(stream, info)` | Load compiled utility functions from DSL library. |
| `acStoreConfig(info, filename)` | Save mesh config to a file. |
| `acVerifyMesh(label, model, candidate)` | Compare two meshes element-by-element (returns `AcResult`). |
| `acVerifyMeshWithMaximumError(label, model, candidate, max_ulp_error)` | Compare meshes with ULP tolerance. |
| `acEvalErrorWithMaximumError(name, error, max_ulp_error)` | Evaluate whether an error struct is within tolerance. |
| `acGetError(model, candidate)` | Compute error struct from model and candidate values. |
| `acDeviceGetLocalConfig(device)` | Get local device configuration parameters. |
| `acGetErrorString(err)` | Get string description of an error code. |
| `acGetErrorName(err)` | Get name of an error code. |

# Runtime Compilation (AC_RUNTIME_COMPILATION)

When enabled, the sample configures runtime compilation of Astaroth's DSL kernels:

| Runtime Constant | Value Set | Description |
| :--- | :--- | :--- |
| `AC_lspherical_coords` | `true` | Enable spherical coordinate system |
| `AC_runtime_int` | `0` | Integer runtime constant |
| `AC_runtime_real` | `0.12345` | Real runtime constant |
| `AC_runtime_real3` | `{0.12345, 0.12345, 0.12345}` | 3D real runtime constant |
| `AC_runtime_int3` | `{0, 1, 2}` | 3D integer runtime constant |
| `AC_runtime_real_arr` | `{-0, -1, -2, -3}` | Real array runtime constant |
| `AC_runtime_int_arr` | `{0, 1}` | Integer array runtime constant |
| `AC_runtime_bool_arr` | `{false, true}` | Boolean array runtime constant |

The build string passed to `acCompile` is hardcoded to a specific set of CMake options:
```
-DOPTIMIZE_FIELDS=ON -DOPTIMIZE_INPUT_PARAMS=ON -DELIMINATE_CONDITIONALS=ON -DOPTIMIZE_ARRAYS=ON -DBUILD_SAMPLES=OFF -DBUILD_STANDALONE=OFF -DBUILD_SHARED_LIBS=ON -DMPI_ENABLED=ON -DOPTIMIZE_MEM_ACCESSES=ON -DBUILD_ACM=OFF
```

# MPI APIs Used

| Function | Description |
| :--- | :--- |
| `MPI_Init(NULL, NULL)` | Initialize MPI. |
| `MPI_Comm_rank(MPI_COMM_WORLD, &pid)` | Get rank ID. |
| `MPI_Comm_size(MPI_COMM_WORLD, &nprocs)` | Get total number of processes. |
| `MPI_Abort(comm, code)` | Abort all ranks with exit code. |
| `MPI_Wtime()` | High-resolution wall-clock timer in seconds. |
| `ac_MPI_Finalize()` | Astaroth's MPI finalize wrapper (not raw `MPI_Finalize`). |

# Preprocessor Constants / Definitions

| Constant | Description |
| :--- | :--- |
| `AC_PROC_MAPPING_STRATEGY_LINEAR` | Linear process-to-device mapping strategy. |
| `AC_DECOMPOSE_STRATEGY_MORTON` | Morton-curve space-filling curve decomposition. |
| `AC_MPI_COMM_STRATEGY_DUP_WORLD` | Duplicate `MPI_COMM_WORLD` for Astaroth. |
| `STREAM_DEFAULT` | Defaults to `STREAM_0`. |
| `max_devices` | 32 (`2 * 2 * 4`) — hardcoded process limit. |
| `srand(321654987)` | Fixed random seed for reproducibility. |
| `dt = FLT_EPSILON` | Minimal time step for integration correctness testing. |
| `ARRAY_SIZE(x)` | `(sizeof(x) / sizeof(x[0]))` — macro for array length. |
| `finalized` | Global flag to prevent `acAbort` from double-aborting during cleanup. |

# Key Reduction Types Tested

| Type | Fields | Reduction Ops | Notes |
| :--- | :--- | :--- | :--- |
| Scalar | 1 (`v0 = handle 0`) | MAX, MIN, SUM, RMS, RMS_EXP | Operates on `VTXBUF_UUX` (x-velocity). |
| Vector | 3 (`v0`, `v1`, `v2` = handles 0-2) | MAX, MIN, SUM, RMS, RMS_EXP | Operates on x/y/z velocity components. |
| Alfven | 4 (`v0`, `v1`, `v2`, `v3` = handles 0-3) | ALFVEN_MAX, ALFVEN_MIN, ALFVEN_RMS | Operates on 3 vector fields + 1 scalar field (Alfven speed). |

# Verification Chain

| Test | GPU Operation | CPU Reference | Compare Function | Tolerance |
| :--- | :--- | :--- | :--- | :--- |
| Load/Store | `acGridLoadMesh` + `acGridStoreMesh` | — | `acVerifyMesh` | Exact |
| Write/Read | `acGridWriteMeshToDiskLaunch` + `acGridAccessMeshOnDiskSynchronous` | `acHostMeshApplyPeriodicBounds` | `acVerifyMesh` | Exact |
| Periodic BCs | `acGridHaloExchange` + `acGridPeriodicBoundconds` | `acHostMeshApplyPeriodicBounds` | `acVerifyMesh` | Exact |
| DSL Periodic BCs | `acGetOptimizedDSLTaskGraph(boundconds)` + `acGridExecuteTaskGraphBase` | `acHostMeshApplyPeriodicBounds` | `acVerifyMesh` | Exact |
| Integration | `acGridIntegrate` (N steps) | `acHostIntegrateStep` (N steps) | `acVerifyMeshWithMaximumError` | `max_ulp_error` ULPs |
| DSL ComputeSteps | `acGetOptimizedDSLTaskGraph(AC_rhs_substep)` (N × 3 substeps) | `acHostIntegrateStep` (N steps) | `acVerifyMeshWithMaximumError` | `max_ulp_error` ULPs |
| Scalar reductions | `acGridReduceScal` (5 ops) | `acHostReduceScal` (5 ops) | `acEvalErrorWithMaximumError` | `max_ulp_error` ULPs |
| Vector reductions | `acGridReduceVec` (5 ops) | `acHostReduceVec` (5 ops) | `acEvalErrorWithMaximumError` | `max_ulp_error` ULPs |
| Alfven reductions | `acGridReduceVecScal` (3 ops) | `acHostReduceVecScal` (3 ops) | `acEvalErrorWithMaximumError` | `max_ulp_error` ULPs |

All checks use `WARNCHK_ALWAYS` (warning, not fatal). Even if a check fails, the test continues to the next check and reports all failures at the end.

# Notable Observations

1. **Most comprehensive MPI test**: At 423 lines, this is the largest MPI sample. It exercises the full Astaroth stack: mesh I/O, boundary conditions, task graphs, integration kernels, and all reduction types. No other MPI sample covers this breadth.

2. **Process limit hardcoded to 32**: `max_devices = 2 * 2 * 4 = 32`. This reflects a target configuration of 2 nodes with 2 GPUs each (or some other 32-GPU arrangement). The test will not run on larger configurations without modifying the source.

3. **Dual integration verification paths**: The sample tests integration twice — once with the legacy `acGridIntegrate` API and once with the newer DSL task graph API (`acGetOptimizedDSLTaskGraph(AC_rhs_substep)` with 3 substeps per step). This suggests the DSL path is intended to supersede the legacy path.

4. **Substep timing instrumentation**: During the DSL integration loop, each of the 3 substeps is individually timed using `MPI_Wtime()`, with results printed to stderr. This is the only sample that provides per-substep profiling.

5. **Dryrun substep smoke test**: Before the full DSL integration, there's a "dryrun" section (lines 217-223) that runs 3 substeps without comparing against CPU. This ensures the task graph execution doesn't crash, before the full correctness test.

6. **`ac_MPI_Finalize()` over raw `MPI_Finalize()`**: The sample uses Astaroth's wrapper `ac_MPI_Finalize()` (which manages Astaroth's internal MPI communicator) instead of raw `MPI_Finalize()`. The `finalized` flag is set afterward to prevent `acAbort` (registered via `atexit`) from double-aborting.

7. **`atexit(acAbort)` safety net**: `acAbort` calls `MPI_Abort(acGridMPIComm(), EXIT_FAILURE)` if `finalized` is false. This provides automatic MPI abort on any unexpected early return. The `finalized` flag is set to `true` only after `ac_MPI_Finalize()` succeeds.

8. **`WARNCHK_ALWAYS` vs `ERRCHK_ALWAYS`**: All verification checks use `WARNCHK_ALWAYS`, which logs a warning but does not abort. This allows the test to run through all 14+ verification stages and report all failures at once, rather than failing fast on the first error.

9. **Rank-0 host mesh pattern**: Host meshes (`model`, `candidate`) are only created on rank 0, but the test compares per-rank GPU data against rank-0 CPU data. This works because `acHostIntegrateStep` and the host reduction functions operate only on the data available on rank 0 (rank 0's local grid chunk). Other ranks' grid data is not verified on the CPU side.

10. **Double `acHostGridMeshRandomize(&model)`**: Lines 168-173 call `acHostGridMeshRandomize` twice in a row with no intervening GPU operation. This is likely a copy-paste artifact and has no effect — the second call simply overwrites the random data from the first.

11. **Field handles instead of `VertexBufferHandle` enums**: Reduction tests use `(VertexBufferHandle)0`, `(VertexBufferHandle)1`, `(VertexBufferHandle)2`, `(VertexBufferHandle)3` instead of named constants like `VTXBUF_UUX`. This makes the code more fragile — if the field ordering changes, the tests would silently operate on the wrong fields.

12. **`acHostReduceVec` vs `acHostReduceVecScal` for Alfven error magnitude**: The error magnitude checks use `acHostReduceVec` for max magnitude but `acHostReduceVecScal` with a typo (`v1` instead of `v2` for the third parameter) in line 356. This appears to be a bug: `acHostReduceVec(model, RTYPE_MIN, v0, v1, v1)` passes `v1` twice instead of `v0, v1, v2`. The same pattern appears for Alfven: `acHostReduceVecScal(model, RTYPE_ALFVEN_MIN, v0, v1, v1, v3)` — `v1` is passed twice instead of `v1, v2`.

13. **`max_ulp_error` default of 5**: The default tolerance is 5 ULPs, which is relatively tight for multi-step integration across MPI-distributed grids. This reflects confidence in the numerical reproducibility of Astaroth's GPU vs CPU implementations.

14. **No error exit on WARNCHK**: Even when `WARNCHK_ALWAYS` catches an error, `retval` is set but the function returns `EXIT_SUCCESS` regardless. The exit status is always 0 unless the program aborts early (e.g., process limit exceeded). This makes automated testing unreliable — failures are logged to stdout/stderr but the exit code doesn't reflect them.

15. **`gmem_arr` for runtime array parameter**: The sample allocates a 100-element `AcReal` array and stores a pointer in `info[AC_real_gmem_arr]`. This is used to test the `AC_runtime_real_arr` runtime compilation constant. The array is not freed — it leaks for the lifetime of the process.

16. **`AC_DECOMPOSE_STRATEGY_MORTON` vs `AC_PROC_MAPPING_STRATEGY_LINEAR`**: The decomposition uses a Morton curve (space-filling curve for good locality) while the process mapping is linear. These are orthogonal concepts — decomposition determines how the grid is partitioned, mapping determines which rank gets which partition.

17. **Snapshot directory reuse**: The I/O write/read test uses `"snapshot"` as both directory and `"0"` as label. If multiple test runs are executed in the same directory, old snapshot data may interfere unless cleaned up.

18. **Substep set per iteration**: Inside the DSL integration loop (lines 271-279), `acDeviceSetInput(AC_SUBSTEP, substep)` is called at the start of each substep iteration. This sets a kernel parameter that controls which substep's stencil is executed. The task graph for `AC_rhs_substep` likely contains all 3 substep kernels, and the substep parameter selects which one to run.

19. **No error checking on MPI calls**: Like other MPI samples, `MPI_Init`, `MPI_Comm_rank`, `MPI_Comm_size`, and `MPI_Abort` have no error checking. An MPI failure would produce undefined behavior.

20. **Compilation log output**: When `AC_RUNTIME_COMPILATION` is enabled, the DSL compilation string is hardcoded to a specific set of CMake flags. This means the runtime-compiled code is built with a fixed optimization profile regardless of the parent project's build configuration.
