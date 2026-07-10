# ⚠️ WARNING ⚠️
# **This is a LLM based analysis.**
# **THE BRANCH SHOULD NEVER BE MERGED TO master OR develop BRANCHES!**

# LLM Information
- **Model:** Qwen3.6-35B-A3B

# Overview
The `benchmark-node` sample is a single-node, single-process GPU benchmarking tool that benchmarks Astaroth's **Node API** (`acNode*`), which provides a direct node-level interface to GPU kernels without MPI communicators. Unlike `samples/benchmark/` (which uses the multi-process `acGrid*` API) or `samples/benchmark-device/` (which uses the generic `acDevice*` API), this benchmark exercises the node abstraction layer — a higher-level wrapper that encapsulates device management, stream operations, and kernel launches within a single-node context.

# Directory Structure & File Descriptions

| File | Description |
| :--- | :--- |
| `CMakeLists.txt` | Build configuration; compiles `main.c` (C, not C++) into the `benchmark-node` executable with position-independent code, linked against `astaroth_core` and `astaroth_utils`. |
| `main.c` | Node-level integration benchmark: uses the `Node` abstraction (`acNodeCreate`, `acNodeIntegrateSubstep`, `acNodePeriodicBoundconds`, etc.), benchmarks full 3-substep integration, and outputs percentile statistics. |

# Notable: C vs C++

This is the only benchmark sample written in **C** (not C++). It includes `<stdlib.h>`, `<math.h>`, and the C-style `acProfilerStop()` and `ERRCHK_AC` macro rather than C++ stream I/O or `std::vector`.

# Command-Line Interface

| Argument | Required | Description |
| :--- | :--- | :--- |
| `<nx> <ny> <nz>` | Yes | Local mesh dimensions (3 positional integers). |

Usage: `./benchmark-node <nx> <ny> <nz>`

The executable requires exactly 4 arguments (program name + 3 dimensions); any other count triggers a usage error.

# Compile-Time Constants

| Macro | Description |
| :--- | :--- |
| `NSAMPLES` | Fixed to `100` — number of benchmark iterations. |
| `IMPLEMENTATION` | Build implementation identifier (written to CSV). |
| `MAX_THREADS_PER_BLOCK` | Maximum threads per block (written to CSV). |

# Internal Utility Functions

| Function | Description |
| :--- | :--- |
| `validate(count, arr)` | Checks that an array is non-decreasing (ascending order). Used after sorting to verify correctness. |
| `sort(count, arr)` | Selection sort implementation (O(n²)). Explicitly noted in a TODO comment to be consolidated with the common sort used in `benchmark`, `benchmark-device`, and `mpi-io`. |

# Program Flow

## 1. Profiler Setup & Field Handle Verification
1. `acProfilerStop()` — Disable the profiler.
2. `acGetNumFields()` — Print the total number of fields.
3. Iterate over all `NUM_VTXBUF_HANDLES` field names:
   - `acGetFieldHandle(field_names[i], &field)` — Resolve each named field to its numeric handle.
   - Print the field name and resolved handle.
4. `acGetFieldHandle("nonexistent", &field)` — Verify that resolving a nonexistent field returns `AC_FAILURE`.

## 2. Configuration & Mesh Info
1. `acLoadConfig(AC_DEFAULT_CONFIG, &info)` — Load default mesh configuration.
2. Parse CLI dimensions into `info.int3_params[AC_nlocal]`.
3. `acHostUpdateParams(&info)` — Compute derived parameters from the config.

## 3. Host Mesh Setup
1. `acHostMeshCreate(info, &model)` / `acHostMeshCreate(info, &candidate)` — Create host meshes.
2. `acHostMeshRandomize(&model)` / `acHostMeshRandomize(&candidate)` — Randomize both.
3. `acHostMeshApplyPeriodicBounds(&model)` — Apply periodic boundary conditions on the host.

## 4. Node Creation & Initialization
1. `acNodeCreate(0, info, &node)` — Create the node object (device ID 0 hardcoded).
2. `acNodePrintInfo(node)` — Print node/device information to stdout.

## 5. Load/Store Verification
1. `acNodeLoadMesh(node, STREAM_DEFAULT, model)` — Upload host mesh to node.
2. `acNodeStoreMesh(node, STREAM_DEFAULT, &candidate)` — Download to candidate mesh.
3. `acVerifyMesh("Load/Store", model, candidate)` — Verify exact match.

## 6. Boundary Condition Verification
1. Compute stencil boundary volumes:
   - `n_min = (STENCIL_ORDER/2, STENCIL_ORDER/2, STENCIL_ORDER/2)`
   - `n_max = (n_min.x + AC_nlocal.x, n_min.y + AC_nlocal.y, n_min.z + AC_nlocal.z)`
2. `acNodePeriodicBoundconds(node, STREAM_DEFAULT)` — Apply periodic boundaries on the device.
3. `acNodeStoreMesh(node, STREAM_DEFAULT, &candidate)` — Store result.
4. `acNodeSynchronizeStream(node, STREAM_DEFAULT)` — Wait for completion.
5. Apply host periodic bounds and compare via `acVerifyMesh("Boundconds", ...)`.

## 7. Integration Dry Run
1. `dt = FLT_EPSILON` — Constant timestep (value irrelevant for throughput measurement).
2. `acNodeIntegrateSubstep(node, STREAM_DEFAULT, 2, n_min, n_max, dt)` — Optimize for the more expensive substep (substep 2).
3. Run all 3 substeps in sequence:
   - `acNodeIntegrateSubstep` for each substep (0, 1, 2).
   - `acNodeSwapBuffers(node)` after each.
   - `acNodePeriodicBoundconds(node, STREAM_DEFAULT)` after each.
4. `acNodeLoadMesh(node, STREAM_DEFAULT, model)` — Reset mesh (workaround for buffer NaN state, noted in a TODO).
5. `acNodeSwapBuffers(node)` and `acNodeLoadMesh` again (redundant, part of the reset sequence).
6. `acNodePeriodicBoundconds(node, STREAM_DEFAULT)` — Final boundary application.

## 8. Verification Run (conditionally, `verify == false` — currently disabled)
If the `verify` flag were enabled:
1. `acNodeStoreMesh(node, STREAM_DEFAULT, &candidate)` — Store GPU result.
2. Run `nsteps` integration steps on host: `acHostIntegrateStep(model, dt)` + `acHostMeshApplyPeriodicBounds(&model)`.
3. `acNodeSynchronizeStream(node, STREAM_DEFAULT)` — Wait.
4. `acVerifyMesh("Integration", model, candidate)` — Compare.

The `verify` variable is hardcoded to `false`, so this path is never executed.

## 9. Warmup
10% of `NSAMPLES` (10 iterations) of 3-substep integration runs discarded as warmup:
- Each iteration: 3 substeps with `acNodeIntegrateSubstep`, `acNodeSwapBuffers`, and `acNodePeriodicBoundconds`.
- `acNodeSynchronizeStream(node, STREAM_DEFAULT)` after warmup.

## 10. Benchmark
For all 100 iterations:
1. `acNodeSynchronizeStream(node, STREAM_ALL)` — Synchronize before timing.
2. `timer_reset(&t)` — Start timer.
3. Run 3 substeps (full integration): each with `acNodeIntegrateSubstep`, `acNodeSwapBuffers`, `acNodePeriodicBoundconds`.
4. `acNodeSynchronizeStream(node, STREAM_ALL)` — Synchronize after timing.
5. `timer_diff_nsec(t) / 1e6` — Record elapsed time in milliseconds.

## 11. Statistics & Output
1. `sort(NSAMPLES, results)` — Sort timing results.
2. `validate(NSAMPLES, results)` — Verify sorted order.
3. Compute statistics:
   - **min**: `results[0]`
   - **median**: middle value (or average of two middle values if even count).
   - **90th percentile**: `results[ceil(0.9 * NSAMPLES)]`
   - **max**: `results[NSAMPLES - 1]`
4. Print to stdout: min, median, 90th percentile, max.
5. Append to `node-benchmark.csv`:
   ```
   implementation, maxthreadsperblock, percentile90th, nx, ny, nz, num_devices
   ```
   Where `num_devices = acGetNumDevicesPerNode()`.

## 12. Profiling
A single substep (substep 2) is profiled:
- `acProfilerStart()`
- `acNodeIntegrateSubstep(node, STREAM_DEFAULT, 2, n_min, n_max, dt)`
- `acProfilerStop()`

## 13. Cleanup
1. `acNodeDestroy(node)` — Destroy the node object.
2. `acHostMeshDestroy(&model)` / `acHostMeshDestroy(&candidate)` — Destroy host meshes.

# Node API Functions Used

| Function | Description |
| :--- | :--- |
| `acNodeCreate(device_id, info, &node)` | Create a node object for the given device. |
| `acNodePrintInfo(node)` | Print node/device information. |
| `acNodeLoadMesh(node, stream, mesh)` | Load a mesh from host to node/device. |
| `acNodeStoreMesh(node, stream, &mesh)` | Store a mesh from node/device to host. |
| `acNodeSynchronizeStream(node, stream)` | Synchronize the node's stream. |
| `acNodePeriodicBoundconds(node, stream)` | Apply periodic boundary conditions on the device. |
| `acNodeIntegrateSubstep(node, stream, substep, n_min, n_max, dt)` | Execute a single integration substep with explicit volume bounds. |
| `acNodeSwapBuffers(node)` | Swap in/out field buffers. |
| `acNodeDestroy(node)` | Destroy the node object and free resources. |
| `acGetNumDevicesPerNode()` | Query the number of devices per node. |

# Volume Bounds: `n_min` and `n_max`

Unlike `benchmark-device` which passes `(dims.n0, dims.n1)` to kernel launches, `benchmark-node` passes explicit volume bounds `(n_min, n_max)` to `acNodeIntegrateSubstep`. These define the inner computation region that excludes the stencil halo:

```
n_min = (STENCIL_ORDER/2, STENCIL_ORDER/2, STENCIL_ORDER/2)
n_max = (n_min.x + AC_nlocal.x, n_min.y + AC_nlocal.y, n_min.z + AC_nlocal.z)
```

This ensures kernels only operate on interior points, with halo regions handled separately by boundary condition operations.

# CSV Output File

The output file is `node-benchmark.csv` (open in append mode `"a"`) with the format:

| Column | Description |
| :--- | :--- |
| `implementation` | Compile-time `IMPLEMENTATION` constant. |
| `maxthreadsperblock` | Compile-time `MAX_THREADS_PER_BLOCK` constant. |
| `percentile90th` | 90th percentile integration time (ms) across 100 samples. |
| `nx, ny, nz` | Input grid dimensions from CLI. |
| `num_devices` | Devices per node (from `acGetNumDevicesPerNode()`). |

# Key Dependencies
- `astaroth.h` — Core API including `Node` abstraction, `acNode*` functions, `acVerifyMesh`.
- `astaroth_utils.h` — Utility functions.
- `astaroth_cuda_wrappers.h` — CUDA wrappers.
- `errchk.h` — Error-checking macros (`ERRCHK_AC`, `ERRCHK_ALWAYS`).
- `timer_hires.h` — High-resolution timer.
- `<math.h>`, `<stdlib.h>` — Standard C library.

# Comparison with Other Benchmarks

| Aspect | `benchmark` | `benchmark-device` | `benchmark-node` |
| :--- | :--- | :--- | :--- |
| **API level** | `acGrid*` (multi-process) | `acDevice*` (single device) | `acNode*` (single-node wrapper) |
| **Language** | C++ | C++ | **C** |
| **MPI required** | Yes (`AC_MPI_ENABLED`) | No (optional) | No |
| **What is timed** | Full integration step via task graph | Single kernel launch (`singlepass_solve`) | Full 3-substep integration |
| **Verification** | Optional CPU-vs-GPU | Optional CPU-vs-GPU (3 stages) | Optional (hardcoded disabled) |
| **Volume bounds** | Implicit (via task graph) | Implicit (via kernel launch dims) | **Explicit** `n_min`, `n_max` |
| **Sort method** | `std::sort` | N/A (no sorting) | Selection sort (manual) |
| **CSV columns** | 10 fields including scaling type | 13 fields including TPB | 7 fields (simpler) |
| **Profiling** | Yes (integrate) | Yes (singlepass_solve) | Yes (substep 2) |
