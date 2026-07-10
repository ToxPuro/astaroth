# ⚠️ WARNING ⚠️
# **This is a LLM based analysis.**
# **THE BRANCH SHOULD NEVER BE MERGED TO master OR develop BRANCHES!**

# LLM Information
- **Model:** Qwen3.6-35B-A3B

# Overview
The `boundcond_test` is Astaroth's comprehensive boundary condition verification test suite. It validates the correctness of all ghost-zone boundary condition implementations (periodic, symmetric, antisymmetric, relative antisymmetry, and prescribed derivative) by comparing device-computed ghost zone values against expected host-side calculations. The test decomposes the 3D mesh boundary into **6 face regions**, **12 edge regions**, and **8 corner regions**, testing each independently for every field.

# Directory Structure & File Descriptions

| File | Description |
| :--- | :--- |
| `CMakeLists.txt` | Build configuration; compiles `main.cc` into the `boundcond_test` executable, linked against `astaroth_core` and `astaroth_utils`. |
| `main.cc` | The test suite: MPI-required C++ program that builds task graphs for each boundary condition type, executes them, and verifies all ghost zone regions (faces, edges, corners) against analytical expectations. |

# Precondition

The executable requires MPI support:

| Condition | Macro | Error Message |
| :--- | :--- | :--- |
| MPI support | `AC_MPI_ENABLED` | "The library was built without MPI support, cannot run mpitest." |

If not enabled, `main()` returns `EXIT_FAILURE`.

# Data Structures

## Test Result Types

| Structure | Description |
| :--- | :--- |
| `CellError` | Single cell mismatch: `field`, `dom` (domain coords), `ghost` (ghost coords), `expected` value, `produced` value. |
| `ErrorRatio` | `total` cells checked, `errors` mismatches found. |
| `TestResultRegion` | Results for one region: `passed`, `normal` direction, list of `failed_cells`, per-field `field_errors`, overall `error_ratio`. |

## Test Case Types

| Structure | Description |
| :--- | :--- |
| `SimpleTestCase` | Tests ghost zones against an analytical kernel: `name`, `task_graph`, `test_func` (a `boundcond_kernel_func`). |
| `MeshCompareTestCase` | Tests ghost zones against a reference mesh: `name`, `task_graph`, `mesh_prior`, `mesh_expected_result`. |

## Aggregate Result

| Structure | Description |
| :--- | :--- |
| `TestResult` | Results for one test case: `name`, `all_faces_passed`, `all_edges_passed`, `all_corners_passed`, `all_regions_passed`, arrays of `face_regions[6]`, `edge_regions[12]`, `corner_regions[8]`. |

# Helper Function

| Function | Description |
| :--- | :--- |
| `All<6>` | Checks if all face regions passed. |
| `All<12>` | Checks if all edge regions passed. |
| `All<8>` | Checks if all corner regions passed. |
| `test_simple_bc()` | Tests a single region type by applying the kernel function to expected boundary/domain values and comparing against the actual ghost zone values. |
| `test_bc_against_mesh()` | Tests a single region type by comparing ghost zone values against a reference mesh. |
| `RunSimpleTest()` | Executes a `SimpleTestCase`: loads mesh, runs task graph, tests all 6 faces + 12 edges + 8 corners. |
| `RunMeshCompareTest()` | Executes a `MeshCompareTestCase`: loads prior mesh, runs task graph, compares against expected mesh. |
| `colored_feedback()` | Prints test name with green "PASSED" or red "FAILED" ANSI coloring. |
| `colored_pass_ratio()` | Prints "N passed / M total" with color coding. |
| `PrintTestResultRegion()` | Prints per-region details: per-field error counts, sample cell errors (controlled by `debug_bc_errors` flag). |
| `PrintTestResult()` | Prints full test results: pass/fail summary, per-region detail for all failed regions. |
| `test_periodic_boundary_queries()` | Tests `acGridTaskGraphHasPeriodicBoundcondsX/Y/Z()` API against all 8 combinations of periodic/non-periodic boundaries. |

# Test Regions

The ghost zone of a 3D decomposed mesh is decomposed into 26 regions:

| Region Type | Count | Description |
| :--- | :--- | :--- |
| **Faces** | 6 | One per face direction (±X, ±Y, ±Z). These are the ghost zones adjacent to the main domain faces. |
| **Edges** | 12 | One per edge of the cuboid. Ghost zones at the intersection of two faces (e.g., +X/+Y edge). |
| **Corners** | 8 | One per corner of the cuboid. Ghost zones at the intersection of three faces (e.g., +X/+Y/+Z corner). |

Each region has an associated **normal direction** indicating which face direction it primarily belongs to, but the test verifies **all ghost cells in that region** regardless of edge/corner adjacency.

# Test Functions (Kernel Signatures)

All kernel functions share this signature:
```cpp
typedef AcReal (*boundcond_kernel_func)(int3 normal, AcReal boundary_val, AcReal domain_val,
                                        size_t r, AcMeshInfo info);
```

| Parameter | Description |
| :--- | :--- |
| `normal` | Boundary normal direction (e.g., `{1,0,0}` for +X face). |
| `boundary_val` | Value at the boundary cell. |
| `domain_val` | Value at the nearest interior domain cell. |
| `r` | Stencil distance (halo layer index). |
| `info` | Mesh information for accessing grid spacings, etc. |

# Boundary Conditions Tested

## 1. Periodic (via `test_periodic_boundary_queries`)
Tests that `acGridTaskGraphHasPeriodicBoundcondsX/Y/Z()` correctly reports periodic boundary conditions for all 8 combinations of X/Y/Z being periodic or not.

## 2. Symmetric (`BOUNDCOND_SYMMETRIC`)
**Kernel**: `mirror` — returns `domain_val` (mirrors the domain value into the ghost zone).
```cpp
auto mirror = [](int3, AcReal, AcReal domain_val, size_t, AcMeshInfo) { return domain_val; };
```
Expected behavior: ghost zone values equal the nearest domain values (reflection).

## 3. Antisymmetric (`BOUNDCOND_ANTISYMMETRIC`)
**Kernel**: `antimirror` — returns `-domain_val`.
```cpp
auto antimirror = [](int3, AcReal, AcReal domain_val, size_t, AcMeshInfo) { return -domain_val; };
```
Expected behavior: ghost zone values are the negation of the nearest domain values.

## 4. Relative Antisymmetry (`BOUNDCOND_A2`)
**Kernel**: `a2_func` — returns `2 * boundary_val - domain_val`.
```cpp
auto a2_func = [](int3, AcReal boundary_val, AcReal domain_val, size_t, AcMeshInfo) {
    return 2 * boundary_val - domain_val;
};
```
Expected behavior: the value at the ghost cell is reflected across the boundary cell value (commonly used for outflow conditions).

## 5. Prescribed Derivative (`BOUNDCOND_PRESCRIBED_DERIVATIVE`)
**Kernel**: `der_bc_func` — computes linear extrapolation based on a prescribed first derivative.
```cpp
auto der_bc_func = [](int3 normal, AcReal, AcReal domain_val, size_t r, AcMeshInfo inf) {
    AcReal d = 0.0;
    if (normal.x != 0)      d = inf.real_params[AC_dsx] * normal.x;
    else if (normal.y != 0) d = inf.real_params[AC_dsy] * normal.y;
    else if (normal.z != 0) d = inf.real_params[AC_dsz] * normal.z;
    AcReal der_val  = inf.real_params[AC_boundary_derivative];
    AcReal distance = AcReal(2 * r) * d;
    return domain_val + distance * der_val;
};
```
Expected behavior: linear extrapolation `ghost = domain_val + (2 * r * ds) * prescribed_derivative`.

The parameter `AC_boundary_derivative` is set to `3.0` in the test setup.

# Program Flow

## 1. MPI Initialization & Configuration
1. `MPI_Init(NULL, NULL)` — Start MPI.
2. `acLoadConfig(AC_DEFAULT_CONFIG, &info)` — Load mesh configuration.
3. Set `AC_boundary_derivative = 3.0` in `info.real_params`.
4. `acGridInit(info)` — Initialize the grid (allocate device buffers, setup MPI communicators).

## 2. Mesh Setup
1. Rank 0 creates and randomizes a host mesh: `acHostMeshCreate`, `acHostMeshRandomize`.
2. `acGridLoadMesh(STREAM_DEFAULT, mesh)` — Upload mesh to all devices.
3. `acGridLoadScalarUniform(STREAM_DEFAULT, AC_dt, FLT_EPSILON)` — Load constant timestep.
4. `acGridSynchronizeStream(STREAM_DEFAULT)` — Wait for completion.

## 3. Periodic Boundary Query Test
`test_periodic_boundary_queries(pid)` is called — tests `acGridTaskGraphHasPeriodicBoundcondsX/Y/Z()` for all 8 combinations of periodic/non-periodic boundaries. Results printed to stdout.

## 4. Task Graph Construction

For each boundary condition type, a task graph is built with the same structure:
```cpp
acGridBuildTaskGraph({
    acHaloExchange(all_fields),
    acBoundaryCondition(BOUNDARY_X, BOUNDCOND_*, all_fields[, params]),
    acBoundaryCondition(BOUNDARY_Y, BOUNDCOND_*, all_fields[, params]),
    acBoundaryCondition(BOUNDARY_Z, BOUNDCOND_*, all_fields[, params])
})
```

The four constructed graphs:

| Graph | Variable | BC Type | Parameters |
| :--- | :--- | :--- | :--- |
| Symmetric BC | `symmetric_bc_graph` | `BOUNDCOND_SYMMETRIC` | None |
| Antisymmetric BC | `antisymmetric_bc_graph` | `BOUNDCOND_ANTISYMMETRIC` | None |
| Relative Antisymmetry | `relative_antisymmetry_bc_graph` | `BOUNDCOND_A2` | None |
| Prescribed Derivative | `prescribed_der_bc_graph` | `BOUNDCOND_PRESCRIBED_DERIVATIVE` | `{AC_boundary_derivative}` |

## 5. Test Execution

Each test is run via `RunSimpleTest(test, info)`:

1. Rank 0 creates a host mesh: `acHostMeshCreate`.
2. `acGridExecuteTaskGraph(test.task_graph, 1)` — Execute the BC task graph.
3. `acGridSynchronizeStream(STREAM_ALL)` — Wait for completion.
4. `acGridStoreMesh(STREAM_DEFAULT, &mesh)` — Download results.
5. Rank 0 iterates over all 26 regions (6 faces + 12 edges + 8 corners), calling `test_simple_bc()` for each.
6. Each `test_simple_bc()` call:
   - Iterates over all `NUM_VTXBUF_HANDLES` fields.
   - Iterates over all ghost cells in the region.
   - Computes the expected value using the kernel function.
   - Compares against the actual device-computed ghost value (with epsilon tolerance `1e-14`).
   - Records mismatches in `failed_cells` and increments `error_ratio.errors`.

## 6. Results Output

Rank 0 prints results:
```
Test case "Symmetric boundconds": PASSED
Test case "AntiSymmetric boundconds": PASSED
Test case "Relative antisymmetry boundconds": PASSED
Test case "Prescribed derivative boundconds": PASSED
```

If a test fails, detailed per-region output shows:
- Per-field error counts (e.g., `lnrho: 0/12345`).
- Sample cell errors with domain/ghost coordinates, expected vs produced values (controlled by `debug_bc_errors`).

## 7. Cleanup
1. `acGridDestroyTaskGraph()` for each test graph.
2. `acGridQuit()` — Destroy device resources.
3. `MPI_Finalize()` — Terminate MPI.

# Debug Flags

These compile-time preprocessor constants control output verbosity:

| Flag | Default | Description |
| :--- | :--- | :--- |
| `debug_bc_errors` | `1` | Print sample cell errors for failed regions. |
| `debug_bc_values` | `0` | Print domain/ghost value pairs (currently commented out). |

# ANSI Color Output

| Macro | Escape Sequence | Color |
| :--- | :--- | :--- |
| `RED` | `\x1B[31m` | Red |
| `GRN` | `\x1B[32m` | Green |
| `RESET` | `\x1B[0m` | Reset |

# Comparison with Other Tests

| Aspect | `boundcond_test` | `benchmark-device` |
| :--- | :--- | :--- |
| **Purpose** | Correctness verification | Performance benchmarking |
| **Test granularity** | 26 regions × NUM_VTXBUF_HANDLES fields | Per-kernel launch |
| **Verification method** | Analytical kernel comparison | CPU-vs-GPU reference |
| **Required build flag** | `AC_MPI_ENABLED` | `AC_INTEGRATION_ENABLED` |
| **Task graphs** | 4 boundary condition graphs | None (direct kernel launch) |

# Key Astaroth APIs Used

| Function | Description |
| :--- | :--- |
| `acBoundaryCondition(boundary, type, fields[, params])` | Create a boundary condition task definition. |
| `acHaloExchange(fields)` | Create a halo exchange task definition. |
| `acGridBuildTaskGraph(ops)` | Build a task graph from an array of task definitions. |
| `acGridDestroyTaskGraph(g)` | Destroy a task graph. |
| `acGridExecuteTaskGraph(g, count)` | Execute a task graph. |
| `acGridTaskGraphHasPeriodicBoundcondsX/Y/Z(g)` | Query whether a task graph contains periodic BCs in each direction. |
| `acGridLoadMesh(stream, mesh)` | Upload mesh to all devices. |
| `acGridStoreMesh(stream, &mesh)` | Download mesh from all devices. |
| `acGridLoadScalarUniform(stream, param, value)` | Load a scalar parameter uniformly to all devices. |
| `acGridInit(info)` | Initialize the distributed grid. |
| `acGridQuit()` | Destroy the distributed grid. |
| `acGridSynchronizeStream(stream)` | Synchronize all streams. |
| `acLogFromRoot_proc()` | Log message from rank 0 only. |
| `acVertexBufferIdx(x, y, z, info)` | Compute linear index from 3D coordinates. |

# Key DSL/API Enums & Constants

| Constant | Description |
| :--- | :--- |
| `BOUNDCOND_SYMMETRIC` | Symmetric (mirror) boundary condition. |
| `BOUNDCOND_ANTISYMMETRIC` | Antisymmetric boundary condition. |
| `BOUNDCOND_A2` | Relative antisymmetry (A2) boundary condition. |
| `BOUNDCOND_PRESCRIBED_DERIVATIVE` | Prescribed first derivative boundary condition. |
| `BOUNDCOND_PERIODIC` | Periodic boundary condition. |
| `BOUNDARY_X`, `BOUNDARY_Y`, `BOUNDARY_Z` | Direction enums for boundary conditions. |
| `BOUNDARY_XYZ` | All directions mask. |
| `NGHOST` | Number of ghost layers. |
| `STENCIL_ORDER` | Order of the stencil operator. |
| `AC_boundary_derivative` | Prescribed derivative value parameter. |
| `AC_dsx`, `AC_dsy`, `AC_dsz` | Grid spacings per direction. |

# Notable Observations

1. **MeshCompareTestCase is commented out** — The comparison test infrastructure (`MeshCompareTestCase`, `RunMeshCompareTest`) is fully implemented but commented out in `main()`. It would allow testing against a pre-computed reference mesh rather than analytical kernels.

2. **Rank 0 only for verification** — The actual test verification (`test_simple_bc`, `PrintTestResult`) is only executed by rank 0. All other ranks execute the task graph and download the mesh but do not verify.

3. **Uniform task graph structure** — All boundary condition task graphs follow the same pattern: `halo_exchange → BC X → BC Y → BC Z`, with only the BC type changing.

4. **Epsilon tolerance** — Cell-by-cell comparison uses `epsilon = 1e-14`, appropriate for double-precision arithmetic.

5. **Per-field granularity** — Error reporting is tracked per vertex buffer handle, allowing detection of field-specific boundary condition failures.
