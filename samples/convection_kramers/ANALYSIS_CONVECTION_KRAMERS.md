# ⚠️ WARNING ⚠️
# **This is a LLM based analysis.**
# **THE BRANCH SHOULD NEVER BE MERGED TO master OR develop BRANCHES!**

# LLM Information
- **Model:** Qwen3.6-35B-A3B

# Overview
The `convection_kramers` sample is a boundary condition verification test for Astaroth, structurally nearly identical to `boundcond_test` but with a distinguishing addition: the **"pilot BCs" test** that uses `acSpecialMHDBoundaryCondition` for MHD-specific boundary conditions (entropy blackbody radiation and entropy prescribed heat flux). It tests four standard boundary condition types (symmetric, antisymmetric, prescribed derivative, relative antisymmetry) plus the pilot MHD test across all ghost zone regions (faces, edges, corners).

# Directory Structure & File Descriptions

| File | Description |
| :--- | :--- |
| `CMakeLists.txt` | Build configuration; compiles `main.cc` into the `convection_kramers` executable, linked against `astaroth_core` and `astaroth_utils`. |
| `dummy-file-can-remove` | Placeholder file (no functional role). |
| `main.cc` | MPI-required C++ program that builds task graphs for 5 boundary condition types, executes them, and verifies all ghost zone regions against analytical expectations. |

# Precondition

The executable requires MPI support:

| Condition | Macro | Error Message |
| :--- | :--- | :--- |
| MPI support | `AC_MPI_ENABLED` | "The library was built without MPI support, cannot run mpitest. Rebuild Astaroth with cmake -DMPI_ENABLED=ON .. to enable." |

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

# Kernel Function Signature

The kernel function signature in `convection_kramers` **differs from `boundcond_test`** — it does **not** take the `int3 normal` parameter:

```cpp
typedef AcReal (*boundcond_kernel_func)(AcReal boundary_val, AcReal domain_val, size_t r,
                                        AcMeshInfo info);
```

| Parameter | Description |
| :--- | :--- |
| `boundary_val` | Value at the boundary cell. |
| `domain_val` | Value at the nearest interior domain cell. |
| `r` | Stencil distance (halo layer index). |
| `info` | Mesh information for accessing grid spacings, etc. |

# Test Regions

The ghost zone of a 3D decomposed mesh is decomposed into 26 regions:

| Region Type | Count | Description |
| :--- | :--- | :--- |
| **Faces** | 6 | One per face direction (±X, ±Y, ±Z). These are the ghost zones adjacent to the main domain faces. |
| **Edges** | 12 | One per edge of the cuboid. Ghost zones at the intersection of two faces. |
| **Corners** | 8 | One per corner of the cuboid. Ghost zones at the intersection of three faces. |

Each region has an associated **normal direction** indicating which face direction it primarily belongs to.

# Boundary Conditions Tested

## 1. Symmetric (`BOUNDCOND_SYMMETRIC`)
**Kernel**: `mirror` — returns `domain_val` (mirrors the domain value into the ghost zone).
```cpp
auto mirror = [](AcReal, AcReal domain_val, size_t, AcMeshInfo) { return domain_val; };
```
Task graph: `acHaloExchange(all_fields) → BC X symmetric → BC Y symmetric → BC Z symmetric`.

## 2. Antisymmetric (`BOUNDCOND_ANTISYMMETRIC`)
**Kernel**: `antimirror` — returns `-domain_val`.
```cpp
auto antimirror = [](AcReal, AcReal domain_val, size_t, AcMeshInfo) { return -domain_val; };
```
Task graph: `acHaloExchange(all_fields) → BC X antisymmetric → BC Y antisymmetric → BC Z antisymmetric`.

## 3. Prescribed Derivative (`BOUNDCOND_PRESCRIBED_DERIVATIVE`)
**Kernel**: `der_bc_func` — computes linear extrapolation based on a prescribed first derivative.
```cpp
auto der_bc_func = [](AcReal, AcReal domain_val, size_t r, AcMeshInfo inf) {
    AcReal d        = inf.real_params[AC_dsx];
    AcReal der_val  = inf.real_params[AC_boundary_derivative];
    AcReal distance = AcReal(2 * r) * d;
    return domain_val + distance * der_val;
};
```
**Key difference from `boundcond_test`**: Uses only `AC_dsx` (assumes isotropic grid where `AC_dsx = AC_dsy = AC_dsz`). Does not branch on normal direction.

`AC_boundary_derivative` is set to **`prescribed_val = 6.0`** (vs. `3.0` in `boundcond_test`).

Task graph passes `bc_param` (`AC_boundary_derivative`) to each `acBoundaryCondition` call.

## 4. Relative Antisymmetry (`BOUNDCOND_A2`)
**Kernel**: `a2_func` — returns `2 * boundary_val - domain_val`.
```cpp
auto a2_func = [](AcReal boundary_val, AcReal domain_val, size_t, AcMeshInfo) {
    return 2 * boundary_val - domain_val;
};
```
**Key difference from `boundcond_test`**: Task graph passes `bc_param` to `acBoundaryCondition` calls.

Task graph: `acHaloExchange(all_fields) → BC X A2 → BC Y A2 → BC Z A2`.

## 5. Pilot MHD BCs (Special MHD Boundary Conditions)
This test is unique to `convection_kramers` and does not exist in `boundcond_test`. It exercises **special MHD boundary conditions** that are not verified against analytical kernels (it is a structural smoke test, not a correctness test).

| Boundary | Condition | Fields Affected |
| :--- | :--- | :--- |
| `BOUNDARY_Z` (both faces) | `BOUNDCOND_PERIODIC` | All fields |
| `BOUNDARY_Y` (both faces) | `BOUNDCOND_PERIODIC` | All fields |
| `BOUNDARY_Z_TOP` | `SPECIAL_MHD_BOUNDCOND_ENTROPY_BLACKBODY_RADIATION` | All fields |
| `BOUNDARY_Z_BOT` | `SPECIAL_MHD_BOUNDCOND_ENTROPY_PRESCRIBED_HEAT_FLUX` | All fields |
| `BOUNDARY_Z` | `BOUNDCOND_A2` | `lnrho` only |
| `BOUNDARY_Z` | `BOUNDCOND_SYMMETRIC` | `uux`, `uuy` |
| `BOUNDARY_Z` | `BOUNDCOND_ANTISYMMETRIC` | `uuz` |
| `BOUNDARY_Z_TOP` | `BOUNDCOND_SYMMETRIC` | `ax`, `ay` |
| `BOUNDARY_Z_BOT` | `BOUNDCOND_ANTISYMMETRIC` | `ax`, `ay` |
| `BOUNDARY_Z_TOP` | `BOUNDCOND_ANTISYMMETRIC` | `az` |
| `BOUNDARY_Z_BOT` | `BOUNDCOND_SYMMETRIC` | `az` |

**Field groupings:**
| Variable | Fields |
| :--- | :--- |
| `all_fields` | All `NUM_VTXBUF_HANDLES` vertex buffer handles |
| `lnrho` | `VTXBUF_LNRHO` |
| `uux_uuy` | `VTXBUF_UUX`, `VTXBUF_UUY` |
| `uuz` | `VTXBUF_UUZ` |
| `ax_ay` | `VTXBUF_AX`, `VTXBUF_AY` |
| `az` | `VTXBUF_AZ` |

**Note**: This test is not added to `test_cases` and therefore is not run or verified. It appears to be a draft/development prototype.

# Program Flow

## 1. MPI Initialization & Configuration
1. `MPI_Init(NULL, NULL)` — Start MPI.
2. `acLoadConfig(AC_DEFAULT_CONFIG, &info)` — Load mesh configuration.
3. Set `AC_boundary_derivative = 6.0` (prescribed_val) in `info.real_params`.
4. `acGridInit(info)` — Initialize the grid.

## 2. Mesh Setup
1. Rank 0 creates and randomizes a host mesh: `acHostMeshCreate`, `acHostMeshRandomize`.
2. `acGridLoadMesh(STREAM_DEFAULT, mesh)` — Upload mesh to all devices.
3. `acGridLoadScalarUniform(STREAM_DEFAULT, AC_dt, FLT_EPSILON)` — Load constant timestep.
4. `acGridSynchronizeStream(STREAM_DEFAULT)` — Wait for completion.

## 3. Task Graph Construction

### Pilot BCs (Draft — not run)
```cpp
AcTaskGraph* pilot_bcs = acGridBuildTaskGraph({
    acHaloExchange(all_fields),
    acBoundaryCondition(BOUNDARY_X, BOUNDCOND_PERIODIC, all_fields),
    acBoundaryCondition(BOUNDARY_Y, BOUNDCOND_PERIODIC, all_fields),
    acSpecialMHDBoundaryCondition(BOUNDARY_Z_TOP, SPECIAL_MHD_BOUNDCOND_ENTROPY_BLACKBODY_RADIATION),
    acSpecialMHDBoundaryCondition(BOUNDARY_Z_BOT, SPECIAL_MHD_BOUNDCOND_ENTROPY_PRESCRIBED_HEAT_FLUX),
    acBoundaryCondition(BOUNDARY_Z, BOUNDCOND_A2, lnrho),
    acBoundaryCondition(BOUNDARY_Z, BOUNDCOND_SYMMETRIC, uux_uuy),
    acBoundaryCondition(BOUNDARY_Z, BOUNDCOND_ANTISYMMETRIC, uuz),
    acBoundaryCondition(BOUNDARY_Z_TOP, BOUNDCOND_SYMMETRIC, ax_ay),
    acBoundaryCondition(BOUNDARY_Z_BOT, BOUNDCOND_ANTISYMMETRIC, ax_ay),
    acBoundaryCondition(BOUNDARY_Z_TOP, BOUNDCOND_ANTISYMMETRIC, az),
    acBoundaryCondition(BOUNDARY_Z_BOT, BOUNDCOND_SYMMETRIC, az)
});
```

### Standard BC Tests (4 test cases)

| Graph | BC Type | Parameters |
| :--- | :--- | :--- |
| `symmetric_bc_graph` | `BOUNDCOND_SYMMETRIC` | None |
| `antisymmetric_bc_graph` | `BOUNDCOND_ANTISYMMETRIC` | None |
| `prescribed_der_bc_graph` | `BOUNDCOND_PRESCRIBED_DERIVATIVE` | `{AC_boundary_derivative}` |
| `relative_antisymmetry_bc_graph` | `BOUNDCOND_A2` | `{AC_boundary_derivative}` |

All graphs follow the pattern: `halo_exchange → BC X → BC Y → BC Z`.

## 4. Test Execution

Each standard test case is run via `RunSimpleTest(test, info)` (same logic as `boundcond_test`):

1. Rank 0 creates a host mesh: `acHostMeshCreate`.
2. `acGridExecuteTaskGraph(test.task_graph, 1)` — Execute the BC task graph.
3. `acGridSynchronizeStream(STREAM_ALL)` — Wait for completion.
4. `acGridStoreMesh(STREAM_DEFAULT, &mesh)` — Download results.
5. Rank 0 iterates over all 26 regions, calling `test_simple_bc()` for each.
6. Each `test_simple_bc()` call verifies cells against the kernel function with `epsilon = 1e-14`.

## 5. Mesh Compare Test (Commented Out)

A commented-out comparison test infrastructure exists at lines 814–833, similar to `boundcond_test`:
```cpp
std::vector<MeshCompareTestCase> comparison_test_cases;
// acHostMeshCreate, acGridStoreMesh, acGridLoadMesh setup
// comparison_test_cases.push_back(MeshCompareTestCase{"...", relative_antisymmetry_bc_graph, mesh, mesh});
```

## 6. Results Output

Rank 0 prints results:
```
Test case "Symmetric boundconds": PASSED
Test case "AntiSymmetric boundconds": PASSED
Test case "Prescribed derivative boundconds": PASSED
Test case "Relative antisymmetry boundconds": PASSED
```

## 7. Cleanup & Finalization
1. `acGridDestroyTaskGraph()` for each test graph (batch cleanup).
2. Rank 0 prints all test results.
3. `acGridQuit()` — Destroy device resources.
4. `MPI_Finalize()` — Terminate MPI.

# Debug Flags

| Flag | Default | Description |
| :--- | :--- | :--- |
| `debug_bc_errors` | `1` | Print sample cell errors for failed regions. |
| `debug_bc_values` | `0` | Print domain/ghost value pairs (commented out). |

# ANSI Color Output

| Macro | Escape Sequence | Color |
| :--- | :--- | :--- |
| `RED` | `\x1B[31m` | Red |
| `GRN` | `\x1B[32m` | Green |
| `RESET` | `\x1B[0m` | Reset |

# Comparison with `boundcond_test`

| Aspect | `boundcond_test` | `convection_kramers` |
| :--- | :--- | :--- |
| **Kernel signature** | Takes `int3 normal` as first param | No `int3 normal` param |
| **Prescribed derivative value** | `3.0` | `6.0` |
| **Prescribed derivative kernel** | Multi-branch: checks `normal.x/y/z`, uses `AC_dsx/sy/sz` | Single: uses only `AC_dsx` |
| **A2 BC params** | No `bc_param` argument | Passes `bc_param` |
| **Periodic query test** | Yes (`test_periodic_boundary_queries`) | No |
| **Pilot MHD BCs** | No | Yes (draft, not executed) |
| **Per-test cleanup** | Destroys task graph immediately after each test | Batch cleanup after all tests |
| **Logging** | `acLogFromRootProc` calls for each test result | No logging |

# Key Astaroth APIs Used

| Function | Description |
| :--- | :--- |
| `acBoundaryCondition(boundary, type, fields[, params])` | Create a boundary condition task definition. |
| `acSpecialMHDBoundaryCondition(boundary, type)` | Create a special MHD boundary condition task definition (entropy blackbody radiation, entropy prescribed heat flux). |
| `acHaloExchange(fields)` | Create a halo exchange task definition. |
| `acGridBuildTaskGraph(ops)` | Build a task graph from an array of task definitions. |
| `acGridDestroyTaskGraph(g)` | Destroy a task graph. |
| `acGridExecuteTaskGraph(g, count)` | Execute a task graph. |
| `acGridLoadMesh(stream, mesh)` | Upload mesh to all devices. |
| `acGridStoreMesh(stream, &mesh)` | Download mesh from all devices. |
| `acGridLoadScalarUniform(stream, param, value)` | Load a scalar parameter uniformly to all devices. |
| `acGridInit(info)` | Initialize the distributed grid. |
| `acGridQuit()` | Destroy the distributed grid. |
| `acGridSynchronizeStream(stream)` | Synchronize all streams. |
| `acLoadConfig(config, &info)` | Load mesh configuration. |
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
| `BOUNDARY_Z_TOP`, `BOUNDARY_Z_BOT` | Z-direction face enums (top/bot). |
| `SPECIAL_MHD_BOUNDCOND_ENTROPY_BLACKBODY_RADIATION` | Special MHD BC: entropy blackbody radiation. |
| `SPECIAL_MHD_BOUNDCOND_ENTROPY_PRESCRIBED_HEAT_FLUX` | Special MHD BC: entropy prescribed heat flux. |
| `AC_boundary_derivative` | Prescribed derivative value parameter. |
| `AC_dsx` | Grid spacing in X direction. |
| `VTXBUF_LNRHO` | Log density vertex buffer handle. |
| `VTXBUF_UUX`, `VTXBUF_UUY`, `VTXBUF_UUZ` | Momentum vertex buffer handles. |
| `VTXBUF_AX`, `VTXBUF_AY`, `VTXBUF_AZ` | Magnetic vector potential vertex buffer handles. |
| `NGHOST` | Number of ghost layers. |
| `STENCIL_ORDER` | Order of the stencil operator. |

# Notable Observations

1. **Pilot MHD BCs are not executed** — The `pilot_bcs` task graph is constructed but never pushed to `test_cases`, never executed, and never destroyed. It appears to be a development draft.

2. **Kernel signature simplification** — The `boundcond_kernel_func` in `convection_kramers` drops the `int3 normal` parameter, making all kernel functions direction-agnostic. The `der_bc_func` assumes isotropic grid (`AC_dsx = AC_dsy = AC_dsz`).

3. **A2 now requires params** — Unlike `boundcond_test`, the A2 test in `convection_kramers` passes `bc_param` to `acBoundaryCondition`, suggesting a newer API where A2 BCs can accept parameter configuration.

4. **Astrophysical naming** — The field names (`lnrho`, `uux/uuy/uuz`, `ax/ay/az`) suggest a compressible MHD code with density, momentum, and magnetic vector potential variables.

5. **Batch cleanup** — Task graphs are destroyed in a batch after all tests complete (vs. immediate destroy per test in `boundcond_test`).

6. **No periodic boundary validation** — The `test_periodic_boundary_queries()` function from `boundcond_test` is absent; periodic boundary behavior is only tested indirectly in the pilot BCs draft.
