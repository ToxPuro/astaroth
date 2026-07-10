# ⚠️ WARNING ⚠️
# **This is a LLM based analysis.**
# **THE BRANCH SHOULD NEVER BE MERGED TO master OR develop BRANCHES!**

# LLM Information
- **Model:** Qwen3.6-35B-A3B

# Overview
The `stencil-loader` directory provides a minimal MPI-based correctness verification tool for the GPU stencil coefficient loading pipeline in Astaroth. It loads stencil derivative coefficients (1st/2nd order, cross derivatives, and upwind derivatives) from host CPU code into GPU device memory, then verifies the resulting GPU computation against a CPU reference implementation across four test phases: boundary conditions, time integration, scalar reductions, and vector reductions. The program uses random initial conditions and `FLT_EPSILON` timesteps to exercise the full stencil-based solver pipeline. It links against `astaroth_core` and `astaroth_utils` libraries. The core logic is split between `main.cc` (MPI orchestration, test phases, error evaluation) and `stencil_loader.h` (stencil coefficient computation and GPU upload).

# Directory Structure & File Descriptions

| File | Description |
| :--- | :--- |
| `CMakeLists.txt` | Build config: creates `stencil-loading` executable from `main.cc`, links `astaroth_core` and `astaroth_utils`. |
| `main.cc` | MPI-based test harness. Initializes MPI, loads config, allocates CPU host meshes, runs four test phases (boundary conditions, integration, scalar reductions, vector reductions), compares GPU vs CPU results. Only runs if `AC_MPI_ENABLED` is defined. |
| `stencil_loader.h` | Stencil coefficient loader. Computes 1st/2nd order derivatives (der1, der2), cross derivatives (derx), and 6th-order upwind derivatives (der6upwd) for x/y/z axes. Fills a 4D stencil array and uploads to GPU via `acGridLoadStencils()`. Uses hardcoded 6th-order coefficients matching the Astaroth convention. |

# Compile-Time Requirements

| Setting | Value | Description |
| :--- | :--- | :--- |
| MPI | `REQUIRED` | Program requires MPI; falls back to error message if built without MPI support. |
| `STENCIL_ORDER` | Config-dependent | Stencil order (2, 4, 6, or 8), controls loop bounds for coefficient loading. |
| `LUPWD` | Per-module | If `1`, includes 6th-order upwind derivative stencils in GPU upload. |
| `VERBOSE_PRINTING` | Config-dependent | Controls diagnostic output. |

Compile options: Inherited from `astaroth_core` (typically `-Wall -Wextra -Werror -Wdouble-promotion -Wfloat-conversion -Wshadow`).

# Compile-Time Options

| Macro | Default | Description |
| :--- | :--- | :--- |
| `AC_MPI_ENABLED` | Config-dependent | If disabled, program prints error and exits. |
| `STENCIL_ORDER` | Config-dependent | Selects coefficient tables; loops from `i = 0` to `STENCIL_ORDER`. |
| `LUPWD` | Per-module | `1` includes upwind derivative stencils (`stencil_der6x_upwd`, `stencil_der6y_upwd`, `stencil_der6z_upwd`). |
| `VERBOSE_PRINTING` | `0` | Controls verbose output. |

# Input Parameters / Command-Line Interface

No command-line arguments. The program uses `AC_DEFAULT_CONFIG` via `acLoadConfig()` for mesh configuration.

Usage: `mpirun -np <num_processes> ./stencil-loading`

# Program Flow

## 1. MPI Initialization
- `MPI_Init(NULL, NULL)`
- `MPI_Comm_size()` / `MPI_Comm_rank()` — get process count and rank.
- `srand(321654987)` — fixed random seed for reproducibility.

## 2. Config & Host Mesh Setup (rank 0 only)
- `acLoadConfig(AC_DEFAULT_CONFIG, &info)` — load mesh configuration.
- `acHostMeshCreate(info, &model)` — allocate CPU reference mesh.
- `acHostMeshCreate(info, &candidate)` — allocate GPU result buffer on CPU side.
- `acHostMeshRandomize(&model)` — randomize model mesh with pseudo-random values.
- `acHostMeshRandomize(&candidate)` — randomize candidate buffer.

## 3. GPU Initialization
- `acGridInit(info)` — initialize GPU subsystem.
- `load_stencil_from_config(info)` — compute and upload stencil coefficients.

## 4. Test Phase 1: Boundary Conditions
- `acGridLoadMesh(STREAM_DEFAULT, model)` — upload random mesh to GPU.
- `acGridPeriodicBoundconds(STREAM_DEFAULT)` — apply periodic BCs on GPU.
- `acGridStoreMesh(STREAM_DEFAULT, &candidate)` — store GPU result to host buffer.
- Rank 0: `acHostMeshApplyPeriodicBounds(&model)` — apply BCs on CPU.
- Rank 0: `acVerifyMesh("Boundconds", model, candidate)` — compare GPU vs CPU.

## 5. Test Phase 2: Time Integration
- Rank 0: `acHostMeshRandomize(&model)` — re-randomize CPU mesh.
- `acGridLoadMesh(STREAM_DEFAULT, model)` — upload to GPU.
- `acGridIntegrate(STREAM_DEFAULT, dt)` — one RK3 integration step with `dt = FLT_EPSILON`.
- `acGridPeriodicBoundconds(STREAM_DEFAULT)` — apply periodic BCs.
- `acGridStoreMesh(STREAM_DEFAULT, &candidate)` — store result.
- Rank 0: `acHostIntegrateStep(model, dt)` — CPU reference integration.
- Rank 0: `acHostMeshApplyPeriodicBounds(&model)` — apply BCs on CPU.
- Rank 0: `acVerifyMesh("Integration", model, candidate)` — compare GPU vs CPU.
- Rank 0: `acHostMeshRandomize(&model)` — re-randomize for next test.

## 6. Test Phase 3: Scalar Reductions
- `acGridLoadMesh(STREAM_DEFAULT, model)` — upload mesh.
- Rank 0: Print header "---Test: Scalar reductions---".
- For each of 2 reductions `{RTYPE_MAX, RTYPE_MIN}`:
  - `acGridReduceScal(STREAM_DEFAULT, reduction, VTXBUF_UUX, &candval)` — GPU scalar reduction.
  - Rank 0: `modelval = acHostReduceScal(model, reduction, VTXBUF_UUX)` — CPU scalar reduction.
  - Rank 0: Compute error bounds (`maximum_magnitude`, `minimum_magnitude`), call `acEvalError()`.

## 7. Test Phase 4: Vector Reductions
- Rank 0: Print header "---Test: Vector reductions---".
- For each of 2 reductions `{RTYPE_MAX, RTYPE_MIN}`:
  - `acGridReduceVec(STREAM_DEFAULT, reduction, VTXBUF_UUX, VTXBUF_UUY, VTXBUF_UUZ, &candval)` — GPU vector reduction.
  - Rank 0: `modelval = acHostReduceVec(model, reduction, VTXBUF_UUX, VTXBUF_UUY, VTXBUF_UUZ)` — CPU vector reduction.
  - Rank 0: Compute error bounds, call `acEvalError()`.
  - **Bug noted:** `minimum_magnitude` uses `VTXBUF_UUX` instead of `VTXBUF_UUZ` for the third argument (line 127).

## 8. Cleanup
- Rank 0: `acHostMeshDestroy(&model)`, `acHostMeshDestroy(&candidate)`.
- `acGridQuit()` — shutdown GPU.
- `MPI_Finalize()` — shutdown MPI.

# Stencil Coefficients (stencil_loader.h)

## 1st Order Derivative Coefficients (6th-order compact stencil)
| Coefficient | Value |
| :--- | :--- |
| `DER1_3` | `1/60` |
| `DER1_2` | `-3/20` |
| `DER1_1` | `3/4` |
| `DER1_0` | `0` |

Used as: `der1[] = {-DER1_3, -DER1_2, -DER1_1, DER1_0, DER1_1, DER1_2, DER1_3}`
= `{-1/60, 3/20, -3/4, 0, 3/4, -3/20, 1/60}` (antisymmetric, center = 0)

## 2nd Order Derivative Coefficients
| Coefficient | Value |
| :--- | :--- |
| `DER2_3` | `1/90` |
| `DER2_2` | `-3/20` |
| `DER2_1` | `3/2` |
| `DER2_0` | `-49/18` |

Used as: `der2[] = {DER2_3, DER2_2, DER2_1, DER2_0, DER2_1, DER2_2, DER2_3}`
= `{1/90, -3/20, 3/2, -49/18, 3/2, -3/20, 1/90}` (symmetric)

## Cross Derivative Coefficients
| Coefficient | Value |
| :--- | :--- |
| `DERX_3` | `2/720` (= 1/360) |
| `DERX_2` | `-27/720` (= -3/80) |
| `DERX_1` | `270/720` (= 3/8) |
| `DERX_0` | `0` |

Used as: `derx[] = {DERX_3, DERX_2, DERX_1, DERX_0, DERX_1, DERX_2, DERX_3}`
= `{2/720, -27/720, 270/720, 0, -270/720, 27/720, -2/720}` (antisymmetric across anti-diagonal)

## 6th-Order Upwind Derivative Coefficients
| Coefficient | Value |
| :--- | :--- |
| `DER6UPWD_3` | `1/60` |
| `DER6UPWD_2` | `-6/60` (= -1/10) |
| `DER6UPWD_1` | `15/60` (= 1/4) |
| `DER6UPWD_0` | `-20/60` (= -1/3) |

Used as: `der6upwd[] = {1/60, -6/60, 15/60, -20/60, 15/60, -6/60, 1/60}`
= `{1/60, -1/10, 1/4, -1/3, 1/4, -1/10, 1/60}`

## Stencil Layout
The 4D array `stencils[NUM_STENCILS][STENCIL_DEPTH][STENCIL_HEIGHT][STENCIL_WIDTH]` is indexed as:
- `stencils[stencil_derx][MID][MID][i]` — x-derivative along x-axis
- `stencils[stencil_derxx][MID][MID][i]` — x-2nd derivative along x-axis
- `stencils[stencil_dery][MID][i][MID]` — y-derivative along y-axis
- `stencils[stencil_deryy][MID][i][MID]` — y-2nd derivative along y-axis
- `stencils[stencil_derz][i][MID][MID]` — z-derivative along z-axis
- `stencils[stencil_derzz][i][MID][MID]` — z-2nd derivative along z-axis
- `stencils[stencil_derxy][MID][i][i]` and `[MID][i][STENCIL_ORDER-i]` — xy cross-derivative (antidiagonal)
- `stencils[stencil_deryz][i][i][MID]` and `[i][STENCIL_ORDER-i][MID]` — yz cross-derivative
- `stencils[stencil_derxz][i][MID][i]` and `[i][MID][STENCIL_ORDER-i]` — xz cross-derivative
- `stencils[stencil_der6x_upwd][MID][MID][i]` (if LUPWD) — x upwind
- `stencils[stencil_der6y_upwd][MID][i][MID]` (if LUPWD) — y upwind
- `stencils[stencil_der6z_upwd][i][MID][MID]` (if LUPWD) — z upwind

Where `MID = STENCIL_ORDER / 2` is the center index.

## Grid Spacing Scaling
Each stencil is scaled by the inverse grid spacing:
- X-axis stencils: `inv_dsx = 1.0 / info.real3_params[AC_ds].x`
- Y-axis stencils: `inv_dsy = 1.0 / info.real3_params[AC_ds].y`
- Z-axis stencils: `inv_dsz = 1.0 / info.real3_params[AC_ds].z`
- 2nd derivatives: `inv_ds * inv_ds` (squared)
- Cross derivatives: `inv_dsx * inv_dsy`, etc.

# GPU API Functions Used

| Function | Description |
| :--- | :--- |
| `acGridInit(info)` | Initialize GPU subsystem with mesh info. |
| `acGridQuit()` | Shutdown GPU subsystem. |
| `acGridLoadMesh(stream, mesh)` | Transfer mesh from host to GPU. |
| `acGridStoreMesh(stream, &candidate)` | Transfer mesh from GPU to host. |
| `acGridPeriodicBoundconds(stream)` | Apply periodic boundary conditions on GPU. |
| `acGridIntegrate(stream, dt)` | GPU time integration (one RK3 step). |
| `acGridReduceScal(stream, rtype, handle, &result)` | GPU scalar reduction, result stored in host pointer. |
| `acGridReduceVec(stream, rtype, h0, h1, h2, &result)` | GPU vector reduction (3-component magnitude). |
| `acGridLoadStencils(stream, stencils)` | Load all stencils at once (bulk 4D array upload). |
| `acGridLoadStencil(stream, stencil, data)` | Load individual stencil (commented-out alternative). |
| `STREAM_DEFAULT` | Alias for `STREAM_0`. |

# CPU/Host API Functions Used

| Function | Description |
| :--- | :--- |
| `acLoadConfig(config_path, &info)` | Load configuration from file. |
| `acHostMeshCreate(info, &mesh)` | Allocate CPU-side `AcMesh` structure. |
| `acHostMeshDestroy(&mesh)` | Free CPU mesh memory. |
| `acHostMeshRandomize(&mesh)` | Fill mesh with pseudo-random values. |
| `acHostMeshApplyPeriodicBounds(&mesh)` | Apply periodic BCs on CPU mesh. |
| `acHostIntegrateStep(mesh, dt)` | Single CPU integration step (reference). |
| `acHostReduceScal(mesh, rtype, handle)` | CPU scalar reduction. |
| `acHostReduceVec(mesh, rtype, h0, h1, h2)` | CPU vector reduction (3-component magnitude). |
| `acVerifyMesh(name, model, candidate)` | Compare two meshes, return `AcResult`. |
| `acGetError(modelval, candval)` | Compute error struct from model and candidate values. |
| `acEvalError(name, error)` | Evaluate error against thresholds, returns `bool`. |

# Reduction Types Tested

| Reduction | Description |
| :--- | :--- |
| `RTYPE_MAX` | Maximum value (scalar) or maximum magnitude (vector). |
| `RTYPE_MIN` | Minimum value (scalar) or minimum magnitude (vector). |

Note: Only `RTYPE_MAX` and `RTYPE_MIN` are tested. The code comments indicate "NOTE: 2 instead of NUM_RTYPES", meaning `RTYPE_SUM`, `RTYPE_RMS`, and `RTYPE_RMS_EXP` are not exercised by this test.

# Error Verification Logic

The error evaluation uses `acGetError(modelval, candval)` which computes:
- Absolute/relative error between CPU reference (`modelval`) and GPU result (`candval`).
- Error bounds set via `maximum_magnitude` and `minimum_magnitude` from CPU reductions.
- `acEvalError(name, error)` — checks error against configurable thresholds per reduction type.
- `ERRCHK_ALWAYS(res == AC_SUCCESS)` — hard assertion; program aborts on failure.

# Notable Observations

1. **MPI-only program:** This is a strictly MPI-based test requiring `mpirun`. If Astaroth is built without MPI, the program prints a clear error message and exits with `EXIT_FAILURE`.

2. **Deterministic randomness:** The random seed is hardcoded to `321654987`, ensuring reproducible test results across runs. This is critical for continuous integration and numerical regression testing.

3. **Bulk stencil upload preferred:** The main code path uses `acGridLoadStencils()` to upload all stencils in a single call with a 4D array, but the commented-out alternative shows `acGridLoadStencil()` can upload individual stencils one at a time in a loop.

4. **Garbage fill verification:** Line 34 calls `acGridLoadStencils(STREAM_DEFAULT, stencils)` with a zero-initialized array before filling — this is intentional to "ensure the coefficients calculated in the next step are correct" by overwriting any pre-existing GPU stencil data.

5. **Bug in vector reduction test:** Line 127 uses `VTXBUF_UUX` instead of `VTXBUF_UUZ` for the third vector component when computing `minimum_magnitude`: `acHostReduceVec(model, RTYPE_MIN, v0, v1, v1)` — the third argument is `v1` (UUZ is `v2`, not `v1`). This means the minimum magnitude is computed for (UUX, UUY, UUY) instead of (UUX, UUY, UUZ).

6. **FLT_EPSILON timestep:** The integration test uses `dt = FLT_EPSILON` — an extremely small timestep designed to exercise the stencil computation with minimal numerical accumulation error, making GPU vs CPU comparison more likely to pass within tolerance.

7. **Partial stencil coverage:** Only 9 of the full stencil complement is loaded: `derx`, `derxx`, `dery`, `deryy`, `derz`, `derzz`, `derxy`, `deryz`, `derxz` plus optionally 3 upwind stencils. Diffusion stencils, forcing stencils, and other domain-specific stencils are not loaded by this test.

8. **Cross-derivative antidiagonal pattern:** The cross derivatives (derxy, deryz, derxz) use an antidiagonal pattern where `stencil[MID][i][i]` is set to the positive coefficient and `stencil[MID][i][STENCIL_ORDER-i]` is set to the negative. This matches the compact cross-derivative approach seen in other Astaroth samples.

9. **Two-mesh comparison architecture:** The test allocates two CPU-side meshes: `model` (the CPU reference computation) and `candidate` (the GPU result stored on the host). This allows rank 0 to compare results element-wise via `acVerifyMesh()`.

10. **Rank 0 exclusive host operations:** All host-side operations (mesh allocation, randomization, CPU integration, error evaluation, mesh destruction) are guarded by `if (pid == 0)`. This avoids duplicate work and potential memory conflicts in MPI execution, since only rank 0 performs host-side comparison.

11. **Stencil coefficient signs follow mathematical conventions:** 1st order derivatives use antisymmetric coefficients (positive on one side, negative on the other), 2nd order derivatives use symmetric coefficients, and cross derivatives use an antisymmetric pattern across the anti-diagonal — all matching standard finite difference conventions.

12. **Upwind derivatives use same coefficient pattern as standalone:** The `DER6UPWD_` coefficients match those in `samples/standalone_mpi/stencil_loader.h` and `samples/standalone/ANALYSIS_STANDALONE.md` (upwind derivative 6th order).

13. **No command-line configuration:** Unlike `standalone/ac_run`, this program has no CLI options. It always uses `AC_DEFAULT_CONFIG` and the hardcoded random seed. This makes it suitable for automated testing but not for interactive parameter exploration.

14. **Single integration step per test:** The integration test performs exactly one `acGridIntegrate()` step (plus BCs) before comparison. This minimizes the chance of error accumulation masking individual stencil bugs.

15. **Commented DSL reference coefficients:** Lines 100–209 contain commented-out stencil coefficient values with explicit coordinate indexing, serving as a human-readable reference for the DSL (Domain-Specific Language) compiler's expected stencil layout.

16. **STENCIL_ORDER loops are inclusive:** The loops iterate `i = 0` to `i <= STENCIL_ORDER` (i.e., `i < STENCIL_ORDER + 1`), filling `STENCIL_ORDER + 1` positions. For 6th order, this means indices 0–6 (7 values), which matches the `STENCIL_WIDTH` dimension being `STENCIL_ORDER + 1`.

17. **No vector reduction sum/RMS tested:** Only `RTYPE_MAX` and `RTYPE_MIN` are tested for both scalar and vector reductions. `RTYPE_SUM`, `RTYPE_RMS`, and `RTYPE_RMS_EXP` would require additional test code.

18. **Periodic BCs only:** The boundary condition test only exercises periodic boundary conditions (`acGridPeriodicBoundconds`). General boundary conditions (`acGridBoundcondStepGBC`) are not tested.

19. **Single buffer reduction:** All reductions use `VTXBUF_UUX` as the single buffer for scalar reductions and `(VTXBUF_UUX, VTXBUF_UUY, VTXBUF_UUZ)` for vector reductions. No other vertex buffers (density, magnetic, entropy) are tested.

20. **Standalone vs stencil-loader distinction:** This sample focuses exclusively on stencil coefficient loading and basic verification, whereas `samples/standalone_mpi/` provides a full simulation toolkit with autotest, benchmark, simulation, and rendering modes. The `stencil-loader` is a minimal, targeted test.
