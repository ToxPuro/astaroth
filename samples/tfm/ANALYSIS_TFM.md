# ⚠️ WARNING ⚠️
# **This is a LLM based analysis.**
# **THE BRANCH SHOULD NEVER BE MERGED TO master OR develop BRANCHES!**

# LLM Information
- **Model:** Qwen3.6-35B-A3B

# Overview
The `tfm` directory provides a comprehensive Test Field Method (TFM) implementation for Astaroth — a magnetohydrodynamic (MHD) simulation framework for studying small-scale dynamo and mean-field astrophysics through the evolution of magnetic test fields. It includes four executable targets (`tfm`, `tfm_benchmark`, `tfm_pipeline`, `tfm_standalone`), a DSL physics file (`mhd/mhd.ac`), INI configuration files, and two Python visualization/debugging scripts. The TFM methodology solves for multiple pairs of magnetic vector potential test fields (b11, b12, b21, b22) advected by a hydrodynamic velocity field, computing mean-field coefficients (alpha, beta/turbulent diffusivity) from the correlations between fluctuating fields and mean profiles.

# Directory Structure & File Descriptions

| File | Description |
| :--- | :--- |
| `CMakeLists.txt` | Build config: creates 4 executables (`tfm`, `tfm_benchmark`, `tfm_pipeline`, `tfm_standalone`) with varying dependency sets. `tfm_mpi` is commented out. Links `astaroth_core`, `astaroth_utils`, and optionally `astaroth_forcing` + `inih`. |
| `build.sh` | Separate build script (required because DSL file is in a subdirectory). Sets cmake options including `DSL_MODULE_DIR` and `PROGRAM_MODULE_DIR` pointing to `samples/tfm/`. |
| `tfm.c` | Low-level device API test program (GPU vs CPU integration verification, scalar/vector/Alfven reduction tests, profile XY-average and derivative verification). Only compiled when `LTFM` is defined. |
| `tfm_mpi.c` | MPI-accelerated version with device/host GPU+CPU comparison. Same test structure as `tfm.c` but uses `acGrid*` MPI API. Includes `acAbort` override and max-processes guard (2×2×4=16). |
| `tfm_pipeline.c` | Production simulation pipeline with benchmarking. Runs `tfm_pipeline()`, verifies against CPU, writes CSV benchmark results, includes CUDA profiling integration. |
| `tfm_benchmark.c` | Per-kernel benchmark utility. Benchmarks `singlepass_solve` and optionally `singlepass_solve_tfm_b21` with dry-run/reset/re-randomize cycle per sample. Outputs per-kernel timing CSV with TPB (threads-per-block) information. |
| `tfm_standalone.c` | Full standalone simulation with INI config parsing, forcing generation, simulation loop, diagnostic output, and B-field profile initialization (sine/cosine waves). Most feature-complete executable. |
| `stencil_loader.h` | Stencil coefficient generator with 6th-order central/upwind derivatives, cross-derivatives, shock viscosity (max5/SmoothWeight kernels), and upwinding coefficients. |
| `bfield-init-function-test.py` | Python visualization for B-field profile initialization. Plots sine/cosine waves, Roberts flow fields, and reads `cosine-profile.out` / `sine-profile.out`. |
| `visualize-debug.py` | Python visualization for simulation output. Animate mesh fields, plot snapshots, and profile plots using matplotlib. |
| `mhd/` | DSL directory containing `mhd.ac` (physics kernels, stencils, forcing functions, RK3 integration) and INI config files. |
| `mhd/mhd.ac` | 1116-line DSL physics file with hydrodynamics, MHD, TFM test field kernels, mean-field profiles, forcing functions, and mathematical operations (gradient, curl, laplace, stress tensor). |
| `mhd/mhd.ini` | Default INI config: 32³ grid, 2π periodic box, 10 simulation steps, SOCA disabled, magnetic laplace diff enabled. |
| `mhd/mhd-test.ini` | Test INI config: 32³ grid, 2500 simulation steps, SOCA enabled, stronger forcing (magnitude=1), more frequent output. |
| `cases/` | Four case-specific INI configs (laplace-soca, laplace-nonsoca, soca, nonsoca). |
| `model/` | 10 model subdirectories (e.g., `soca-roberts-uualfvendt/`, `nonsoca-turbulence-uudt/`). |

# Compile-Time Requirements

| Setting | Value | Description |
| :--- | :--- | :--- |
| MPI | Config-dependent | `tfm_mpi.c` requires MPI; `tfm_mpi` target is commented out in CMakeLists.txt. |
| `LTFM` | Build-dependent | `tfm.c` and `tfm_mpi.c` require `LTFM` defined; print error and exit if disabled. |
| `AC_INTEGRATION_ENABLED` | Build-dependent | `tfm_pipeline.c` and `tfm_benchmark.c` require this; print error and exit if disabled. |
| `AC_INTEGRATION_ENABLED` | Config-dependent | If disabled, `tfm_pipeline.c` prints error and exits. |
| `AC_MPI_ENABLED` | Config-dependent | If disabled, `tfm_mpi.c` prints error and exits. |

Compile options: Inherited from `astaroth_core` (typically `-Wall -Wextra -Werror -Wdouble-promotion -Wfloat-conversion -Wshadow`).

# Compile-Time Options

| Macro | Default | Description |
| :--- | :--- | :--- |
| `LTFM` | Build-dependent | If disabled, `tfm.c` and `tfm_mpi.c` print error and exit. |
| `AC_INTEGRATION_ENABLED` | Build-dependent | If disabled, `tfm_pipeline.c` and `tfm_benchmark.c` print error and exit. |
| `AC_MPI_ENABLED` | Config-dependent | If disabled, `tfm_mpi.c` prints error and exits. |
| `AC_INTEGRATION_ENABLED` | Config-dependent | If disabled, `tfm_pipeline.c` prints error and exits. |
| `AC_INTEGRATION_ENABLED` | Config-dependent | If disabled, `tfm_pipeline.c` prints error and exits. |
| `AC_DOUBLE_PRECISION` | Build-dependent | Controls `DOUBLE_PRECISION` flag in benchmark CSV output. |
| `LSHOCK` | Build-dependent | Enables shock viscosity stencils (max5, SmoothWeight) in `stencil_loader.h`. |
| `LMAGNETIC` | 0 | Enables magnetic vector potential fields in `mhd.ac`. |
| `LFORCING` | 1 | Enables helical forcing in `mhd.ac`. |
| `LUPWD` | 1 | Enables upwinding stencils in `mhd.ac`. |
| `LTFM_HANDWRITTEN` | 1 | Enables handwritten TFM kernels (SOCA not supported if off). |
| `LOO` | 1 | Enables vorticity/omega fields in `mhd.ac`. |

# Input Parameters / Command-Line Interface

| Program | Parameters | Description |
| :--- | :--- | :--- |
| `tfm_pipeline` | `nx ny nz jobid num_samples verify salt` (argv[1]-argv[7]) | Mesh dimensions, job ID, sample count, verification flag, random salt. |
| `tfm_benchmark` | `nx ny nz jobid num_samples verify salt` (argv[1]-argv[7]) | Same as pipeline but focuses on per-kernel benchmarking. |
| `tfm_standalone` | `--config <path>` / `-c <path>` | Path to INI config file. Defaults to `AC_DEFAULT_TFM_CONFIG` (compiled-in, points to `mhd.ini`). |

# Program Flow

## `tfm.c` — Device API Verification Test

### 1. CPU Setup (rank 0 only)
- `srand(321654987)` — fixed seed for reproducibility.
- `acLoadConfig(AC_DEFAULT_CONFIG, &info)` — load mesh configuration.
- `acSetMeshDims(32, 32, 32, &info)` — override to 32³ grid.
- `acHostMeshCreate` + `acHostMeshRandomize` for both `model` and `candidate`.

### 2. GPU Initialization & Stencil Loading
- `acDeviceCreate(0, info, &device)` — create GPU device.
- `get_stencil_coeffs(info)` — generate stencil coefficients from `stencil_loader.h`.
- `acDeviceLoadStencils(device, STREAM_DEFAULT, stencils)` — upload stencils to GPU.
- `acDevicePrintInfo(device)` — print device info.

### 3. Profiles (rank 0 only)
- `acDeviceLaunchKernel(device, STREAM_DEFAULT, init_profiles, ...)` — initialize profiles to zero.
- `acDeviceSwapAllProfileBuffers(device)` — swap profile buffers.

### 4. Boundary Conditions + Mesh Store
- `acDeviceLoadMesh(device, STREAM_DEFAULT, model)` — load mesh.
- `acDevicePeriodicBoundconds(device, STREAM_DEFAULT, mmin, mmax)` — apply periodic BCs.
- `acDeviceStoreMesh(device, STREAM_DEFAULT, &candidate)` — store to candidate.
- `acHostMeshApplyPeriodicBounds(&model)` + `acVerifyMesh("Boundconds", model, candidate)` — CPU-GPU verification.

### 5. Dryrun (3 substeps)
- `acDeviceIntegrateSubstep(device, STREAM_DEFAULT, i, nmin, nmax, dt)` for i=0,1,2 with dt=FLT_EPSILON.

### 6. Integration (100 steps)
- 3-substep loop × 100 iterations:
  1. `acDevicePeriodicBoundconds` — periodic boundary conditions.
  2. `acDeviceIntegrateSubstep(device, STREAM_DEFAULT, i, nmin, nmax, dt)` — RK3 substep.
  3. `acDeviceSwapBuffers(device)` — swap read/write buffers.
- After loop: `acDeviceStoreMesh` + CPU-GPU integration verification via `acHostIntegrateStep`.

### 7. Scalar Reductions Test
- Test `RTYPE_MAX, RTYPE_MIN, RTYPE_SUM, RTYPE_RMS, RTYPE_RMS_EXP` on field 0.
- `acDeviceReduceScal` vs `acHostReduceScal` comparison.

### 8. Vector Reductions Test
- Same 5 reduction types on 3 vector fields (v0, v1, v2).
- `acDeviceReduceVec` vs `acHostReduceVec` comparison.

### 9. Alfven Reductions Test
- `RTYPE_ALFVEN_MAX, RTYPE_ALFVEN_MIN, RTYPE_ALFVEN_RMS` on 4 fields (v0, v1, v2, v3).
- `acDeviceReduceVecScal` vs `acHostReduceVecScal` comparison.

### 10. Profile XY-Average Test
- Set alternating values (-1, +1 pattern) in model vertex buffer.
- `acDeviceReduceXYAverage(device, STREAM_DEFAULT, field, profile)`.
- Compare with `acHostReduceXYAverage`.

### 11. Profile Derivatives Test
- `acDeviceLaunchKernel(device, STREAM_DEFAULT, diff_profiles, ...)` — compute derz.
- `acDeviceLaunchKernel(device, STREAM_DEFAULT, diff2_profiles, ...)` — compute derzz.
- Compare with `acHostProfileDerz` and `acHostProfileDerzz`.

## `tfm_mpi.c` — MPI-Accelerated Device Test

Similar flow to `tfm.c` but uses `acGrid*` MPI API, includes max-processes guard (16 processes), and has `acAbort` override via `MPI_Abort`.

## `tfm_pipeline.c` — Production Simulation Pipeline

### 1. Command-line Parsing
- Parse `nx, ny, nz, jobid, num_samples, verify, salt` from argv.
- Generate seed: `12345 + salt + (1 + nx + ny + nz + jobid + num_samples + verify) * time(NULL)`.

### 2. Verification (if verify flag set)
- Dryrun + autotune: 3 substep integrations.
- Load/store verification: randomize → load → store → compare.
- Boundary condition verification: randomize → load → apply BCs → store → compare.
- Integration verification: 5 steps × 3 substeps each, compare device vs CPU.

### 3. Benchmarking
- Open `benchmark-device-<jobid>-<seed>.csv`.
- Warm-up: `tfm_dryrun(device, info)`.
- Benchmark loop: `num_samples` iterations of `tfm_pipeline(device, info)`.
- Per-iteration benchmark of components:
  - `acDeviceLoadIntUniform`
  - `singlepass_solve` (hydrodynamics)
  - BC for 4 hydro fields
  - `acDeviceReduceXYAverages`
  - `singlepass_solve_tfm_b11/b12/b21/b22` (4 test fields)
  - BC for 12 test fields
- CUDA profiling: run pipeline with `cudaProfilerStart/Stop`.

### 4. TFM Pipeline Execution (`tfm_pipeline`)
For each of 3 substeps:
1. `acDeviceLoadScalarUniform(AC_dt, dt)` — load timestep.
2. `acDeviceLoadIntUniform(AC_step_number, step_number)` — set step index.
3. `acDeviceLaunchKernel(singlepass_solve, dims.n0, dims.n1)` — hydrodynamics.
4. `acDevicePeriodicBoundcondStep` for 4 hydro fields.
5. `acDeviceReduceXYAverages(device, STREAM_DEFAULT)` — profile averages.
6. `acDeviceLaunchKernel(singlepass_solve_tfm_b11/b12/b21/b22)` — 4 test field kernels.
7. `acDevicePeriodicBoundcondStep` for 12 test fields.
8. `acDeviceSwapBuffers(device)` — swap buffers.
After loop: apply final BCs for all 16 fields.

### 5. B-field Profile Initialization
- `acHostInitProfileToCosineWave` / `acHostInitProfileToSineWave`.
- Initialize 12 B-field profiles + write `cosine-profile.out` / `sine-profile.out`.

### 6. Simulation Loop
- 200 steps, output every 100 steps.
- Write all vertex buffers and profiles to disk.

## `tfm_standalone.c` — Standalone Production Simulation

### 1. Argument Parsing
- Parse `--config` / `-c` option (default: compiled-in `AC_DEFAULT_TFM_CONFIG`).

### 2. INI Config Parsing
- Parse all int/real parameters from INI file.
- Validate mesh dimensions and physical parameters.
- Compute derived parameters: `dsx/dsy/dsz`, `cs2_sound`, center coordinates.
- Generate forcing parameters via `generateForcingParams()`.

### 3. Device Setup
- `acDeviceCreate(0, info, &device)` — create GPU device.
- `get_stencil_coeffs(info)` — generate stencil coefficients.
- `acDeviceLoadStencils(device, STREAM_DEFAULT, stencils)` — upload.

### 4. Random Initialization
- `acRandInitAlt(seed, count, pid)` + `srand(seed)`.
- Dryrun: 3 substeps + `tfm_run_pipeline`.

### 5. Simulation Setup
- `acDeviceResetMesh(device, STREAM_DEFAULT)` — reset mesh.
- `acDeviceSwapBuffers(device)` — swap.
- `acDevicePeriodicBoundconds(device, STREAM_DEFAULT, dims.m0, dims.m1)` — periodic BCs.
- `tfm_init_profiles(device)` — initialize B-field profiles (sine/cosine waves).
- `write_diagnostic_step(device, 0)` — write initial state.

### 6. Simulation Loop
For each step 1 to `AC_simulation_nsteps`:
1. Generate forcing: `generateForcingParams()` → `loadForcingParamsToDevice()`.
2. `tfm_run_pipeline(device)` — execute TFM pipeline.
3. If `step % output_interval == 0`: `write_diagnostic_step(device, step)`.

### 7. Pipeline Execution (`tfm_run_pipeline`)
Per step:
1. Load `AC_current_time` and compute `AC_dt` via `calc_timestep()`.
2. For each of 3 substeps:
   - Boundary conditions: hydro fields (4) + test fields (12) + derived fields (12) = 28 BC calls.
   - `acDeviceReduceXYAverages(device, STREAM_DEFAULT)` — profile averages.
   - `acDeviceLaunchKernel(singlepass_solve, ...)` — hydrodynamics.
   - `acDeviceLaunchKernel(singlepass_solve_tfm_b11/b12/b21/b22, ...)` — 4 test field kernels.
   - `acDeviceSwapBuffers(device)` — swap buffers.
3. Final boundary conditions: 28 BC calls for all fields.
4. `current_time += dt` — update simulation time.

# DSL Physics (mhd.ac)

## Field Definitions

| Category | Fields | Count |
| :--- | :--- | :--- |
| Hydrodynamics | `VTXBUF_LNRHO`, `VTXBUF_UUX`, `VTXBUF_UUY`, `VTXBUF_UUZ` | 4 |
| TFM Test Fields | `TF_a11_x/y/z`, `TF_a12_x/y/z`, `TF_a21_x/y/z`, `TF_a22_x/y/z` | 12 |
| TFM Derived Fields | `TF_uxb11_x/y/z`, `TF_uxb12_x/y/z`, `TF_uxb21_x/y/z`, `TF_uxb22_x/y/z` | 12 |
| TFM BB Fields | `TF_bb11_x/y/z`, `TF_bb12_x/y/z`, `TF_bb21_x/y/z`, `TF_bb22_x/y/z` | 12 |
| Mean-field Profiles (Z) | `PROFILE_Umean_x/y/z`, `PROFILE_ucrossbNNmean_x/y/z` (4×3), `PROFILE_BNNmean_x/y/z` (4×3) | 21 |
| Optional (LMAGNETIC) | `VTXBUF_AX, AY, AZ` | 3 |
| Optional (LENTROPY) | `VTXBUF_ENTROPY` | 1 |
| Optional (LOO) | `VTXBUF_OOX/Y/Z`, `VTXBUF_UU_DOT_OO` | 4 |

## Mean-Field TFM Methodology

The TFM computes magnetic field evolution using four test field pairs:
- `b11`: Two identical copies of magnetic field component 1
- `b12`: Two copies with different initial phases
- `b21`: Two copies with orthogonal initial conditions
- `b22`: Two copies with different initial phases

Each pair evolves according to the induction equation:
```
∂a/∂t = Umean × B + u × Bmean - η∇×J + u×B - <u×B>
```

The `roc_tf()` function implements four variants controlled by `AC_TFM_SOCA_enabled` and `AC_TFM_magnetic_laplace_diff`:
1. **SOCA reference**: `cross(Umean, bb) + cross(uu, Bmean) - eta * jj`
2. **SOCA laplace**: `cross(Umean, bb) + cross(uu, Bmean) + eta * veclaplace(aa)`
3. **Full reference**: `cross(Umean, bb) + cross(uu, Bmean) - eta * jj + cross(uu, bb) - uxbmean`
4. **Full laplace**: `cross(Umean, bb) + cross(uu, Bmean) + eta * veclaplace(aa) + cross(uu, bb) - uxbmean`

## Timestep Calculation

```cpp
const long double uu_dt = cdt * dsmin / (fabs(uumax) + sqrt(cs2_sound + vAmax*vAmax));
const long double visc_dt = cdtv * dsmin * dsmin / max(max(nu_visc, max(eta, eta_tfm)), gamma * chi);
const long double dt = min(uu_dt, visc_dt);
```

Based on Pencil Code user manual p. 38 — includes both advective (Courant) and viscous limits. Alfvén speed is optional via `TFM_DT_INCLUDE_ALFVEN` define.

## Mathematical Operations

| Operation | Description |
| :--- | :--- |
| `gradient(s)` | 6th-order central gradient (derx, dery, derz). |
| `gradient6_upwd(s)` | 6th-order upwind gradient. |
| `gradients(v)` | Full 3×3 gradient matrix of vector field. |
| `gradients_upwd(v)` | Full 3×3 upwind gradient matrix. |
| `divergence(v)` | ∂/∂x(vx) + ∂/∂y(vy) + ∂/∂z(vz). |
| `curl(v)` | Standard curl operator. |
| `laplace(s)` | ∂²/∂x² + ∂²/∂y² + ∂²/∂z². |
| `veclaplace(v)` | Component-wise laplacian. |
| `stress_tensor(v)` | Visous stress tensor (symmetric). |
| `gradient_of_divergence(v)` | Gradient of divergence. |
| `contract(mat)` | Sum of squares of all matrix elements. |

## Stencils (6th-order)

| Stencil | Pattern | Description |
| :--- | :--- | :--- |
| `derx` | [-3,-2,-1,0,1,2,3] on x | 1st order x-derivative |
| `dery` | [-3,-2,-1,0,1,2,3] on y | 1st order y-derivative |
| `derz` | [-3,-2,-1,0,1,2,3] on z | 1st order z-derivative |
| `derxx` | [-3,-2,-1,0,1,2,3] on x | 2nd order x-derivative |
| `deryy` | [-3,-2,-1,0,1,2,3] on y | 2nd order y-derivative |
| `derzz` | [-3,-2,-1,0,1,2,3] on z | 2nd order z-derivative |
| `derxy` | Diagonal + anti-diagonal | Cross derivative xy |
| `derxz` | Diagonal + anti-diagonal | Cross derivative xz |
| `deryz` | Diagonal + anti-diagonal | Cross derivative yz |
| `der6x_upwd` | [-3,-2,-1,0,1,2,3] on x | 6th order upwind x |
| `der6y_upwd` | [-3,-2,-1,0,1,2,3] on y | 6th order upwind y |
| `der6z_upwd` | [-3,-2,-1,0,1,2,3] on z | 6th order upwind z |
| `max5` (LSHOCK) | All ones | Shock detection |
| `smooth_kernel` (LSHOCK) | Jacobi weight products | Smooth weighting |

## RK3 Integration

Third-order Runge-Kutta with coefficients:
- `alpha = [0, -5/9, -153/128]`
- `beta = [1, 1/3, 15/16, 8/15]`

Three step functions: `rk3_step0`, `rk3_step1`, `rk3_step2`.

# Kernel References

| Kernel | Description |
| :--- | :--- |
| `singlepass_solve` | Main hydrodynamics kernel (RK3: continuity, momentum, induction, entropy). |
| `singlepass_solve_step0/1/2` | Per-substep hydrodynamics kernels. |
| `singlepass_solve_tfm_b11/b12/b21/b22` | TFM test field kernels (one per step variant: step0/step1/step2 = 12 kernels total in HANDWRITTEN mode). |
| `init_profiles` | Initialize mean-field profiles to zero. |
| `diff_profiles` | Compute derz of profiles. |
| `diff2_profiles` | Compute derzz of profiles. |
| `randomize` | Randomize all fields (scale ±1e-5). |
| `average_hydro` | 7×7×7 spatial averaging for hydro fields. |
| `average_tfm` | 7×7×7 spatial averaging for TFM fields. |

# API Functions Used

## Device API (tfm.c, tfm_standalone.c)

| Function | Description |
| :--- | :--- |
| `acDeviceCreate(id, info, &device)` | Create GPU device. |
| `acDeviceDestroy(device)` | Destroy device. |
| `acDeviceLoadMesh(device, stream, mesh)` | Load mesh to device. |
| `acDeviceStoreMesh(device, stream, &mesh)` | Store mesh from device. |
| `acDeviceLoadStencils(device, stream, stencils)` | Upload stencil coefficients. |
| `acDeviceLoadScalarUniform(device, stream, param, value)` | Set scalar parameter. |
| `acDeviceLoadIntUniform(device, stream, param, value)` | Set int parameter. |
| `acDeviceLoadProfile(device, data, count, profile)` | Load profile data. |
| `acDeviceStoreProfile(device, profile, data, count)` | Store profile data. |
| `acDeviceIntegrateSubstep(device, stream, step, nmin, nmax, dt)` | Execute one RK3 substep. |
| `acDevicePeriodicBoundconds(device, stream, mmin, mmax)` | Apply periodic BCs to all fields. |
| `acDevicePeriodicBoundcondStep(device, stream, field, mmin, mmax)` | Apply periodic BC to single field. |
| `acDeviceSwapBuffers(device)` | Swap read/write buffers. |
| `acDeviceSwapAllProfileBuffers(device)` | Swap all profile buffers. |
| `acDeviceResetMesh(device, stream)` | Reset mesh to zero. |
| `acDeviceLaunchKernel(device, stream, kernel, n0, n1)` | Launch arbitrary GPU kernel. |
| `acDeviceReduceScal(device, stream, type, field, &val)` | Scalar reduction. |
| `acDeviceReduceVec(device, stream, type, v0, v1, v2, &val)` | Vector reduction. |
| `acDeviceReduceVecScal(device, stream, type, v0, v1, v2, v3, &val)` | Alfven reduction. |
| `acDeviceReduceXYAverage(device, stream, field, profile)` | XY-average to profile. |
| `acDeviceReduceXYAverages(device, stream)` | Compute all XY-averages. |
| `acDeviceSynchronizeStream(device, stream)` | Wait for stream completion. |
| `acDeviceWriteMeshToDisk(device, field, filepath)` | Write mesh to binary file. |
| `acDeviceWriteProfileToDisk(device, profile, filepath)` | Write profile to binary file. |
| `acDevicePrintInfo(device)` | Print device info. |
| `acDeviceGetLocalConfig(device)` | Get local mesh config. |

## Grid API (tfm_mpi.c)

| Function | Description |
| :--- | :--- |
| `acGridInit(info)` | Initialize GPU subsystem. |
| `acGridQuit()` | Shutdown GPU subsystem. |
| `acGridLoadMesh(stream, mesh)` | Load mesh (MPI). |
| `acGridStoreMesh(stream, &mesh)` | Store mesh (MPI). |
| `acGridLoadStencils(stream, stencils)` | Load stencils (MPI). |
| `acGridPeriodicBoundconds(stream)` | Apply periodic BCs (MPI). |
| `acGridIntegrate(stream, dt)` | Integrate one timestep. |
| `acGridReduceScal(stream, type, field, &val)` | MPI scalar reduction. |
| `acGridReduceVec(stream, type, v0, v1, v2, &val)` | MPI vector reduction. |
| `acGridReduceVecScal(stream, type, v0, v1, v2, v3, &val)` | MPI Alfven reduction. |
| `acGridReduceXYAverage(stream, field, profile)` | MPI XY-average. |
| `acGridGetDefaultTaskGraph()` | Get default task graph. |

## Host API

| Function | Description |
| :--- | :--- |
| `acLoadConfig(path, &info)` | Load configuration. |
| `acSetMeshDims(nx, ny, nz, &info)` | Override mesh dimensions. |
| `acHostMeshCreate(info, &mesh)` | Allocate host mesh. |
| `acHostMeshDestroy(&mesh)` | Free host mesh. |
| `acHostMeshRandomize(&mesh)` | Randomize mesh. |
| `acHostMeshApplyPeriodicBounds(&mesh)` | Apply periodic BCs on CPU. |
| `acHostMeshSet(value, &mesh)` | Set all values. |
| `acHostIntegrateStep(mesh, dt)` | One CPU integration step. |
| `acHostReduceScal(mesh, type, field)` | CPU scalar reduction. |
| `acHostReduceVec(mesh, type, v0, v1, v2)` | CPU vector reduction. |
| `acHostReduceVecScal(mesh, type, v0, v1, v2, v3)` | CPU Alfven reduction. |
| `acHostReduceXYAverage(buffer, dims, profile)` | CPU XY-average. |
| `acHostProfileDerz(data, count, dsz, profile)` | CPU profile derivative. |
| `acHostProfileDerzz(data, count, dsz, profile)` | CPU profile second derivative. |
| `acHostInitProfileToValue(value, count, profile)` | Init profile to constant. |
| `acHostInitProfileToCosineWave(...)` | Init profile to cosine wave. |
| `acHostInitProfileToSineWave(...)` | Init profile to sine wave. |
| `acHostWriteProfileToFile(path, data, count)` | Write profile to file. |
| `acHostWriteMeshInfoToDisk(mesh, path)` | Write mesh to disk. |

## Debug/Introspection API

| Function | Description |
| :--- | :--- |
| `acVerifyMesh(label, model, candidate)` | Compare model vs candidate, return error code. |
| `acGetError(modelval, candval)` | Compute error struct. |
| `acEvalError(label, error)` | Print error, return success/failure. |
| `acPrintMeshInfo(info)` | Print mesh configuration. |

## INI/Parsing API (tfm_standalone.c)

| Function | Description |
| :--- | :--- |
| `ini_parse(path, handler, user)` | Parse INI file. |
| `acParseINI(path, info)` | Parse INI into AcMeshInfo. |
| `acVerifyMeshInfo(info)` | Verify all params are initialized. |

# Configuration Files

## `mhd.ini` (Default Config)

| Section | Key | Value |
| :--- | :--- | :--- |
| Dimensions | global_nx/ny/nz | 32 |
| Dimensions | global_sx/sy/sz | 2π |
| Boundaries | bc_type_* | 0 (periodic) |
| Simulation | simulation_nsteps | 10 |
| Simulation | output intervals | 1000000 (effectively never) |
| Physical | cdt/cdtv/cdts | 0.4/0.3/1.0 |
| Physical | nu_visc/cs_sound | 5e-2/1.0 |
| Physical | eta/eta_tfm | 1e-2/5e-2 |
| Physical | mu0 | 1.4 |
| Forcing | relhel/forcing_magnitude | 1.0/0.08 |
| Forcing | kmin/kmax | 4.5/5.5 |
| Profiles | amplitude/wavenumber | 1.0/1.0 |
| Debug | TFM_SOCA_enabled | 0 (SOCA disabled) |
| Debug | TFM_magnetic_laplace_diff | 1 (use laplace form) |

## `mhd-test.ini` (Test Config)

Differences from `mhd.ini`: 2500 steps, output every 2500/50, SOCA enabled, forcing magnitude 1, mu0=1.0.

# Notable Observations

1. **Most complex sample**: With 685 lines in `tfm_standalone.c`, 1116 lines in `mhd.ac`, plus 4 executables and multiple configuration files, this is the most physics-rich and feature-complete sample in the suite.

2. **Test Field Method (TFM)**: Implements the TFM for mean-field dynamo theory, solving for 4 pairs of magnetic vector potential test fields. The `roc_tf()` function has 4 variants controlled by `AC_TFM_SOCA_enabled` and `AC_TFM_magnetic_laplace_diff` compile-time parameters.

3. **Two kernel implementations**: `LTFM_HANDWRITTEN` (1, default) uses generic `singlepass_solve_tfm_bNN` kernels parameterized at runtime; `!LTFM_HANDWRITTEN` (2) uses 12 explicitly written kernels (one per `b11/b12/b21/b22` × `step0/step1/step2`). The handwritten version is noted as "SOCA not supported if this is off."

4. **Four executables for different purposes**: `tfm` (GPU verification test), `tfm_mpi` (MPI-accelerated verification), `tfm_pipeline` (production simulation with benchmarking), `tfm_standalone` (full standalone simulation with INI config).

5. **Separate build script required**: `build.sh` sets `DSL_MODULE_DIR` and `PROGRAM_MODULE_DIR` because the DSL file lives in a subdirectory (`mhd/`). This is a pattern unique to `tfm/` among all samples.

6. **timestep calculation follows Pencil Code**: The `calc_timestep()` function in both `tfm_pipeline.c` and `tfm_standalone.c` implements Courant conditions based on the Pencil Code user manual p. 38, including Alfvén speed in the advective limit (via optional `TFM_DT_INCLUDE_ALFVEN` define).

7. **`tfm_standalone.c` includes forcing generation**: Uses `generateForcingParams()` with helical forcing based on Pencil Code's `forcing_hel_noshear()`, supporting adjustable helicity via `AC_relhel`.

8. **Stencil loader with shock viscosity**: `stencil_loader.h` provides 6th-order central derivatives, cross-derivatives, upwind derivatives, and (when `LSHOCK` is defined) shock detection (`max5`) and smoothing (`smooth_kernel`) stencils using Jacobi weight products.

9. **B-field initialization via sine/cosine waves**: `tfm_init_profiles()` initializes all 12 B-field profiles to zero, then sets `B1c/B11` and `B2c/B21` to cosine waves, and `B1s/B12` and `B2s/B22` to sine waves, with amplitude and wavenumber from INI config.

10. **28 boundary condition calls per substep**: `tfm_run_pipeline()` calls `acDevicePeriodicBoundcondStep` 28 times per substep: 4 hydro + 12 test fields + 12 derived fields. This is significantly more than any other sample.

11. **Dual mesh creation but single use**: Both `tfm.c` and `tfm_mpi.c` create `model` and `candidate` meshes. In `tfm.c`, only `model` is loaded to GPU; `candidate` is used for verification storage. In `tfm_mpi.c`, the same pattern applies.

12. **Profile XY-average test**: `tfm.c` includes a unique test where the model vertex buffer is set to an alternating pattern (-1, +1) in the interior, then `acDeviceReduceXYAverage` is compared against `acHostReduceXYAverage`.

13. **Reduction type arrays use manual ARRAY_SIZE**: Both `tfm.c` and `tfm_mpi.c` define reduction arrays with `ARRAY_SIZE()` but include NOTE comments saying "NOTE: not using NUM_RTYPES here" and "NOTE: 2 instead of NUM_RTYPES", suggesting the code was copied/modified from other samples and the constants may be outdated.

14. **Bug in vector/Alfven reduction error checking**: In `tfm.c` lines 179 and 207, `acHostReduceVec` and `acHostReduceVecScal` are called with `v1, v1` instead of `v1, v2` for the minimum magnitude computation. This is a copy-paste bug that passes `v1` twice instead of `v0, v1, v2`.

15. **`tfm_benchmark.c` uses dry-run/reset cycle**: For each benchmark sample, it launches the kernel, resets the mesh, re-randomizes, swaps buffers, and synchronizes before timing the next run. This ensures each measurement starts from a clean state.

16. **CUDA profiling integration**: Both `tfm_pipeline.c` and `tfm_benchmark.c` call `cudaProfilerStop()` at start and optionally `cudaProfilerStart()`/`cudaProfilerStop()` for GPU profiling during benchmarking.

17. **No MPI in tfm_pipeline/benchmark**: Despite having `tfm_mpi.c`, the `tfm_pipeline.c` and `tfm_benchmark.c` files use the non-MPI `acDevice*` API and do not include MPI headers. They are single-GPU programs.

18. **INI parser validates uninitialized parameters**: `acVerifyMeshInfo()` checks all int/real/int3/real3 parameters against sentinel values (`INT_MIN`, `NAN`) and warns about any uninitialized values.

19. **`tfm_standalone.c` has `tfm_run_pipeline_original`**: An older version of the pipeline (lines 389-435) that places hydrodynamics before boundary conditions and test fields, unlike the current `tfm_run_pipeline` (lines 438-550) which applies boundary conditions first and includes derived field BCs and buffer swapping after each substep.

20. **`tfm_standalone.c` has `current_time` global variable**: Line 13 declares `static AcReal current_time = 0;` as a module-level global, updated after each pipeline call via `current_time += dt`. This is used for forcing phase calculations.

21. **`tfm_standalone.c` prints `sizeof(AcReal)`**: Line 574 prints the size of `AcReal`, useful for confirming single vs. double precision at runtime.

22. **`tfm_standalone.c` includes `ini.h` from inih**: Links against `astaroth_forcing` and `inih` (INI parser library), with include path set via `target_include_directories`.

23. **`tfm_standalone.c` uses `AC_DEFAULT_TFM_CONFIG` compile definition**: Line 22 of CMakeLists.txt defines `AC_DEFAULT_TFM_CONFIG` to point to the DSL module directory's `mhd.ini`.

24. **`tfm_standalone.c` has commented-out `loadForcingParamsToMeshInfo`**: Lines 27-45 show a commented-out function that would load forcing parameters into `AcMeshInfo` instead of directly to the device. The active code path uses `loadForcingParamsToDevice` (lines 48-68).

25. **`tfm_pipeline.c` has hardcoded `output_interval=100, nsteps=200`**: The simulation loop (lines 443-460) uses `const size_t nsteps = 200;` and `const size_t output_interval = 100;` rather than reading from config. This is inconsistent with `tfm_standalone.c` which reads from INI.

26. **`tfm_standalone.c` has dryrun before simulation**: Lines 644-648 run a dryrun (3 substeps + pipeline) before the actual simulation loop, likely for kernel compilation and autotuning.

27. **`tfm.c` hardcodes 32³ grid**: Line 47 overrides the config dimensions to `acSetMeshDims(32, 32, 32, &info)`.

28. **`tfm_mpi.c` has max_processes guard of 16**: Line 61 sets `max_devices = 2 * 2 * 4 = 16`, rejecting runs with more processes with an error message pointing to `mpitest/main.cc`.

29. **`tfm_mpi.c` overrides `acAbort`**: Lines 35-40 define a custom `acAbort()` that calls `MPI_Abort()` if MPI hasn't been finalized yet. This ensures clean MPI abort on errors.

30. **`bfield-init-function-test.py` includes Roberts flow**: Lines 45-86 implement Roberts flow fields (divergence-free velocity field used in dynamo research), with RMS normalization check.

31. **`visualize-debug.py` hardcodes 38³ reshaping**: Lines 23 and 49 reshape data to `(38, 38, 38)` — 32 interior + 2×3 halo on each side (STENCIL_DEPTH=7, (7-1)/2=3 on each side). This matches the stencil depth used in the TFM sample.

32. **Profile count is hardcoded to `dims.m1.z`**: `acDeviceStoreProfile(device, profile, candidate_profile, dims.m1.z)` uses the local mesh Z dimension, not the global Z. This accounts for MPI domain decomposition.

33. **`init_profiles()` in DSL iterates over profiles**: Line 1061 uses `for i in PROFILE_Umean_x:PROFILE_Umean_x+NUM_PROFILES` to zero all 21 profiles in one kernel.

34. **`randomize()` kernel uses `AC_rng_scale = 1e-5`**: The randomization scale is very small, indicating that the fields are initialized near zero for perturbation studies.

35. **`singlepass_solve` kernels are RK3-integrated**: Each substep kernel (`step0`, `step1`, `step2`) performs one RK3 stage, and the C code loops 3 times per integration step to complete one full timestep.

36. **`mhd.ac` supports Roberts flow mode**: Lines 507-514 define `roberts_flow()` as a forcing function used when `LHYDRO` is disabled. This provides an analytic divergence-free velocity field for testing.

37. **`tfm_standalone.c` has `calc_timestep` with Alfvén speed**: The timestep calculation uses Alfvén speed from magnetic vector potential, but the `vAmax` variable is computed via `acDeviceReduceVec` on velocity fields (lines 352-354), which would always return 0.0 for the magnetic contribution unless there is additional logic not visible in this code.

38. **`chi` is hardcoded to 0**: Line 364 in `calc_timestep` sets `chi = 0` with a TODO comment, meaning the heat conduction contribution to the viscous timestep limit is disabled.

39. **`tfm_standalone.c` has `cudaProfilerStop()` at startup**: Line 573 calls `cudaProfilerStop()` which may be a no-op if the profiler was not already running, but suggests the code was adapted from a CUDA profiling workflow.

40. **`mhd.ini` has very large output intervals (1000000)**: This effectively disables output for the default configuration, suggesting it is meant for quick testing rather than production runs.
