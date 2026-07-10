# ⚠️ WARNING ⚠️
# **This is a LLM based analysis.**
# **THE BRANCH SHOULD NEVER BE MERGED TO master OR develop BRANCHES!**

# LLM Information
- **Model:** Qwen3.6-35B-A3B

# Overview
The `plasma-meets-ai-workshop` directory is a self-contained educational workshop package for the Astaroth GPU-accelerated plasma simulation framework. It contains a progressive series of exercises ranging from a simple blur filter to full 3D magnetohydrodynamics (MHD) with subgrid-scale (SGS) turbulence modeling. The workshop is designed for a "Plasma Physics Meets AI" presentation and targets participants who need to learn the Astaroth Device API and Domain-Specific Language (DSL) through hands-on coding exercises. Each exercise builds on the previous one: (1) blur filter, (2) hydrodynamics solver, (3) hydrodynamics with SGS stresses, (4) hydrodynamics with SGS + smoothing/forcing, and (5) MHD solver. The directory also includes a `blur-demo` with completed solutions and `model-examples` subdirectories containing pre-completed versions of exercises 1–3.

# Directory Structure & File Descriptions

| File | Description |
| :--- | :--- |
| `README.md` | Workshop instructions, build instructions, exercise descriptions, and troubleshooting guide. |
| `getting-started-with-astaroth.md` | Standalone getting-started guide with build instructions, dependencies, and common issues. |
| `astaroth.conf` | Shared mesh configuration: 256×256×1 grid. |
| `animate-snapshots.py` | Python visualization script using matplotlib to animate field snapshots as X-Y slices. |
| `blur-demo/CMakeLists.txt` | Build config for the completed blur demo (Exercise 1 solution). |
| `blur-demo/blur.c` | Completed blur demo (59 lines). |
| `blur-demo/blur.ac` | Completed blur DSL kernel (20 lines, 3×3 2D filter). |
| `blur/CMakeLists.txt` | Build config for Exercise 1 (incomplete, student must fill in). |
| `blur/blur.c` | Incomplete blur host code with `acDeviceLaunchKernel` commented out. |
| `blur/blur.ac` | Incomplete blur DSL with only half the stencil coefficients filled in. |
| `hydro/CMakeLists.txt` | Build config for Exercise 2. |
| `hydro/hydro.c` | Incomplete hydro host code with `AC_dt` and kernel launch commented out. |
| `hydro/hydro.ac` | Hydro DSL with 4th-order compact finite differences, RK3 integrator, stress tensor. Continuity equation is a placeholder returning `1.0`. |
| `hydro/stencil.ach` | Shared stencil definitions: value, gradients, divergence, curl, laplace, RK3 integrator (7th-order compact FD). |
| `hydro-sgs/CMakeLists.txt` | Build config for Exercise 3. |
| `hydro-sgs/hydro-sgs.c` | Complete hydro-SGS host code (SGS stress computation kernel added). |
| `hydro-sgs/hydro-sgs.ac` | Hydro-SGS DSL: adds Smagorinsky SGS stress tensor (T00–T22), `stress_tensor_divergence`, full continuity/momentum. |
| `hydro-sgs/stencil.ach` | Identical to `hydro/stencil.ach` (7th-order compact FD). |
| `hydro-sgs-smoothed/CMakeLists.txt` | Build config for Exercise 4. |
| `hydro-sgs-smoothed/hydro-sgs-smoothed.c` | Complete hydro-SGS-smoothed host code with forcing and smoothing kernels. |
| `hydro-sgs-smoothed/hydro-sgs-smoothed.ac` | DSL with scale_velocity, forcing (vortex ring in XY plane), smooth filter, full SGS hydrodynamics. |
| `hydro-sgs-smoothed/stencil.ach` | Identical to `hydro/stencil.ach`. |
| `hydro-sgs-smoothed/filter.ach` | Pre-computed 7×7×7 box filter (343 coefficients, all equal to ~0.002915). |
| `hydro-sgs-smoothed/generate-smoothing-filter.py` | Python script that generates the filter coefficients (uniform 7³ box filter). |
| `mhd-les/CMakeLists.txt` | Build config for the MHD LES exercise (partially complete). |
| `mhd-les/mhd-les.c` | MHD LES host code: velocity scaling, forcing, smoothing, but momentum is a TODO placeholder. |
| `mhd-les/mhd-les.ac` | MHD LES DSL: includes density, momentum, magnetic field. `compute_adv` and `momentum` are incomplete (TODO). |
| `mhd-les/stencil.ach` | Identical to `hydro/stencil.ach`. |
| `mhd-les/filter.ach` | Pre-computed 7×7×7 box filter (identical to `hydro-sgs-smoothed/filter.ach`). |
| `model-examples/blur/CMakeLists.txt` | Pre-completed blur model example (duplicate of blur-demo). |
| `model-examples/blur/blur.c` | Completed blur host code. |
| `model-examples/blur/blur.ac` | Completed blur DSL (identical to `blur-demo/blur.ac`). |
| `model-examples/hydro/CMakeLists.txt` | Pre-completed hydro model example. |
| `model-examples/hydro/hydro.c` | Completed hydro host code (AC_dt loaded, kernel launched). |
| `model-examples/hydro/hydro.ac` | Completed hydro DSL (continuity filled in). |
| `model-examples/hydro-sgs/CMakeLists.txt` | Pre-completed hydro-SGS model example. |
| `model-examples/hydro-sgs/hydro-sgs.c` | Completed hydro-SGS host code. |
| `model-examples/hydro-sgs/hydro-sgs.ac` | Completed hydro-SGS DSL. |
| `slides.pdf` | Workshop presentation slides (PDF, not analyzed). |

# Compile-Time Requirements

All exercises share the same build pattern:

| Setting | Value | Description |
| :--- | :--- | :--- |
| `BUILD_SAMPLES` | `OFF` (forced) | Disables the global samples build to avoid conflicts. |
| `DSL_MODULE_DIR` | (per-exercise) | Points to the DSL source directory so the Astaroth build system knows where to find the `.ac` and `.ach` files. |

Each exercise links against `astaroth_core` and `astaroth_utils` (no MPI, no math library needed).

# Compile-Time Options

No exercise-specific compile-time macros. The shared code uses `AC_step_number` and `AC_dt` as device constant handles, loaded at runtime via `acDeviceLoadIntUniform` and `acDeviceLoadScalarUniform`.

# Input Parameters / Command-Line Interface

| Parameter | Position | Default | Description |
| :--- | :--- | :--- | :--- |
| (none) | — | — | No command-line arguments. All parameters are hardcoded or loaded from `astaroth.conf`. |

Usage: `./<exercise_name>` (single-GPU, no MPI)

Hardcoded parameters:

| Constant | Value | Description |
| :--- | :--- | :--- |
| Grid dimensions | 256×256×1 | From `astaroth.conf`. |
| `AC_dt` | `1e-3` | Fixed timestep for all time-dependent exercises. |
| Max timesteps | 2000 | All time-dependent exercises run 2000 macro-steps. |
| Snapshot interval | 100 | Snapshots written at steps 0, 100, 200, …, 2000. |
| RK3 substeps | 3 | Each macro-step uses 3 Runge-Kutta substeps. |
| DSL spacing (DSX/DSY/DSZ) | 0.04908738521 | Grid spacing used in stencil definitions (≈ 2π/128). |
| Viscosity (NU_VISC) | 5e-4 | Artificial viscosity coefficient. |
| Bulk viscosity (ZETA) | 1e-2 | Bulk viscosity coefficient. |
| Speed of sound (CS0) | 1.0 | Isentropic sound speed. |
| Smagorinsky coefficient | 1.16 | For SGS models. |

# Program Flow (All Time-Dependent Exercises)

## 1. Mesh Configuration
`acLoadConfig("../samples/plasma-meets-ai-workshop/astaroth.conf", &info)` — load 256×256×1 grid config.

## 2. GPU Device Creation
`acDeviceCreate(0, info, &device)` — create device on GPU 0.
`acDevicePrintInfo(device)` — print device info.
`acGetMeshDims(info)` — get mesh dimensions.

## 3. Initial Conditions
`acHostMeshCreate(info, &mesh)` — create host mesh.
`acHostMeshRandomize(&mesh)` — fill with random data.
`acDeviceLoadMesh(device, STREAM_DEFAULT, mesh)` — transfer to GPU.
`acDevicePeriodicBoundconds(device, STREAM_DEFAULT, dims.m0, dims.m1)` — set periodic BCs.

## 4. Snapshot 0
`acHostMeshWriteToFile(mesh, 0)` — write initial state.

## 5. Time Loop (2000 macro-steps)
For each step `i = 1..1999`:
a. Per-substep loop (3×):
   1. `acDeviceLoadScalarUniform(device, STREAM_DEFAULT, AC_dt, 1e-3)` — set timestep.
   2. `acDeviceLoadIntUniform(device, STREAM_DEFAULT, AC_step_number, substep)` — set RK3 substep.
   3. `acDeviceLaunchKernel(device, STREAM_DEFAULT, <kernel>, dims.n0, dims.n1)` — run kernel.
   4. `acDeviceSwapBuffers(device)` or `acDeviceSwapBuffer(device, field)` — swap buffers.
   5. `acDevicePeriodicBoundconds(device, STREAM_DEFAULT, dims.m0, dims.m1)` — apply periodic BCs.

b. Snapshot (every 100 steps):
   1. `acDeviceSynchronizeStream(device, STREAM_DEFAULT)` — wait for GPU.
   2. `acDeviceStoreMesh(device, STREAM_DEFAULT, &mesh)` — transfer to host.
   3. `acHostMeshWriteToFile(mesh, i)` — write to disk.

## 6. Cleanup
`acDeviceDestroy(device)`, `acHostMeshDestroy(&mesh)`.

# Exercise-Specific Program Flows

## Exercise 1: Blur Filter
1. Create device, load mesh, write snapshot 0.
2. For step 1–19: launch `blur` kernel, swap buffers, apply BCs, synchronize, store, write snapshot.
3. 3×3 2D box blur (coefficient = 1/9), applied 19 times.

## Exercise 2: Hydrodynamics
1. Create device, load mesh, write snapshot 0.
2. For each of 2000 macro-steps:
   a. 3× RK3 substep: load `AC_dt`, load `AC_step_number`, launch `hydro` kernel, swap buffers, apply BCs.
   b. Snapshot every 100 steps.
3. Solves: continuity equation (student must fill in), momentum equation (complete).

## Exercise 3: Hydro + SGS
1. Create device, load mesh, write snapshot 0.
2. For each of 2000 macro-steps:
   a. 3× RK3 substep:
      1. Launch `compute_sgs_stress` kernel → writes SGS stress tensor (T00–T22), swap each buffer.
      2. Launch `hydro_sgs` kernel → writes LNRHO, UUX, UUY, UUZ, swap each buffer.
   b. Snapshot every 100 steps.
3. SGS model: Smagorinsky eddy viscosity with Galilean-invariant strain-rate estimation.

## Exercise 4: Hydro + SGS + Smoothing + Forcing
1. Create device, load mesh, write snapshot 0.
2. Pre-simulation: launch `scale_velocity` kernel → maps velocity to [-1, 1].
3. For each of 2000 macro-steps:
   a. Apply forcing: launch `forcing` kernel → applies annular vortex force in XY plane.
   b. Apply smoothing: launch `smooth` kernel → 7×7×7 box filter on velocity.
   c. 3× RK3 substep:
      1. Launch `compute_sgs_stress` → SGS stress tensor.
      2. Launch `hydro_sgs` → continuity + momentum.
   d. Snapshot every 100 steps.
4. Forcing: annular ring force (0.4 < dist < 0.6) applied in XY plane, tangential direction.

## Exercise 5 (MHD): MHD-LES (Partially Complete)
1. Create device, load mesh, write snapshot 0.
2. Pre-simulation: `scale_velocity` on momentum components.
3. For each of 2000 macro-steps:
   a. Forcing on momentum (annular ring).
   b. Smoothing on momentum.
   c. 3× RK3 substep: launch `solve` kernel → writes RHO, RHOUX, RHOUY, RHOUZ.
4. **Status**: `momentum()` function returns `real3(0, 0, 0)` with TODO comment. `compute_adv()` is commented out. The MHD solver is intentionally incomplete as a workshop challenge.

# DSL Fields Used

| Field | Description | Used In |
| :--- | :--- | :--- |
| `field0` | Single scalar field | blur exercises |
| `UUX, UUY, UUZ` | Velocity components (vector `UU`) | hydro, hydro-sgs, hydro-sgs-smoothed |
| `LNRHO` | Log density (scalar) | hydro, hydro-sgs, hydro-sgs-smoothed |
| `T00, T01, T02, T11, T12, T22` | SGS stress tensor (6 components, symmetric) | hydro-sgs, hydro-sgs-smoothed |
| `RHO` | Density | mhd-les |
| `RHOUX, RHOUY, RHOUZ` | Momentum components (vector `RHOU`) | mhd-les |
| `BX, BY, BZ` | Magnetic field components (vector `B`) | mhd-les (unused in solve) |
| `ADVX, ADVY, ADVZ` | Advection components (vector `ADV`) | mhd-les (unused, commented out) |

# DSL Kernel Definitions

| Kernel | File | Fields Written | Description |
| :--- | :--- | :--- | :--- |
| `blur()` | blur.ac | `field0` | 3×3 2D box blur filter. |
| `hydro()` | hydro.ac | `LNRHO`, `UUX`, `UUY`, `UUZ` | Hydrodynamics: continuity + momentum (RK3). |
| `compute_sgs_stress()` | hydro-sgs.ac | `T00`, `T01`, `T02`, `T11`, `T12`, `T22` | Compute Smagorinsky SGS stress tensor. |
| `hydro_sgs()` | hydro-sgs.ac | `LNRHO`, `UUX`, `UUY`, `UUZ` | Hydro-SGS: continuity + momentum with SGS term. |
| `scale_velocity()` | hydro-sgs-smoothed.ac | `UUX`, `UUY`, `UUZ` | Map velocity to [-1, 1] via `2*v - 1`. |
| `forcing()` | hydro-sgs-smoothed.ac | `UUX`, `UUY`, `UUZ` | Annular ring forcing in XY plane. |
| `smooth()` | hydro-sgs-smoothed.ac | `UUX`, `UUY`, `UUZ` | 7×7×7 box filter on velocity. |
| `solve()` | mhd-les.ac | `RHO`, `RHOUX`, `RHOUY`, `RHOUZ` | MHD solver (incomplete, TODO). |
| `compute_adv()` | mhd-les.ac | (none, all commented out) | MHD advection (not yet implemented). |

# DSL Stencil Functions & Operators

Defined in `stencil.ach` (shared across all hydro/SGS exercises):

| Function/Stencil | Description |
| :--- | :--- |
| `value(v)` | Identity stencil (1 at center). |
| `ddx(v)`, `ddy(v)`, `ddz(v)` | 7th-order compact first derivatives. |
| `ddxx(v)`, `ddyy(v)`, `ddzz(v)` | 7th-order compact second derivatives. |
| `ddxy(v)`, `ddxz(v)`, `ddyz(v)` | 7th-order compact mixed derivatives. |
| `vecvalue(v)` | Convert vector field to `real3`. |
| `gradient(s)` | `real3(ddx(s), ddy(s), ddz(s))`. |
| `gradients(v)` | Jacobian matrix: `Matrix(gradient(v.x), gradient(v.y), gradient(v.z))`. |
| `divergence(v)` | `ddx(v.x) + ddy(v.y) + ddz(v.z)`. |
| `curl(v)` | `real3(ddy(v.z) - ddz(v.y), ddz(v.x) - ddx(v.z), ddx(v.y) - ddy(v.x))`. |
| `laplace(s)` | `ddxx(s) + ddyy(s) + ddzz(s)`. |
| `veclaplace(v)` | Component-wise Laplacian: `real3(laplace(v.x), laplace(v.y), laplace(v.z))`. |
| `gradient_of_divergence(v)` | Gradient of the divergence (mixed derivatives). |
| `contract(mat)` | Frobenius norm squared: `Σ |row_i|²`. |
| `rk3(s0, s1, roc)` | 3rd-order Runge-Kutta integrator using `AC_step_number` and `AC_dt`. |

# DSL Kernel Helpers

| Helper | Description |
| :--- | :--- |
| `stress_tensor(v)` | Compute strain-rate tensor from velocity (symmetric, trace-adjusted for incompressibility). |
| `continuity()` | Density continuity: `-dot(UU, grad(LNRHO)) - div(UU)` (complete in hydro-sgs, placeholder in hydro). |
| `momentum()` | Momentum RHS: advection + pressure + viscosity + bulk viscosity + SGS stress divergence. |
| `galilean_invariant_estimation(mat)` | Galilean-invariant strain-rate magnitude: `sqrt(2 * contract(mat))`. |
| `smagorinsky_eddy_viscosity(stress)` | Smagorinsky eddy viscosity: `(C * Δ)² * |S|`. |
| `stress_tau()` | SGS stress: `-2 * ν_t * S` (ν_t = eddy viscosity, S = strain rate). |
| `stress_tensor_divergence()` | Divergence of SGS stress tensor → `real3`. |

# Device API Functions Used

| Function | Description |
| :--- | :--- |
| `acDeviceCreate(0, info, &device)` | Create GPU device on device ID 0 with mesh config. |
| `acDevicePrintInfo(device)` | Print device information. |
| `acDeviceLoadMesh(device, stream, mesh)` | Transfer host mesh to GPU memory. |
| `acDevicePeriodicBoundconds(device, stream, m0, m1)` | Set periodic boundary conditions on ghost cells. |
| `acDeviceLaunchKernel(device, stream, kernel, n0, n1)` | Launch DSL kernel on GPU. |
| `acDeviceSwapBuffers(device)` | Swap all field buffers. |
| `acDeviceSwapBuffer(device, field)` | Swap single field buffer (used in SGS exercises for granular control). |
| `acDeviceSynchronizeStream(device, stream)` | Wait for kernel completion on stream. |
| `acDeviceStoreMesh(device, stream, &mesh)` | Transfer host mesh back from GPU. |
| `acDeviceLoadScalarUniform(device, stream, AC_dt, value)` | Set scalar device constant (`AC_dt`). |
| `acDeviceLoadIntUniform(device, stream, AC_step_number, value)` | Set integer device constant (RK3 substep). |
| `acDeviceDestroy(device)` | Destroy GPU device, free memory. |

# Host Mesh API Functions Used

| Function | Description |
| :--- | :--- |
| `acHostMeshCreate(info, &mesh)` | Allocate host mesh memory. |
| `acHostMeshRandomize(&mesh)` | Fill mesh with random values. |
| `acHostMeshApplyPeriodicBounds(&mesh)` | Apply periodic boundary conditions on host mesh (for snapshot 0). |
| `acHostMeshWriteToFile(mesh, step)` | Write mesh data to disk as binary file with step number in filename. |
| `acHostMeshDestroy(&mesh)` | Free host mesh memory. |

# Host Config API Functions Used

| Function | Description |
| :--- | :--- |
| `acLoadConfig(config_path, &info)` | Load mesh configuration from `astaroth.conf`. |
| `acGetMeshDims(info)` | Extract mesh dimensions (n0, n1, m0, m1). |

# Mesh Configuration (astaroth.conf)

```
AC_nx = 256
AC_ny = 256
AC_nz = 1
```

A thin 2D slice (1 cell in Z). All derivatives in Z are computed but operate on a single plane. This keeps memory usage low for workshop exercises.

# Filter Coefficients

The 7×7×7 box filter in `filter.ach` has 343 coefficients, all equal to `1/343 = 0.0029154518950437317`. The filter spans indices `[-3, -3, -3]` to `[3, 3, 3]` (7³ = 343 cells). Generated by `generate-smoothing-filter.py` which iterates over the order-6 grid and outputs `1/(order+1)³` for each stencil point.

# Exercise Completeness Matrix

| Exercise | Host Code | DSL | Status |
| :--- | :--- | :--- | :--- |
| Blur demo | ✅ Complete | ✅ Complete | Reference solution. |
| Blur (Exercise 1) | ⬜ Incomplete | ⬜ Incomplete | Student fills `acDeviceLaunchKernel` and stencil. |
| Hydro (Exercise 2) | ⬜ Incomplete | ⬜ Incomplete | Student fills `AC_dt`, kernel launch, continuity. |
| Hydro-SGS (Exercise 3) | ✅ Complete | ✅ Complete | Fully implemented with SGS stress. |
| Hydro-SGS-Smoothed (Exercise 4) | ✅ Complete | ✅ Complete | Full pipeline with forcing and smoothing. |
| MHD-LES | ⬜ Incomplete | ⬜ Incomplete | `momentum()` is TODO, `compute_adv()` is commented out. |
| Model examples (blur/hydro/hydro-sgs) | ✅ Complete | ✅ Complete | Pre-completed versions for reference. |

# Notable Observations

1. **Progressive exercise design**: The workshop builds from a trivial 3×3 blur (5 lines of DSL) to a full 3D hydrodynamic solver with SGS turbulence modeling (~100 lines of DSL). This progressive difficulty curve is pedagogically sound.

2. **256×256×1 grid**: The single-cell Z dimension is a deliberate choice to keep workshop exercises lightweight. It means the 3D compact FD stencils operate on a single plane, but the Z-derivative operators still contribute zero to the results. This is fine for learning but would need `AC_nz > 1` for any real 3D simulation.

3. **Device API vs. Grid API**: All exercises use the standalone `acDevice*` API (device-level control), not the `acGrid*` API used in production samples. This gives participants explicit control over buffer swapping, constant loading, and kernel launches — essential for understanding the simulation loop.

4. **Dual buffer-swap patterns**: The exercises demonstrate two patterns: `acDeviceSwapBuffers(device)` (swap all buffers atomically) and `acDeviceSwapBuffer(device, field)` (swap individual fields). The SGS exercises use the per-field pattern because the SGS stress tensor (T00–T22) needs to be swapped after `compute_sgs_stress` but before `hydro_sgs`.

5. **Periodic boundary conditions applied twice per substep**: Once after kernel execution and after buffer swap. This ensures ghost cells are properly filled for the next iteration.

6. **RK3 conditional on AMD vs. NVIDIA**: The `rk3()` function contains a comment noting "abysmal performance on AMD for some reason" for the `if AC_step_number > 0` branch, with a commented-out workaround. This is an important portability note for GPU code running on AMD GPUs.

7. **7th-order compact finite differences**: The `stencil.ach` file implements high-order compact FD stencils using 7-point (3 on each side) compact schemes. The derivative coefficients (DER1_3, DER1_2, DER1_1, etc.) are carefully chosen for 7th-order accuracy. The grid spacing DSX = 0.04908738521 ≈ 2π/128, suggesting these are tuned for periodic domains with a wavelength of approximately 128 cells.

8. **Smagorinsky SGS model**: The SGS exercises implement the classic Smagorinsky model with:
   - Smagorinsky constant: 1.16 (higher than the typical 0.1–0.2 for standard LES, likely because the grid is very coarse relative to physical scales in the workshop setup).
   - Characteristic scale: `DSX` (grid spacing).
   - Galilean-invariant strain-rate magnitude estimation.

9. **Annular ring forcing**: The `forcing()` kernel applies a tangential force in an annular region (0.4 < dist < 0.6) in the XY plane. The force direction is `(sy, -sx, 0) / dist` — a vortex centered at the origin. This is a standard test case for decay of isotropic turbulence.

10. **7×7×7 box filter**: The smoothing filter in `hydro-sgs-smoothed/filter.ach` is generated programmatically by `generate-smoothing-filter.py`. It applies a uniform 343-point average — essentially a very aggressive low-pass filter. At `1/343 ≈ 0.0029`, this smooths the velocity field significantly each timestep.

11. **Model examples are copies, not symlinks**: The `model-examples/` subdirectories contain complete copies of the exercise solutions, not references to the exercise directories. Each `model-examples/*/CMakeLists.txt` sets `DSL_MODULE_DIR` to the corresponding exercise directory, so they share the same DSL sources but have independent host code.

12. **Blur demo vs. blur exercise**: The `blur-demo/` directory contains the completed solution, while `blur/` contains the student exercise version. They are structurally identical (same `blur.c` and `blur.ac` content), but the `blur/` version has comments replacing the key lines that students must fill in. This pattern allows the workshop instructor to provide a working reference without accidentally giving away the solution.

13. **Snapshot filename convention**: `acHostMeshWriteToFile(mesh, i)` writes files with the step number in the filename. The `animate-snapshots.py` script reads these as binary data, reshapes them using a `data-format.csv` header file, and creates an animated visualization.

14. **Hydro exercise continuity placeholder**: The `continuity()` function in `hydro/hydro.ac` returns `1.0` — a placeholder. The actual physics is implemented in `hydro-sgs/hydro-sgs.ac` as `-dot(vecvalue(UU), gradient(LNRHO)) - divergence(UU)`. This is the convective continuity equation for variable-density flow.

15. **MHD solver incomplete**: The `mhd-les/` exercise is the most incomplete — the `momentum()` function returns `real3(0, 0, 0)` with an explicit TODO comment, and `compute_adv()` is entirely commented out. This suggests the MHD exercise is intended as a significant challenge for advanced participants.

16. **Field3 macro**: Both the hydro and mhd exercises define `#define UU Field3(UUX, UUY, UUZ)` and `#define B Field3(BX, BY, BZ)`. The `Field3` construct allows vector fields to be used with `real3` operators (`.x`, `.y`, `.z`) and `vecvalue()`.

17. **Stress tensor symmetry**: The `T01 = T10`, `T02 = T20`, `T12 = T21` symmetry of the SGS stress tensor is noted in the code comment (`// Note: the stress tensor is symmetric`) but only 6 of the 9 components are stored/used.

18. **No MPI**: All exercises run on a single GPU (`acDeviceCreate(0, ...)`). There is no MPI initialization, no domain decomposition, and no parallel communication. This is appropriate for a workshop but limits the maximum grid size to what fits in a single GPU's memory.

19. **Fixed random initial conditions**: `acHostMeshRandomize(&mesh)` is called once at the start of each exercise, so results are non-deterministic across runs (depends on the random seed). For reproducibility in workshops, a fixed seed would be preferred.

20. **Python visualization dependency**: `animate-snapshots.py` requires `python3`, `numpy`, and `matplotlib`. It reads a `data-format.csv` file (generated by Astaroth) to determine precision and grid dimensions, then creates a matplotlib `ArtistAnimation` from the X-Y slices at Z = mz/2. The script also contains two alternative visualization approaches in commented-out blocks.
