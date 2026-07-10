# ⚠️ WARNING ⚠️
# **This is a LLM based analysis.**
# **THE BRANCH SHOULD NEVER BE MERGED TO master OR develop BRANCHES!**

# LLM Information
- **Model:** Qwen3.6-35B-A3B

# Overview
The `les` sample is a Large Eddy Simulation (LES) of turbulent fluid flow using Astaroth's DSL (Domain-Specific Language). It implements a compressible Navier-Stokes solver with Smagorinsky subgrid-scale turbulence model, 6th-order compact finite-difference stencils, and a 3-substep strong-stability-preserving RK3 time integrator. The simulation runs 2499 time steps on a 32³ periodic grid, computing stress tensors, momentum advection, viscous diffusion, and continuity evolution. It includes two physics variants (`main.ac` for incompressible-style and `main.ac.compressible` for fully compressible) and a Python visualization tool for output data slices.

# Directory Structure & File Descriptions

| File | Description |
| :--- | :--- |
| `CMakeLists.txt` | Build configuration; compiles `main.c` into the `les` executable, linked against `astaroth_core` and `astaroth_utils`. |
| `main.c` | C driver: loads mesh config, creates device, runs verification tests (load/store, read/write, boundary conditions), performs warmup, profiles with nvprof, then runs the full 2499-step LES integration with stress tensor and RK3 substeps. Saves slices to disk every 25 steps. |
| `main.ac` | Primary Astaroth DSL definition: declares fields (UUX, UUY, UUZ, LNRHO, T00–T22), physical constants, subgrid-scale model, stress tensor computation, momentum equation, continuity equation, and two kernels (`compute_stress_tensor_tau`, `singlepass_solve`). |
| `main.ac.compressible` | Compressible variant: includes `UNIVERSAL_GAS_CONSTANT_R`, modifies pressure/continuity equations for fully compressible flow. Note: commented-out pressure term causes the simulation to explode per the TODO. |
| `stencil.ach` | Astaroth compiler header: defines compile-time constants (`AC_dt`, `AC_step_number`, grid spacings), 6th-order compact finite-difference stencils (ddx, ddy, ddz, ddxx, ddyy, ddzz, ddxy, ddxz, ddyz), vector calculus operators (gradient, divergence, curl, laplace), and RK3 time integration helpers. |
| `les-stencilgenerator.py` | Script that generates a `STENCIL_ORDER` definition and a 3×3×3 sum stencil (for halo exchange, currently unused). |
| `analysis.py` | Python visualization: reads `data-format.csv` header and binary field data files, reshapes to 3D, extracts the mid-Z slice, and displays as an animated matplotlib plot. |
| `build.sh` | Build wrapper with hardcoded path to a local cmake binary; sets `PROGRAM_MODULE_DIR` and `DSL_MODULE_DIR` to `samples/les`. |
| `README.md` | Usage notes: cmake command with custom module dirs, and NVIDIA Nsight Compute profiling command. |

# Physical Model

## Fields

| Field | Type | Description |
| :--- | :--- | :--- |
| `UUX` | scalar | x-velocity component (log density?) |
| `UUY` | scalar | y-velocity component |
| `UUZ` | scalar | z-velocity component |
| `LNRHO` | scalar | Log density |
| `T00` | scalar | Stress tensor component σ_xx |
| `T01` | scalar | Stress tensor component σ_xy (= σ_yx) |
| `T02` | scalar | Stress tensor component σ_xz (= σ_zx) |
| `T11` | scalar | Stress tensor component σ_yy |
| `T12` | scalar | Stress tensor component σ_yz (= σ_zy) |
| `T22` | scalar | Stress tensor component σ_zz |

Note: `UU = Field3(UUX, UUY, UUZ)` — velocity vector type alias.

## Governing Equations

### Momentum (incompressible-style, `main.ac`)

```
momentum() = -∇U · U - (1/ρ) · p(∇ρ) + ν(ρ) · ∇²U - ∇·σ
```

Where:
- **Advection**: `-gradients(UU) * vecvalue(UU)` — convective term
- **Pressure**: `-(1/value(LNRHO)) * pressure(gradient(LNRHO))` — pressure gradient with `p = ρ · (γ-1) · CV · T`
- **Viscosity**: `kinematic_viscosity() * veclaplace(UU)` — viscous diffusion with ν = μ/ρ
- **Stress**: `construct_stress_term()` — divergence of subgrid-scale stress tensor

### Continuity (incompressible-style)

```
continuity() = -U · ∇(ln ρ)
```

### Compressible variant (`main.ac.compressible`)

```
momentum() = -∇U · U - c_s²·∇(ln ρ) + ν·(∇²U + (1/3)·∇(∇·U) + 2S·∇(ln ρ)) + ζ·∇(∇·U)
continuity() = -U · ∇(ln ρ) - ∇·U
```

## Subgrid-Scale Model (Smagorinsky)

```
S_ij = (2/3)(∂u_i/∂x_j - (1/3)·δ_ij·∂u_k/∂x_k)  — deviatoric strain rate
|S| = sqrt(2 · S_ij · S_ij)                        — Galilean invariant
ν_t = (C_s · Δ)² · |S|                              — eddy viscosity
σ_t = -2 · ν_t · S                                  — subgrid stress
```

Constants:
| Constant | Value | Description |
| :--- | :--- | :--- |
| `TEMPERATURE_T` | 288 | Temperature (K), 15°C |
| `DYNAMIC_VISCOSITY_MU` | 5e-3 | Dynamic viscosity (Pa·s) |
| `SMAGORINSKY_COEFFICIENT_C` | 0.16 | Smagorinsky constant C_s |
| `CHARACTERISTIC_SCALE` | DSX | Grid spacing Δ |
| `CP` | 1.0 | Specific heat at constant pressure |
| `CV` | 2.0 | Specific heat at constant volume |
| `GAMMA` | 0.5 | γ = Cp/Cv (note: < 1, unusual) |

For the compressible variant, additional constants:
| Constant | Value | Description |
| :--- | :--- | :--- |
| `UNIVERSAL_GAS_CONSTANT_R` | 8.314... | Ideal gas constant |
| `CS0` | 1.0 | Sound speed parameter |
| `RHO0` | 1.0 | Reference density |
| `ZETA` | 1e-2 | Bulk viscosity coefficient |
| `NU_VISC` | 5e-3 | Kinematic viscosity |

## RK3 Time Integration

Strong-stability-preserving Runge-Kutta 3rd order with 3 substeps:

| Substep | α (stage) | β (stage) |
| :--- | :--- | :--- |
| 0 | 0 | 1/3 |
| 1 | -5/9 | 15/16 |
| 2 | -153/128 | 8/15 |

```
rk3(s0, s1, roc) = s1 + β[n] * (α[n-1]/β[n-1] * (s1 - s0) + roc * dt)
```

Where `s0` = previous solution, `s1` = current solution, `roc` = rate of change.

Note: The original conditional implementation (`if AC_step_number > 0`) had "abysmal performance on AMD" — replaced with a workaround using array indexing.

# Finite-Difference Stencils (`stencil.ach`)

## Grid Spacing

| Constant | Value |
| :--- | :--- |
| `DSX = DSY = DSZ` | 0.04908738521 (= 2π/128, for periodic domain [0, 2π]³) |

## First Derivatives (6th-order compact, radius 3)

| Stencil | Order | Coefficients | Formula |
| :--- | :--- | :--- | :--- |
| `ddx`, `ddy`, `ddz` | 6 | `DER1_0=0, DER1_1=3/4, DER1_2=-3/20, DER1_3=1/60` | Central difference: ±3/4·f₁ ∓ 3/20·f₂ ± 1/60·f₃ |

## Second Derivatives (6th-order compact, radius 3)

| Stencil | Order | Coefficients | Formula |
| :--- | :--- | :--- | :--- |
| `ddxx`, `ddyy`, `ddzz` | 6 | `DER2_0=-49/18, DER2_1=3/2, DER2_2=-3/20, DER2_3=1/90` | Central difference |

## Mixed Derivatives (6th-order compact, radius 3)

| Stencil | Order | Coefficients | Formula |
| :--- | :--- | :--- | :--- |
| `ddxy`, `ddxz`, `ddyz` | 6 | `DERX_0=0, DERX_1=270/720, DERX_2=-27/720, DERX_3=2/720` | Mixed central difference |

## Vector Operators (built from stencils)

| Operator | Description |
| :--- | :--- |
| `value(v)` | Scalar field value (identity stencil) |
| `gradient(s)` | ∇s = (ddx(s), ddy(s), ddz(s)) |
| `gradients(v)` | ∇v = Matrix(gradient(v.x), gradient(v.y), gradient(v.z)) |
| `divergence(v)` | ∇·v = ddx(v.x) + ddy(v.y) + ddz(v.z) |
| `curl(v)` | ∇×v = (ddy(v.z)-ddz(v.y), ddz(v.x)-ddx(v.z), ddx(v.y)-ddy(v.x)) |
| `laplace(s)` | ∇²s = ddxx(s) + ddyy(s) + ddzz(s) |
| `veclaplace(v)` | ∇²v = (laplace(v.x), laplace(v.y), laplace(v.z)) |
| `gradient_of_divergence(v)` | ∇(∇·v) using mixed derivatives |
| `vecvalue(v)` | (value(v.x), value(v.y), value(v.z)) |
| `contract(mat)` | Frobenius norm: √(Σᵢⱼ mᵢⱼ²) |

# Program Flow (main.c)

1. **Device availability check**: `acCheckDeviceAvailability() == AC_SUCCESS`.
2. **Mesh setup**: 32×32×32 grid from `AC_DEFAULT_CONFIG`.
3. **Mesh allocation**: Create `mesh` (randomized with periodic BCs), `tmp_mesh`, `candidate`.
4. **Device creation**: Create device, load mesh, set `AC_dt = 1e-3`, `AC_step_number = 2`.
5. **Verification suite**:
   - **Load/Store**: randomize → load → store → `acVerifyMesh("Load/Store", ...)`.
   - **Read/Write**: write `candidate` to file → read back → `acVerifyMesh("Read/Write", ...)`.
   - **Boundary conditions**: randomize → load → apply periodic BCs on device → store → `acVerifyMesh("Boundconds", ...)`.
6. **Warmup**: Launch all `NUM_KERNELS` kernels once (no timing).
7. **Profiling run**: Start `cudaProfilerStart()`, launch all kernels, stop profiler.
8. **Main simulation** (2499 steps, step 1–2499):
   - Set `AC_dt = 1e-2`, initialize `LNRHO = 1.0`.
   - For each of 3 substeps:
     - Load `AC_step_number = substep`.
     - Launch `compute_stress_tensor_tau()` → swap T00, T01, T02, T11, T12, T22 buffers → periodic BCs.
     - Launch `singlepass_solve()` → swap UUX, UUY, UUZ, LNRHO buffers → periodic BCs.
   - Save slice to disk every 25 steps.
   - Print step number and field min/max ranges.
   - Check for NaN values → exit on failure.
9. **Print final field ranges** and cleanup.

# Astaroth DSL Kernels

| Kernel | Description |
| :--- | :--- |
| `compute_stress_tensor_tau()` | Computes 6 components of the subgrid-scale stress tensor σ from the velocity field UU. Uses `stress_tau()` → `stress_tensor(UU)` → `smagorinsky_eddy_viscosity(stress)`. |
| `singlepass_solve()` | One RK3 time step for momentum and continuity. Computes `momentum()`, applies `rk3(previous(UUX), value(UUX), mom.x)` for each velocity component, and `rk3(previous(LNRHO), value(LNRHO), continuity())` for density. |

# Key Astaroth C APIs Used

| Function | Description |
| :--- | :--- |
| `acCheckDeviceAvailability()` | Check GPU availability; returns `AC_SUCCESS` or error. |
| `acLoadConfig(config, &info)` | Load mesh configuration. |
| `acSetMeshDims(nn, nn, nn, &info)` | Set 32³ grid dimensions. |
| `acPrintMeshInfo(info)` | Print mesh configuration. |
| `acConstructInt3Param(x, y, z, info)` | Construct int3 from mesh info params. |
| `acHostMeshCreate(info, &mesh)` | Create host-side mesh. |
| `acHostMeshRandomize(&mesh)` | Randomize mesh field values. |
| `acHostMeshApplyPeriodicBounds(&mesh)` | Apply periodic boundary conditions on host. |
| `acHostMeshWriteToFile(mesh, id)` | Write mesh to binary file `<id>`. |
| `acHostMeshReadFromFile(id, &mesh)` | Read mesh from binary file `<id>`. |
| `acHostMeshDestroy(&mesh)` | Free host mesh. |
| `acHostVertexBufferSet(handle, val, &mesh)` | Set vertex buffer values (e.g., LNRHO = 1.0). |
| `acDeviceCreate(id, info, &device)` | Create GPU device context. |
| `acDevicePrintInfo(device)` | Print device information. |
| `acDeviceDestroy(device)` | Destroy device. |
| `acDeviceLoadMesh(stream, mesh)` | Upload mesh to device. |
| `acDeviceStoreMesh(stream, &mesh)` | Download mesh from device. |
| `acDeviceLoadScalarUniform(stream, param, val)` | Load scalar uniform (e.g., AC_dt, AC_step_number). |
| `acDeviceLoadIntUniform(stream, param, val)` | Load integer uniform. |
| `acDeviceLaunchKernel(stream, kernel, n0, n1)` | Launch a GPU kernel. |
| `acDeviceSwapBuffers(device)` | Swap all input/output buffer pairs. |
| `acDeviceSwapBuffer(device, handle)` | Swap a specific buffer handle. |
| `acDevicePeriodicBoundconds(stream, m0, m1)` | Apply periodic boundary conditions on device. |
| `acDeviceSynchronizeStream(stream, all)` | Synchronize stream(s). |
| `acDeviceReduceScal(stream, type, i, &val)` | Reduce scalar field (RTYPE_MIN/RTYPE_MAX). |
| `acVerifyMesh(name, original, candidate)` | Compare two meshes for equality. |

# Notable Observations

1. **Dual-physics variants**: `main.ac` (incompressible-style with constant temperature) and `main.ac.compressible` (fully compressible with ideal gas law). The compressible variant has a known issue: the pressure term is commented out because it causes the simulation to "explode."

2. **6th-order compact finite differences**: Uses radius-3 stencils for all derivatives, providing high accuracy on the relatively coarse 32³ grid. The stencil coefficients follow standard 6th-order central-difference formulas.

3. **Smagorinsky LES model**: Implements the classic Smagorinsky subgrid-scale model with C_s = 0.16 (standard value for homogeneous turbulence). The stress tensor computation uses the Galilean-invariant strain rate magnitude.

4. **RK3 with AMD workaround**: The `rk3()` function has a commented-out conditional implementation that "has abysmal performance on AMD" — replaced with array-indexing workaround. The `rk3_intermediate()` function has the same AMD issue with a TODO note.

5. **Stress tensor computation is separate from solve**: The `compute_stress_tensor_tau` kernel runs first each substep, then `singlepass_solve` uses the pre-computed stress. This two-kernel approach per substep adds overhead but keeps the DSL kernels simpler.

6. **Buffer swapping after each kernel**: After `compute_stress_tensor_tau`, 6 stress tensor buffers are individually swapped. After `singlepass_solve`, 4 velocity/density buffers are swapped. This manual buffer management is required by Astaroth's double-buffered kernel execution model.

7. **Periodic boundary conditions applied twice per substep**: Once after stress tensor computation and once after the solve step. This may be redundant since neither kernel should affect boundary data, but ensures consistency.

8. **NaN detection**: Each step checks for NaN in all field ranges — critical for detecting numerical instability in turbulent simulations.

9. **Profiling integration**: The sample integrates `cudaProfilerStart()`/`cudaProfilerStop()` for Nsight Compute profiling, with the README providing an exact `ncu` command line.

10. **Build requires custom module dirs**: The `build.sh` sets both `PROGRAM_MODULE_DIR` and `DSL_MODULE_DIR` to `samples/les`, and disables all other samples (`-DBUILD_SAMPLES=OFF`). This is because the Astaroth DSL compiler needs to find `main.ac` and `stencil.ach` during compilation.

11. **No MPI**: This is a single-GPU simulation on a 32³ grid — small enough to fit on one device without domain decomposition.

12. **User-defined includes**: `#include "user_defines.inc"` is expected from the DSL compiler output — this file is generated during the build process when Astaroth compiles the `.ac` files.

13. **Simulation parameters**: `dt = 1e-2` for the main simulation, `dt = 1e-3` for setup steps. Grid spacing `DSX = 2π/128 ≈ 0.049` despite the 32³ grid, suggesting the physical domain is normalized to 128 equivalent points.

14. **Output frequency**: Slices saved every 25 steps over 2499 steps = ~100 frames, suitable for the matplotlib animation viewer in `analysis.py`.
