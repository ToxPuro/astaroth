# ⚠️ WARNING ⚠️
# **This is a LLM based analysis.**
# **THE BRANCH SHOULD NEVER BE MERGED TO master OR develop BRANCHES!**

# LLM Information
- **Model:** Qwen3.6-35B-A3B

# Overview
The `stress` directory provides an MPI-based 2D elastic wave stress simulation driven by the Astaroth Domain-Specific Language (DSL). It simulates dynamic stress propagation in an elastic solid with E = 200 GPa steel-like material (Poisson's ratio ν = 1/3) on a 168×168 grid (doubled to 84×84 via `npoints` constants, then ×2 = 168). The program uses a custom 5-kernel pipeline — initial condition, displacement calculation, stress computation (Hooke's law with full 2D elasticity tensor), acceleration computation (momentum balance with damping), and velocity smoothing with coordinate update — all executed via the DSL task graph execution system. A 5-point smoothing stencil is applied to velocities to control numerical instability. The simulation writes 1000 timesteps of results to `stress11.dat` containing x-coordinates, duplicate x-coordinates, and STRESS11 values. It runs as an MPI program with `mpirun`.

# Directory Structure & File Descriptions

| File | Description |
| :--- | :--- |
| `CMakeLists.txt` | Build config: creates `stress` executable from `main.cpp`, links `astaroth_core` and `astaroth_utils`. |
| `main.cpp` | MPI-based orchestration. Initializes MPI, loads config, overrides grid spacing, sets mesh dimensions to 168×168×1, initializes GPU, retrieves DSL task graphs for `AC_initialize` and `AC_update`, executes initialization once, then runs `AC_update` for `nsteps` iterations, and writes `stress11.dat`. |
| `DSL/solver.ac` | DSL (Astaroth DSL) solver definition. Contains material constants (E = 200 GPa, ν = 1/3), field declarations, 5 kernels (initial_condition, calc_displacement, calculate_stresses, calculate_acceleration, smooth_velocities_and_update_coordinates), boundary conditions (periodic on XY), and compute step definitions for AC_initialize and AC_update. |

# Compile-Time Requirements

| Setting | Value | Description |
| :--- | :--- | :--- |
| MPI | `REQUIRED` | Program requires MPI; falls back to error message if built without MPI support. |
| `AC_MPI_ENABLED` | Config-dependent | If disabled, program prints error and exits. |
| `HOST_MODEL` | Config-dependent | Controls whether CPU reference model is compiled. |

Compile options: Inherited from `astaroth_core` (typically `-Wall -Wextra -Werror -Wdouble-promotion -Wfloat-conversion -Wshadow`).

# Compile-Time Options

| Macro | Default | Description |
| :--- | :--- | :--- |
| `AC_MPI_ENABLED` | Config-dependent | If disabled, program prints error and exits. |
| `NGHOST` | Config-dependent | Number of ghost/halo cells (used in output loop bounds). |
| `STENCIL_ORDER` | `2` (in DSL) | Set in `solver.ac` line 87 via `hostdefine`. |

# Input Parameters / Command-Line Interface

No command-line arguments. The program uses `AC_DEFAULT_CONFIG` via `acLoadConfig()` and overrides grid parameters from `user_constants.inc` / `math_utils.h` includes. The DSL solver (`solver.ac`) defines its own constants: `Lengthscale = 324`, `npoints = 84`, `npointsx_grid = npoints`, `npointsy_grid = npoints`, `dsx = Lengthscale/npointsx_grid`, `dsy = Lengthscale/npointsy_grid`, `nsteps = 1000`.

Usage: `mpirun -np <num_processes> ./stress`

# Program Flow

## 1. MPI Initialization
- `MPI_Init(NULL, NULL)`
- `MPI_Comm_size()` / `MPI_Comm_rank()` — get process count and rank.
- `atexit(acAbort)` — register abort handler for cleanup on exit.
- `srand(321654987)` — fixed random seed (matches `stencil-loader`).
- `acAbort()` — custom abort function that calls `MPI_Abort()` if MPI is not yet finalized.

## 2. Configuration & Grid Override
- `acLoadConfig(AC_DEFAULT_CONFIG, &info)` — load default configuration.
- `info[AC_ds] = {dsx, dsy, 0}` — override grid spacing from `user_constants.inc`.
- `acSetMeshDims(npointsx_grid, npointsy_grid, 1, &info)` — set mesh to 84×84×1 (then doubled to 168×168×1).

## 3. Host Mesh Allocation (rank 0 only)
- `acHostMeshCreate(info, &model)` — allocate CPU reference mesh.
- `acHostMeshCreate(info, &candidate)` — allocate GPU result buffer.
- `acHostMeshRandomize(&model)` — randomize model mesh.
- `acHostMeshRandomize(&candidate)` — randomize candidate buffer.

## 4. GPU Initialization
- `acGridInit(info)` — initialize GPU subsystem.

## 5. Test Phase 1: Load/Store Verification
- `acGridLoadMesh(STREAM_DEFAULT, model)` — upload random mesh to GPU.
- `acGridStoreMesh(STREAM_DEFAULT, &candidate)` — store GPU result to host.
- Rank 0: `acVerifyMesh("Load/Store", model, candidate)` — compare GPU vs CPU.
- **Note:** Only executed if `res != AC_SUCCESS` triggers `WARNCHK_ALWAYS`.

## 6. DSL Task Graph Retrieval
- `acGetDSLTaskGraph(AC_initialize)` — retrieve initialization task graph.
- `acGetDSLTaskGraph(AC_update)` — retrieve update task graph.
- All vertex buffer handles are collected into `all_fields` vector (not used further).

## 7. Initialization Execution
- `acGridExecuteTaskGraph(initialize, 1)` — execute AC_initialize once (runs `initial_condition()` kernel).
- `acGridSynchronizeStream(STREAM_ALL)` — wait for completion.
- `acGridStoreMesh(STREAM_DEFAULT, &candidate)` — store results to host.
- `acHostMeshApplyPeriodicBounds(&candidate)` — apply periodic BCs on host copy.

## 8. Time Integration Loop (1000 steps)
```
for (int i = 0; i < nsteps; ++i)  // nsteps = 1000 from DSL
{
    acGridExecuteTaskGraph(update, 1);   // execute AC_update (4 kernels)
    acGridSynchronizeStream(STREAM_ALL); // wait for completion
}
```

Each AC_update iteration executes 4 kernels in sequence:
1. `calc_displacement()` — compute UX = COORDS.x - ORIGINAL_COORDS.x, UY = COORDS.y - ORIGINAL_COORDS.y
2. `calculate_stresses()` — compute strains and stresses via Hooke's law (2D elasticity)
3. `calculate_acceleration()` — compute acceleration from stress divergence, apply damping, update velocity
4. `smooth_velocities_and_update_coordinates()` — smooth velocities, update coordinates with second-order integration

## 9. Output (`stress11.dat`)
- Open `stress11.dat` for writing (rank 0, after loop).
- Loop over `x = NGHOST` to `npointsx_grid - NGHOST` and `y = NGHOST` to `npointsy_grid - NGHOST`.
- Write 3 values per line: `COORDS_X` (x), `COORDS_X` (duplicate x), `STRESS11`.
- Format: `%.7e` (7 decimal places, scientific notation).
- Close file.

## 10. Cleanup
- Rank 0: `acHostMeshDestroy(&model)`, `acHostMeshDestroy(&candidate)`.
- `acGridQuit()` — shutdown GPU.
- `ac_MPI_Finalize()` — shutdown MPI.
- `finalized = true` — flag to suppress `acAbort()`.

# DSL Solver Definition (DSL/solver.ac)

## Material Parameters

| Parameter | Value | Description |
| :--- | :--- | :--- |
| `Eg` | `200 × 10⁹` (= 200 GPa) | Young's modulus (steel-like). |
| `vg` | `1/3` | Poisson's ratio. |
| `ro` | `7800` | Density (kg/m³, steel). |
| `Lengthscale` | `324` (= 162×2) | Physical length scale. |
| `npoints` | `168` (= 84×2) | Number of grid points per dimension. |
| `nsteps` | `1000` | Number of timesteps. |
| `SimTime` | `75.0` (= 3×25) | Simulated time (not used in main.cpp loop). |
| `matrix_coeff` | `Eg / ((1+vg)(1-2vg))` | Elasticity tensor scaling factor. |

## Elasticity Tensor (2D Plane Stress/Strain)

The stress-strain relationship uses a 4th-order elasticity tensor expressed as 3×3 matrices:

| Matrix | Diagonal | Off-diagonal | Description |
| :--- | :--- | :--- | :--- |
| `cg11` | `{1-ν, 0, 0}` | — | Normal stiffness in x. |
| `cg22` | `{ν, 1-ν, 0}` | — | Normal stiffness in y. |
| `cg33` | `{ν, ν, 1-ν}` | — | Out-of-plane stiffness. |
| `cg12` | `{0, 0.5(1-2ν), 0}` | — | Coupling x-y shear. |
| `cg21` | `{0.5(1-2ν), 0, 0}` | — | Coupling y-x shear. |
| `cg13` | `{0, 0, 0.5(1-2ν)}` | — | Coupling x-z shear. |
| `cg31` | `{0.5(1-2ν), 0, 0}` | — | Coupling z-x shear. |
| `cg23` | `{0, 0, 0.5(1-2ν)}` | — | Coupling y-z shear. |
| `cg32` | `{0, 0.5(1-2ν), 0}` | — | Coupling z-y shear. |

All multiplied by `matrix_coeff = Eg / ((1+ν)(1-2ν))`.

## Pre-computed Matrix Coefficient Values

| Constant | Value |
| :--- | :--- |
| `cg_diagonal` | `1 - ν = 2/3` |
| `cg_off_diagonal` | `ν = 1/3` |
| `cg_non_main` | `0.5 × (1 - 2ν) = 1/6` |
| `matrix_coeff` | `200×10⁹ / ((4/3)(1/3)) = 200×10⁹ / (4/9) = 450×10⁹` |

## Fields Declared

| Field | Type | Description |
| :--- | :--- | :--- |
| `PHI` | `Field` | Phase field (initiation zone indicator). |
| `ORIGINAL_COORDS` | `Field2` | Original (Lagrangian) coordinates. |
| `UX, UY` | `Field` | Displacement components. |
| `STRESS11, STRESS12, STRESS21, STRESS22` | `Field` | Stress tensor components. |
| `VELOCITY_X, VELOCITY_Y` | `Field` | Velocity components. |
| `ACCELERATION_X, ACCELERATION_Y` | `Field` | Acceleration components. |
| `U` | `Field` | Displacement magnitude (unused). |
| `SOLUTION` | `Field` | Analytical solution buffer (unused). |
| `AC_COORDS` | `Field2` (built-in) | Current (Eulerian) coordinates. |

## Kernels

### initial_condition()
- Computes grid coordinates centered at origin: `x = dsx × (i - (ngrid.x+1)/2)`.
- Writes coordinates to `AC_COORDS` and `ORIGINAL_COORDS`.
- Creates a circular phase field `PHI` centered at grid center:
  - `initial_phi_size = 0.5 × 10⁻⁷`
  - `transitlength = 5 × dsx × 10⁻⁸`
  - `magic_coeff = 0.51 × dsx × 10⁻⁸`
  - If inside `initial_phi_size + magic_coeff`: `PHI = 1.0`
  - If in transition zone: `PHI = 0.5 × (1 + cos(π × (r - phi_size) / transitlength))`
  - Below `10⁻³⁰⁸`: `PHI = 0.0`
- `VELOCITY_X = VELOCITY_Y = 0.0`

### calc_displacement()
- `UX = AC_COORDS.x - ORIGINAL_COORDS.x` (x displacement).
- `UY = AC_COORDS.y - ORIGINAL_COORDS.y` (y displacement).

### calculate_stresses()
- Computes strain via `get_first_order_derivatives(ORIGINAL_COORDS.x, ORIGINAL_COORDS.y, UX)` and same for UY.
- `strainxx = ∂UX/∂x - PHI × TrStr[0][0]`
- `strainxy = 0.5 × (∂UY/∂x + ∂UX/∂y) - PHI × TrStr[0][1]`
- `strainyx = 0.5 × (∂UX/∂y + ∂UY/∂x) - PHI × TrStr[1][0]`
- `strainyy = ∂UY/∂y - PHI × TrStr[1][1]`
- Applies Hooke's law:
  - `stress11 = cg11[0][0] × strainxx + cg11[1][1] × strainyy`
  - `stress12 = cg12[0][1] × strainxy`
  - `stress21 = cg21[1][0] × strainyx`
  - `stress22 = cg22[1][1] × strainyy + cg22[0][0] × strainxx`
- `TrStr` is a small perturbation matrix `{{0.01, 0, 0}, {0, 0.01, 0}, {0, 0, 0.01}}`.

### calculate_acceleration()
- Computes stress gradients: `get_first_order_derivatives(AC_COORDS.x, AC_COORDS.y, STRESS11)` etc.
- `ax = (∂STRESS11/∂x + ∂STRESS12/∂y) / ρ - VELOCITY_X × damping_coeff / ρ`
- `ay = (∂STRESS21/∂x + ∂STRESS22/∂y) / ρ - VELOCITY_Y × damping_coeff / ρ`
- `damping_coeff = 10¹⁵` — very strong velocity damping.
- `eldt = 2 × 10⁻¹²` — extremely small effective timestep.
- `VELOCITY_X += ax × eldt`, `VELOCITY_Y += ay × eldt`.
- `ACCELERATION_X = ax`, `ACCELERATION_Y = ay`.

### smooth_velocities_and_update_coordinates()
- Applies 5-point smoothing stencil to velocities: `smooth(VELOCITY_X)`, `smooth(VELOCITY_Y)`.
- Second-order coordinate update: `AC_COORDS.x += eldt × smoothed_vel_x + 0.5 × eldt² × ACCELERATION_X`.
- `VELOCITY_X = smoothed_vel_x` (replace with smoothed).

## Smoothing Stencil

| Offset | Value |
| :--- | :--- |
| `[0][0][1]` | `0.20` |
| `[0][0][-1]` | `0.20` |
| `[0][0][0]` | `0.20` |
| `[0][1][0]` | `0.20` |
| `[0][-1][0]` | `0.20` |

5-point cross stencil: each neighbor contributes 0.20 (sum = 1.0, conservative).

## Boundary Conditions

```
BoundConds bcs {
    periodic(BOUNDARY_XY)
}
```

Periodic boundary conditions on both X and Y axes. Z is 1D (no boundaries).

## Compute Steps

| Step | Kernels |
| :--- | :--- |
| `AC_initialize(bcs)` | `initial_condition()` |
| `AC_update(bcs)` | `calc_displacement()`, `calculate_stresses()`, `calculate_acceleration()`, `smooth_velocities_and_update_coordinates()` |

# GPU/DSL API Functions Used

| Function | Description |
| :--- | :--- |
| `acGridInit(info)` | Initialize GPU subsystem. |
| `acGridQuit()` | Shutdown GPU subsystem. |
| `acGridLoadMesh(stream, mesh)` | Transfer mesh from host to GPU. |
| `acGridStoreMesh(stream, &candidate)` | Transfer mesh from GPU to host. |
| `acGridExecuteTaskGraph(graph, n)` | Execute a DSL task graph n iterations. |
| `acGetDSLTaskGraph(graph_type)` | Retrieve task graph for DSL compute step. |
| `acGridSynchronizeStream(stream)` | Wait for stream completion. |
| `acGridMPIComm()` | Get MPI communicator handle. |
| `ac_MPI_Finalize()` | Finalize MPI. |

# Host API Functions Used

| Function | Description |
| :--- | :--- |
| `acLoadConfig(path, &info)` | Load configuration. |
| `acSetMeshDims(nx, ny, nz, &info)` | Set mesh dimensions. |
| `acHostMeshCreate(info, &mesh)` | Allocate CPU-side `AcMesh`. |
| `acHostMeshDestroy(&mesh)` | Free CPU mesh. |
| `acHostMeshRandomize(&mesh)` | Fill with pseudo-random values. |
| `acHostMeshApplyPeriodicBounds(&mesh)` | Apply periodic BCs on CPU. |
| `acVerifyMesh(name, model, candidate)` | Compare two meshes, return `AcResult`. |
| `acAbort()` | Custom abort via `MPI_Abort()`. |

# Input/Output DSL Parameters

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `AC_current_time` | `input real` | Current simulation time. |
| `AC_dt` | `input real` | Timestep size. |
| `AC_step_num` | `input int` | Current step number. |
| `AC_ds.x/y/z` | `const real` | Grid spacing (from DSL or config). |
| `AC_ngrid.x/y/z` | `const int` | Grid dimensions. |
| `AC_REAL_PI` | `const real` | Pi constant. |

# Output Data Format (`stress11.dat`)

| Column | Format | Description |
| :--- | :--- | :--- |
| 1 | `%.7e` | X coordinate (from `AC_COORDS.x`). |
| 2 | `%.7e` | X coordinate (duplicate of column 1 — appears to be a copy/paste error in line 147). |
| 3 | `%.7e` | STRESS11 stress component value. |

Grid loop: `x ∈ [NGHOST, npointsx_grid - NGHOST)`, `y ∈ [NGHOST, npointsy_grid - NGHOST)`.

# Notable Observations

1. **2D elastic wave simulation:** This is a solid mechanics simulation (not fluid dynamics), modeling stress wave propagation in a 2D steel-like material (E = 200 GPa, ρ = 7800 kg/m³, ν = 1/3). It uses Hooke's law with a full 2D elasticity tensor.

2. **DSL-driven architecture:** Unlike `stencil-loader` which uses direct GPU API calls (`acGridIntegrate`), this program uses the DSL task graph system (`acGetDSLTaskGraph` + `acGridExecuteTaskGraph`). The solver logic is entirely in `DSL/solver.ac`, making it a DSL-centric program.

3. **Two-phase time integration:** The acceleration step uses Euler update for velocity (`v += a × eldt`) but a second-order Verlet-like update for coordinates (`x += eldt × v + 0.5 × eldt² × a`), which is more accurate for position but inconsistent with the velocity update.

4. **Extremely small effective timestep:** `eldt = 2 × 10⁻¹²` — this is orders of magnitude smaller than the CFL-limited timestep used in fluid solvers. With `dsx = 324/168 ≈ 1.93` meters, the wave speed for steel is `c = √(E/ρ) ≈ 4740 m/s`, giving a CFL timestep of roughly `ds/c ≈ 0.0004 s`. The `eldt` here is 15 orders of magnitude smaller, suggesting this is meant to simulate a very fast transient event or is a debugging/scaled parameter set.

5. **Massive damping:** `damping_coeff = 10¹⁵` is enormous. The damping term `VELOCITY × damping_coeff / ρ` would dominate the acceleration equation unless velocities are already extremely small. This suggests the simulation is designed to quickly dissipate kinetic energy, possibly to study the initial stress wave in isolation.

6. **Output contains a duplicate column:** Line 147 writes `COORDS_X` twice: `candidate.vertex_buffer[AC_COORDS_X][...]` appears for both the first and second columns. The second column should likely be `AC_COORDS_Y`.

7. **Hardcoded output filename:** `stress11.dat` is hardcoded in `main.cpp:142`. Unlike `standalone` which generates per-step snapshot files, this program writes a single output file after all 1000 steps complete.

8. **Load/Store test is the only GPU verification:** The only correctness check is the initial `acVerifyMesh("Load/Store", ...)` comparison. There is no CPU reference for the stress computation, acceleration, or DSL kernels — the program trusts the DSL compilation.

9. **Phase field initialization creates a localized disturbance:** The `initial_condition()` kernel creates a smooth circular "bubble" of `PHI ≈ 1.0` with cosine transition, acting as a source term in the strain equations. The `TrStr` perturbation matrix (`0.01` diagonal) means the phase field contributes a small initial strain.

10. **Stress computation uses original (Lagrangian) coordinates:** `get_first_order_derivatives(ORIGINAL_COORDS.x, ORIGINAL_COORDS.y, UX)` computes derivatives with respect to the original material coordinates, making this a Lagrangian formulation. The acceleration computation, however, uses `AC_COORDS` (current/Eulerian coordinates) for stress gradients.

11. **5-point smoothing stencil is conservative:** Each of the 5 stencil points (center + 4 neighbors) has weight 0.20, summing to 1.0. This preserves constants (a uniform velocity field remains unchanged).

12. **Boundary conditions are periodic in 2D:** `periodic(BOUNDARY_XY)` means both X and Y axes use periodic boundaries, appropriate for studying wave propagation in an infinite medium approximation. No Z dimension is needed (1D in Z).

13. **npoints = 84×2 = 168:** The DSL defines `npoints = 42×2 = 84` but then `npointsx_grid = npoints`. However, `main.cpp` uses `npointsx_grid` which may be overridden by `user_constants.inc` or `math_utils.h` includes. The comment in `main.cpp:72` references `npointsx_grid` suggesting it may be a preprocessor constant from the includes.

14. **Unused code paths:** Lines 102-126 (periodic BCs test) and lines 152-182 (1D data extraction to `a.dat`, `u.dat`, `x.dat`) are commented out. These suggest earlier versions tested specific analytical solutions (possibly shock tube or wave propagation benchmarks).

15. **Atexit abort handler:** `atexit(acAbort)` registers `acAbort()` to run on normal exit. This function calls `MPI_Abort()` only if `finalized == false`, preventing double-finalization errors. The `finalized = true` flag is set at the end of `main()`.

16. **Max process limit:** `max_devices = 2 × 2 × 4 = 16`. If `nprocs > 16`, the program aborts. This is a hard limit tied to the expected GPU/node configuration (2 GPUs per node, 2 nodes, 4 processes per GPU).

17. **TrStr perturbation is isotropic and small:** `TrStr = {{0.01, 0, 0}, {0, 0.01, 0}, {0, 0, 0.01}}` adds a uniform 1% pre-strain to the diagonal components, scaled by `PHI`. This creates a localized initial stress perturbation that triggers wave propagation.

18. **Strain computation is symmetric:** `strainxy` and `strainyx` use the same formula `0.5 × (∂UY/∂x + ∂UX/∂y)`, which is the standard engineering shear strain. In linear elasticity, these should be equal, and the stress tensor components `STRESS12` and `STRESS21` may differ due to the elasticity tensor's off-diagonal coupling.

19. **No diagnostic output during simulation:** The program does not print any status, progress, or field statistics during the 1000-step loop. All output is deferred to the single `stress11.dat` file written after completion. This makes debugging difficult for long runs.

20. **SIMTime is defined but unused:** `SimTime = 75.0` (= 3 × 25.0) is defined in `solver.ac:4` but never used in the main execution loop. The loop runs for `nsteps = 1000` iterations with `eldt = 2 × 10⁻¹²` per step, giving total simulated time of `2 × 10⁻⁹` — vastly different from `SimTime`. This suggests `SimTime` was intended for a different timestep strategy or is a leftover from a previous version.
