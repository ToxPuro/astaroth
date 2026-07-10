# ⚠️ WARNING ⚠️
# **This is a LLM based analysis.**
# **THE BRANCH SHOULD NEVER BE MERGED TO master OR develop BRANCHES!**

# LLM Information
- **Model:** Qwen3.6-35B-A3B

# Overview
> **⚠️ DEPRECATED:** This directory (`samples/standalone/`) is deprecated. The up-to-date version is `samples/standalone_mpi/`.

The `standalone` directory provides a CPU-based reference implementation and a four-mode multi-tool executable (`ac_run`) for the Astaroth framework. It consists of two major subsystems: (1) a comprehensive CPU-side numerical model written in C++ that re-implements the full hydrodynamic/MHD solver, boundary conditions, and time integration on the host using `long double` (`ModelScalar`) precision — designed for correctness verification against the GPU kernel results; and (2) a unified CLI tool (`main.cc`) that exposes four operational modes: autotest (GPU-CPU numerical verification), benchmark (performance profiling), simulation (full time-advance loop with I/O and diagnostics), and real-time rendering (SDL2-based visualization). The standalone build compiles into a static library (`astaroth_standalone`) linked against OpenMP for parallelism and the core Astaroth GPU library. A `model/` subdirectory contains 13 files split across mesh operations, reduction operations, RK3 time integration, boundary conditions, diffusion stencils, host forcing, and host memory management — all implemented in `long double` precision as a reference against which the GPU `AcReal` (float/double) results are compared.

# Directory Structure & File Descriptions

| File | Description |
| :--- | :--- |
| `CMakeLists.txt` | Build config: creates `astaroth_standalone` static library (all `.cc` in root and `model/`) and `ac_run` executable (only `main.cc`). Requires OpenMP, optional SDL2 for `BUILD_RT_VISUALIZATION`. |
| `main.cc` | CLI entry point with option parsing (`--help`, `--test`, `--benchmark`, `--simulate`, `--render`, `--config <path>`). Dispatches to `run_autotest`, `run_benchmark`, `run_simulation`, or `run_renderer`. |
| `run.h` | Function declarations for all four operational modes. |
| `timer_hires.h` | High-resolution timer using `clock_gettime(CLOCK_REALTIME)`, returns nanosecond differences. |
| `config_loader.cc` | Parses `astaroth.conf` (keyword = value format), initializes `AcMeshInfo` struct via `memset` to `0xFF` for uninitialized-value detection, calls `update_config()` to compute derived parameters. |
| `config_loader.h` | `load_config()` and `update_config()` declarations. |
| `autotest.cc` | GPU-CPU numerical verification engine. Runs all test initial conditions across 7 mesh sizes, checking scalar/vector reductions and full RK3 time-step integration. Produces colored terminal output with ULP/absolute/relative error reporting. Includes two modes: `QUICK_TEST` and `THOROUGH_TEST` (via preprocessor). |
| `benchmark.cc` | Performance benchmarking. Runs 100 synchronized `acIntegrate()` steps on `n×n×n` or `n×n×(n*num_processes)` grids with `dt = FLT_EPSILON`, collects per-step timing, reports 90th percentile. |
| `simulation.cc` | Full production simulation driver. Manages the time loop, saves/loads binary `.mesh` snapshots, writes `timeseries.ts` diagnostics, handles forcing, sink particles, shock viscosity, and conditional early termination (NaN, dt too low, STOP file). |
| `renderer.cc` | SDL2-based real-time X-Y slice visualization. Displays all vertex buffers as individual tiles plus a vector composite, supports camera zoom/pan, key-slice navigation, and spacebar initial-condition cycling. |
| `model/host_memory.cc` | Host mesh allocation, initialization (10 init types), and host↔device data transfer. |
| `model/host_memory.h` | `acmesh_create`, `acmesh_init_to`, `acmesh_destroy`, `modelmesh_create`, `acmesh_to_modelmesh`, `modelmesh_to_acmesh` declarations. |
| `model/host_timestep.cc` | CFL-based adaptive timestep computation. |
| `model/host_timestep.h` | `host_timestep()`, `set_timescale()` declarations. |
| `model/host_forcing.cc` | Helical forcing generator (Pencil Code-inspired, manual Eq. 222), random number generation, forcing parameter struct and device/host loading. |
| `model/host_forcing.h` | `ForcingParams` struct, `generateForcingParams()`, `loadForcingParamsToDevice()`, `loadForcingParamsToHost()` declarations. |
| `model/modelmesh.h` | `ModelMesh` struct: `ModelScalar*` (long double) vertex buffers + `AcMeshInfo` config. |
| `model/model_reduce.cc` | CPU reference reductions: scal and vec for RTYPE_MAX, RTYPE_MIN, RTYPE_SUM, RTYPE_RMS, RTYPE_RMS_EXP. |
| `model/model_reduce.h` | `model_reduce_scal()`, `model_reduce_vec()` declarations. |
| `model/model_rk3.cc` | Full CPU-side 3rd-order Runge-Kutta time integrator with stencil assembly, derivative computation, hydrodynamics equations, induction, entropy, forcing, and Williamson (1980) RK3 coefficients. |
| `model/model_rk3.h` | `model_rk3()`, `model_rk3_step()` declarations. |
| `model/model_boundconds.cc` | CPU-side boundary condition enforcement for `ModelMesh`. |
| `model/model_boundconds.h` | `boundconds()` declaration. |
| `model/model_diff.h` | Template-based diffusion derivative functions (`der_scal`, etc.) using 7-point stencils (3 on each side). |

# Compile-Time Requirements

| Setting | Value | Description |
| :--- | :--- | :--- |
| `CMAKE_CXX_STANDARD` | `11` | C++11 required. |
| `BUILD_RT_VISUALIZATION` | `ON`/`OFF` | If `ON`: defines `AC_BUILD_RT_VISUALIZATION`, links SDL2 from `3rdparty/SDL2`. |
| OpenMP | `REQUIRED` | Always required for `astaroth_standalone` library. |

The `astaroth_standalone` static library links `astaroth_core` (the GPU library) to share type definitions, config constants, and mesh structures. The `ac_run` executable links only against `astaroth_standalone`.

Compile options: `-pipe -Wall -Wextra -Werror -Wdouble-promotion -Wfloat-conversion -Wshadow`

# Compile-Time Options

| Macro | Default | Description |
| :--- | :--- | :--- |
| `AC_BUILD_RT_VISUALIZATION` | `0` or `1` | Enables SDL2 renderer code in `renderer.cc`. |
| `AC_DOUBLE_PRECISION` | Config-dependent | Controls output filenames for test results. |
| `STENCIL_ORDER` | Config-dependent | Stencil order (2, 4, 6, or 8), selects coefficient tables in `model_rk3.cc`. |
| `VERBOSE_PRINTING` | `0` | Prints full config dimensions after `update_config()`. |
| `LDENSITY` | `1` | Include density (LNRHO) in CPU model. |
| `LHYDRO` | `1` | Include hydrodynamics in CPU model. |
| `LMAGNETIC` | `1` | Include magnetic vector potential (AX, AY, AZ) in CPU model. |
| `LENTROPY` | `0` | Include entropy equation in CPU model. |
| `LTEMPERATURE` | `0` | Include temperature in CPU model. |
| `LFORCING` | Per-module | `0` in `simulation.cc`, `1` in `renderer.cc`. Controls forcing kernel in `model_rk3.cc`. |
| `LUPWD` | `1` | Include upwind derivatives in CPU model. |
| `LSINK` | Per-module | `1` if `VTXBUF_ACCRETION` is defined. Controls sink particle logic in `simulation.cc`. |
| `LSHOCK` | `0` | Shock viscosity module in `simulation.cc` (uses device API directly). |
| `LBFIELD` | Per-module | `1` if `BFIELDX/BFIELDY/BFIELDZ` buffers exist. Controls magnetic diagnostics in `simulation.cc`. |
| `QUICK_TEST` | `0` | Autotest mode 1 (colored terminal output). |
| `THOROUGH_TEST` | `1` | Autotest mode 2 (writes `.testresult` files with ULP/absolute/relative errors). |
| `GEN_TEST_RESULT` | `1` | In `QUICK_TEST` mode, writes test results to a file. |
| `BENCH_STRONG_SCALING` | `1` | Benchmark grid: strong scaling (`n×n×n`) vs weak (`n×n×(n*num_processes)`). |

# Input Parameters / Command-Line Interface

| Parameter | Position | Default | Description |
| :--- | :--- | :--- | :--- |
| `--help`, `-h` | Any | — | Print help and available options. |
| `--test`, `-t` | Any | — | Run autotests (GPU vs CPU numerical verification). |
| `--benchmark`, `-b` | Any | — | Run performance benchmark. |
| `--simulate`, `-s` | Any | — | Run full simulation. |
| `--render`, `-r` | Any | — | Run real-time SDL2 renderer. |
| `--config <path>`, `-c <path>` | Any | `AC_DEFAULT_CONFIG` | Path to custom configuration file. |
| `--config` without argument | — | — | Prints error: "Syntax error. Usage: --config <config path>." |

Usage: `./ac_run [options]`

# Program Flow (All Modes)

## 1. Config Loading
`load_config(config_path, &config)` — reads key-value pairs from config, fills `AcMeshInfo.int_params` and `AcMeshInfo.real_params` via string matching against `intparam_names[]`/`realparam_names[]` arrays. Initializes with `memset` to `0xFF` (0xFFFFFFFF) so uninitialized fields are detectable.

## 2. Derived Parameter Computation
`update_config(&config)` — computes:
- Grid padding: `mx = nx + STENCIL_ORDER`, `my = ny + STENCIL_ORDER`, `mz = nz + STENCIL_ORDER`
- Computational domain bounds: `nx_min = STENCIL_ORDER/2`, `nx_max = nx_min + nx`, etc.
- Grid spacing minimum: `dsmin = min(dsx, dsy, dsz)`
- Grid lengths (including ghost zones): `xlen = dsx * mx`, etc.
- Grid origins: `xorig = 0.5 * xlen`, etc.
- Derived helpers: `mxy`, `nxy`, `nxyz`, `cs2_sound`, `cv_sound`
- Physical constants: `G_CONST_CGS`, `M_sun`, `AC_unit_mass`, `AC_M_sink`, `AC_sq2GM_star`

## 3. Mesh Creation
`acmesh_create(info)` — allocates GPU-side `AcMesh` structure.
`modelmesh_create(info)` — allocates CPU-side `ModelMesh` structure (with `long double` buffers).

# Autotest Mode (`--test`)

## 1. Initialization
`acInit(mesh_info)` — initialize GPU subsystem.
Meshes: `gpu_mesh = acmesh_create()`, `model_mesh = modelmesh_create()`.

## 2. Reduction Verification (`check_reductions`)
For each of 7 mesh sizes ({32,64}^3 variants with axis permutations):
  For each init type ({RANDOM, XWAVE, GAUSSIAN_RADIAL_EXPL, ABC_FLOW}):
    1. `acmesh_init_to(type, gpu_mesh)` — initialize GPU mesh.
    2. `acmesh_to_modelmesh(*gpu_mesh, model_mesh)` — copy to CPU mesh.
    3. `acLoad(*gpu_mesh)` — load to GPU.
    4. For each reduction type (RTYPE_MAX, RTYPE_MIN, RTYPE_SUM [skipped], RTYPE_RMS, RTYPE_RMS_EXP):
       a. `model_reduce_scal(*model_mesh, rtype, VTXBUF_UUX)` — CPU scalar reduction.
       b. `acReduceScal(rtype, VTXBUF_UUX)` — GPU scalar reduction.
       c. Compare with `verify()` using configurable error thresholds.
       d. `model_reduce_vec(*model_mesh, rtype, VTXBUF_UUX/UUY/UUZ)` — CPU vec reduction.
       e. `acReduceVec(rtype, VTXBUF_UUX/UUY/UUZ)` — GPU vec reduction.
       f. Compare with `verify()`.

## 3. RK3 Integration Verification (`check_rk3`)
For each of 7 mesh sizes:
  For each init type:
    1. Initialize GPU + CPU meshes.
    2. `acBoundcondStep()` on GPU, `boundconds()` on CPU.
    3. For `num_iterations` (default 1) steps:
       a. `acIntegrate(1e-2)` — GPU time step.
       b. `model_rk3(1e-2, model_mesh)` — CPU time step.
       c. `boundconds()` on CPU, `acBoundcondStep()` on GPU.
    4. `acStore(gpu_mesh)` — transfer GPU mesh to host.
    5. `verify_meshes(*model_mesh, *gpu_mesh)` — per-element comparison.

## Error Verification Logic
`verify(model_val, cand_val, range)` returns `true` if:
- Relative error < 30 machine epsilons (`AC_REAL_EPSILON`), OR
- Absolute error < `range * AC_REAL_EPSILON`

Where relative error = `|1 - candidate/model| / AC_REAL_EPSILON` (measured in ULPs).

# Benchmark Mode (`--benchmark`)

## 1. Initialization
`acmesh_init_to(INIT_TYPE_ABC_FLOW, mesh)` — initialize with ABC flow.
`acInit(mesh_info)`, `acLoad(*mesh)`.

## 2. Warmup
10 iterations of `acIntegrate(0)` (dt=0 to skip computation).

## 3. Timed Loop
For `num_iters` (100) steps:
  1. `timer_reset(&step_time)`.
  2. `acIntegrate(dt)` with `dt = FLT_EPSILON`.
  3. `acSynchronize()`.
  4. Record `timer_diff_nsec(step_time)` in milliseconds.

## 4. Statistics
- Sort results, report 90th percentile per step.
- Write to `{nprocs}_{strong/weak}.bench` file.

# Simulation Mode (`--simulate`)

## 1. Pre-simulation
Load config, create mesh, initialize to `INIT_TYPE_GAUSSIAN_RADIAL_EXPL` (configurable).
If `start_step > 0`: `read_mesh(*mesh, start_step, &t_step)` — load from binary `.mesh` files.
GPU init: `acInit(mesh_info)`, `acLoad(*mesh)`.

## 2. Diagnostics File Setup
Open `timeseries.ts` for append. Write header row with field names (min/rms/max per buffer).

## 3. Mesh Info Export
`write_mesh_info(&mesh_info)` — writes `mesh_info.list` with all config parameters in machine-readable format. Also writes `purge.sh` cleanup script.

## 4. Initial Diagnostics
Print and log initial field statistics (min/rms/max for all buffers, velocity vector, magnetic field if enabled).

## 5. Time Loop (`i = start_step+1` to `max_steps-1`)

### a. Sink Particle Updates (if LSINK)
`acReduceScal(RTYPE_SUM, VTXBUF_ACCRETION)` — sum accreted mass.
`sink_mass = AC_M_sink_init + accreted_mass`.
`acLoadDeviceConstant(AC_M_sink, sink_mass)`.

### b. Adaptive Timestep
`umax = acReduceVec(RTYPE_MAX, UUX, UUY, UUZ)` — max velocity magnitude.
`vAmax = acReduceVecScal(RTYPE_ALFVEN_MAX, BX, BY, BZ, LNRHO)` — max Alfvén speed (if LBFIELD).
`dt = host_timestep(uref, ...)` — CFL-limited timestep.

### c. Forcing (if LFORCING)
`generateForcingParams(mesh_info)` — generate helical forcing parameters.
`loadForcingParamsToDevice(forcing_params)`.

### d. Integration
**With LSHOCK:**
  For each RK3 substep (3×):
  1. `acDevice_shock_1_divu()` — compute divergence of velocity.
  2. `acDeviceSwapBuffer(VTXBUF_SHOCK)` — swap shock buffer.
  3. `acDeviceGeneralBoundconds()` — apply boundary conditions.
  4. `acDevice_shock_2_max()` — compute max shock viscosity.
  5. `acDevice_shock_3_smooth()` — smooth shock viscosity.
  6. `acDeviceIntegrateSubstep(isubstep, dt)` — RK3 substep.
  7. `acDeviceSwapBuffers()` — swap all buffers.
  8. `acDeviceSwapBuffer(VTXBUF_SHOCK)` — restore shock buffer.

**Without LSHOCK:**
`acIntegrateGBC(mesh_info, dt)` — standard integration with general boundary conditions.

### e. Post-step
`t_step += dt`.

### f. Diagnostics (every `save_steps`)
Print field statistics to stdout and `timeseries.ts`.

### g. Snapshot save (every `bin_save_steps` or `t_step >= bin_crit_t`)
`acBoundcondStepGBC(mesh_info)` — apply boundary conditions.
`acStore(mesh)` — transfer to host.
`save_mesh(*mesh, i, t_step)` — write binary `.mesh` files.
`bin_crit_t += bin_save_t`.

### h. Early termination checks
- `t_step >= max_time` → break.
- `dt < dt_typical / 1e5` for 10+ consecutive steps → break (dt collapse).
- NaN detection in any buffer → break.
- `STOP` file exists in CWD → break.

## 6. Cleanup
`acQuit()` (or `acDeviceDestroy()` if LSHOCK), `acmesh_destroy()`, close diagnostics file.

# Binary Mesh I/O Format (`save_mesh` / `read_mesh`)

For each vertex buffer:
1. Write `AcReal` timestamp (t_step) — 1 value.
2. Write `n` `AcReal` values — the full vertex buffer data.
Filename: `<vtxbuf_name>_<step>.mesh` (e.g., `UUX_0.mesh`, `UUZ_100.mesh`).

To continue from a snapshot: set `AC_start_step` in config, `read_mesh()` loads all buffers for that step number.

# Renderer Mode (`--render`)

## 1. SDL2 Initialization
`SDL_InitSubSystem(SDL_INIT_VIDEO | SDL_INIT_EVENTS)`
800×600 window with hardware-accelerated renderer.
Create `NUM_VTXBUF_HANDLES` + 1 SDL surfaces (one per buffer + one vector composite).

## 2. Camera Setup
Orthographic projection with zoom/pan. Tiles arranged in 3-row layout.
`camera.scale = min(window_width / (ds * num_tiles_per_row), window_height / (ds * num_tile_rows))`

## 3. Simulation Loop (while running)
### a. Input Processing
- Escape / SDL_QUIT → exit.
- Space → cycle through init types.
- I/K → scroll Z slice (k_slice).
- Arrow keys → camera pan.
- PageUp/PageDown → camera zoom.
- Comma/Period → timescale 0.1x / 1.0x.

### b. Simulation Step
- Sink particle mass update (if LSINK).
- Forcing parameter reload (if LFORCING).
- `acReduceVec(RTYPE_MAX)` for velocity → `host_timestep()` for dt.
- `acIntegrate(dt)` — GPU integration.

### c. Rendering (every `desired_frame_time` = 1/60s)
- `acBoundcondStep()` — apply BCs.
- `acStoreWithOffset(dst, num_vertices, mesh)` — partial transfer for current Z slice.
- `renderer_draw(*mesh)` — draw all tiles:
  - Per-buffer: map values to grayscale via `|value - mid| / range * 255`. Color channels assigned per-tile: R, G, or B depending on tile index % 3.
  - Vector composite: R = |UUX - mid|/range, G = |UUY - mid|/range, B = |UUZ - mid|/range.
- Print field min/max values to stdout.

# Host Mesh API Functions Used

| Function | Description |
| :--- | :--- |
| `acmesh_create(info)` | Allocate GPU-side AcMesh. |
| `acmesh_init_to(type, mesh)` | Initialize mesh to one of 10 init types. |
| `acmesh_clear(mesh)` | Zero-fill all buffers. |
| `acmesh_destroy(mesh)` | Free GPU mesh memory. |
| `acmesh_to_modelmesh(acmesh, modelmesh)` | Copy GPU mesh data to CPU ModelMesh (host→device transfer). |
| `modelmesh_to_acmesh(model, acmesh)` | Copy CPU ModelMesh data to GPU AcMesh (device→host transfer). |
| `modelmesh_create(info)` | Allocate CPU-side ModelMesh with long double buffers. |
| `modelmesh_destroy(mesh)` | Free CPU mesh memory. |
| `vertex_buffer_set(handle, val, mesh)` | Set all values in a vertex buffer to a constant. |

# CPU Model API Functions Used

| Function | Description |
| :--- | :--- |
| `model_reduce_scal(mesh, rtype, a)` | CPU scalar reduction (max/min/sum/rms/rms_exp). |
| `model_reduce_vec(mesh, rtype, a, b, c)` | CPU vector reduction (3-component magnitude). |
| `model_rk3(dt, mesh)` | Full RK3 time integration (3 substeps) on CPU. |
| `model_rk3_step(step_number, dt, mesh)` | Single RK3 substep on CPU (used by autotest). |
| `boundconds(info, mesh)` | Apply boundary conditions on CPU. |
| `host_timestep(umax, vAmax, shock_max, info)` | CFL-limited adaptive timestep. |
| `set_timescale(scale)` | Set simulation timescale for renderer. |
| `generateForcingParams(info)` | Generate helical forcing parameters from config. |
| `loadForcingParamsToDevice(params)` | Load forcing parameters to GPU constants. |
| `loadForcingParamsToHost(params, mesh)` | Load forcing parameters to host buffers. |

# Device API Functions Used (Simulation)

| Function | Description |
| :--- | :--- |
| `acInit(info)` | Initialize GPU subsystem. |
| `acQuit()` | Shutdown GPU subsystem. |
| `acLoad(mesh)` | Transfer mesh to GPU. |
| `acStore(mesh)` | Transfer mesh from GPU to host. |
| `acStoreWithOffset(dst, n, mesh)` | Transfer partial mesh with offset. |
| `acIntegrate(dt)` | GPU time integration (all buffers). |
| `acIntegrateGBC(info, dt)` | GPU integration with general boundary conditions. |
| `acBoundcondStep()` | Apply periodic boundary conditions on GPU. |
| `acBoundcondStepGBC(info)` | Apply general boundary conditions on GPU. |
| `acReduceScal(rtype, handle)` | GPU scalar reduction. |
| `acReduceVec(rtype, h0, h1, h2)` | GPU vector reduction (3-component magnitude). |
| `acReduceVecScal(rtype, h0, h1, h2, scalar)` | GPU vector reduction scaled by scalar field. |
| `acReduceVecScal(rtype, h0, h1, h2, h3)` | GPU vector reduction with 4th param. |
| `acSynchronize()` | Wait for all GPU streams. |
| `acSynchronizeStream(stream)` | Wait for specific stream. |
| `acGetNumDevicesPerNode()` | Get device count per node (for multi-GPU benchmark). |
| `acDeviceCreate(id, info, &device)` | Create device on specific GPU ID (LSHOCK mode). |
| `acDeviceDestroy(device)` | Destroy specific device (LSHOCK mode). |
| `acDeviceLoadMesh(device, stream, mesh)` | Load mesh to specific device. |
| `acDeviceStoreMesh(device, stream, mesh)` | Store mesh from specific device. |
| `acDeviceSwapBuffers(device)` | Swap all buffers on device. |
| `acDeviceSwapBuffer(device, handle)` | Swap single buffer on device. |
| `acDeviceGeneralBoundconds(device, stream, b1, b2, info, bindex)` | Apply general BCs on device. |
| `acDevice_shock_1_divu(device, stream, start, end)` | Compute divergence for shock viscosity. |
| `acDevice_shock_2_max(device, stream, start, end)` | Compute max shock viscosity. |
| `acDevice_shock_3_smooth(device, stream, start, end)` | Smooth shock viscosity. |
| `acDeviceIntegrateSubstep(device, stream, substep, start, end, dt)` | RK3 substep on specific device. |
| `acDeviceReduceVec(device, stream, rtype, h0, h1, h2, &result)` | Device vector reduction (pointer output). |
| `acDeviceReduceScal(device, stream, rtype, h0, &result)` | Device scalar reduction (pointer output). |
| `acDeviceReduceVecScal(device, stream, rtype, h0, h1, h2, h3, &result)` | Device vector-scalar reduction (pointer output). |
| `acDevicePrintInfo(device)` | Print device info (LSHOCK mode). |
| `acDeviceLoadDeviceConstant(param, value)` | Set GPU device constant. |

# Mesh Configuration Parameters

From `astaroth.conf`, parsed by `load_config()`:

**Integer parameters:**
- `AC_nx`, `AC_ny`, `AC_nz` — computational grid dimensions (no ghost zones).
- `AC_mx`, `AC_my`, `AC_mz` — total dimensions including ghost zones (`nx + STENCIL_ORDER`).
- `AC_nx_min`, `AC_nx_max` — computational domain bounds.
- `AC_start_step` — step to resume from (0 = fresh start).
- `AC_max_steps` — maximum number of timesteps.
- `AC_save_steps` — diagnostics output interval.
- `AC_bin_steps` — binary mesh save interval.

**Real parameters:**
- `AC_dsx`, `AC_dsy`, `AC_dsz` — grid spacing per dimension.
- `AC_dsmin` — minimum grid spacing.
- `AC_max_time` — maximum simulation time (0 = disabled).
- `AC_bin_save_t` — binary save interval in simulation time.
- `AC_cs_sound` — isentropic sound speed.
- `AC_gamma` — adiabatic index.
- `AC_cp_sound` — specific heat at constant pressure.
- `AC_nu_visc` — kinematic viscosity.
- `AC_zeta` — bulk viscosity.
- `AC_eta` — magnetic diffusivity (if LBFIELD).
- `AC_mu0` — permeability of free space (if LENTROPY).
- `AC_forcing_magnitude`, `AC_forcing_phase` — forcing parameters.
- `AC_k_forcex`, `AC_k_forcey`, `AC_k_forcez` — forcing wavevector.
- `AC_ff_hel_rex`, `AC_ff_hel_rey`, `AC_ff_hel_rez` — helical forcing real components.
- `AC_ff_hel_imx`, `AC_ff_hel_imy`, `AC_ff_hel_imz` — helical forcing imaginary components.
- `AC_kaver` — forcing wavenumber average.
- `AC_G_const` — gravitational constant (normalized).
- `AC_M_sink_Msun` — sink mass in solar masses.
- `AC_GM_star` — stellar gravitational parameter.
- `AC_soft` — gravitational softening length.
- `AC_lnrho0` — reference log density.
- `AC_lnT0` — reference log temperature.
- `AC_unit_length`, `AC_unit_density`, `AC_unit_velocity` — normalization units.

# Stencil Derivative Coefficients (model_rk3.cc)

First derivative coefficients by `STENCIL_ORDER`:

| Order | Coefficients (i=1..MID) |
| :--- | :--- |
| 2 | {1/2} |
| 4 | {2/3, -1/12} |
| 6 | {3/4, -3/20, 1/60} |
| 8 | {4/5, -1/5, 4/105, -1/280} |

Second derivative coefficients:

| Order | Coefficients (0..MID) |
| :--- | :--- |
| 2 | {-2, 1} |
| 4 | {-5/2, 4/3, -1/12} |
| 6 | {-49/18, 3/2, -3/20, 1/90} |
| 8 | {-205/72, 8/5, -1/5, 8/315, -1/560} |

Cross-derivative coefficients (STENCIL_ORDER == 6):
{0, 270/720, -27/720, 2/720} = {0, 3/8, -3/80, 1/360}

Upwind derivative (6th order, LUPWD):
`1/60 * inv_ds * (-20*f[0] + 15*(f[1]+f[-1]) - 6*(f[2]+f[-2]) + (f[3]+f[-3]))`

# RK3 Time Integration (model_rk3.cc)

The CPU RK3 uses Williamson (1980) coefficients:

| Substep | alpha (previous slope weight) | beta (new slope weight) |
| :--- | :--- | :--- |
| 0 | 0.0 | 1/3 |
| 1 | -5/9 | 15/16 |
| 2 | -153/128 | 8/15 |

Full step: `w^{n+1} = w^n + dt * (beta_0*f^0 + beta_1*f^1 + beta_2*f^2)`
where `f^k` is the slope at the kth stage, computed from intermediate states.

# CPU Model Equations (model_rk3.cc)

## Continuity
`continuity(uu, lnrho) = -dot(uu, grad(lnrho)) + upwd_der6(uu, lnrho) - div(uu)`
(with upwind corrective hyperdiffusion if LUPWD)

## Momentum
`momentum(uu, lnrho) = -grad(uu)·uu - cs2*grad(lnrho) + nu_visc*(lap(uu) + 1/3*grad(div(uu)) + 2*S:grad(lnrho)) + zeta*grad(div(uu))`

Where `S` is the traceless strain-rate tensor (stress tensor), computed from velocity gradients.

## Entropy (if LENTROPY)
`entropy(ss, uu, lnrho, aa) = -dot(uu, grad(ss)) + inv_pT*(H - C + eta*mu0*j·j + 2*rho*nu_visc*contract(S) + zeta*rho*div(uu)^2) + heat_conduction(ss, lnrho)`

## Induction (if LMAGNETIC)
`induction(uu, aa) = cross(uu, curl(aa)) - eta*(grad(div(aa)) - lap(aa))`
Uses identity: `nabla x (nabla x A) = nabla(nabla·A) - nabla^2(A)` to avoid second derivatives.

## Helical Forcing
Pencil Code-inspired (manual Eq. 222):
`force = Re[(ff_re + i*ff_im) * exp(i*phi) * exp(i*k·x)]`
Scaled by `sqrt(dt) * cs * sqrt(k_aver * cs)`.

# Vertex Buffer Handles

| Handle | Description |
| :--- | :--- |
| `VTXBUF_LNRHO` | Log density |
| `VTXBUF_UUX` | X velocity |
| `VTXBUF_UUY` | Y velocity |
| `VTXBUF_UUZ` | Z velocity |
| `VTXBUF_AX` | X vector potential (if LMAGNETIC) |
| `VTXBUF_AY` | Y vector potential (if LMAGNETIC) |
| `VTXBUF_AZ` | Z vector potential (if LMAGNETIC) |
| `VTXBUF_ENTROPY` | Entropy (if LENTROPY) |
| `VTXBUF_ACCRETION` | Accretion buffer (if LSINK, i.e., VTXBUF_ACCRETION defined) |
| `VTXBUF_SHOCK` | Shock viscosity buffer (if LSHOCK) |
| `VTXBUF_BFIELDX` | X magnetic field (if LBFIELD) |
| `VTXBUF_BFIELDY` | Y magnetic field (if LBFIELD) |
| `VTXBUF_BFIELDZ` | Z magnetic field (if LBFIELD) |

# Notable Observations

1. **Long double reference precision:** The entire CPU model uses `long double` (typically 80-bit extended precision on x86, or 128-bit quadruple on some platforms) as `ModelScalar`, providing a high-precision reference against which GPU `AcReal` (32-bit or 64-bit float) results are compared. The autotest tolerance is 30 machine epsilon — extremely tight, meaning the GPU code must reproduce CPU results within ~30 ULPs for single precision.

2. **Dual-buffer integration pattern:** `model_rk3()` allocates a temporary `ModelMesh* tmp` for the α-step slope accumulation, then adds β-weighted slopes to the main mesh. This matches the GPU's double-buffered integration but in host memory.

3. **OpenMP parallelization:** All three loops in `model_rk3()` (α-step compute, β-step accumulation, and the full RK3 loop) use `#pragma omp parallel for` on the z-loop, providing multi-threaded CPU execution. The GPU side uses CUDA/HIP streams.

4. **STENCIL_ORDER selects coefficients at compile time:** The derivative coefficients in `model_rk3.cc` are chosen via `#if STENCIL_ORDER == N` preprocessor conditionals. This means the same code supports 2nd, 4th, 6th, and 8th order spatial derivatives without runtime overhead — but the config must be consistent between GPU and CPU builds.

5. **Cross-derivative stencil uses antidiagonal pencil:** The `derxy()` and similar functions read two "pencils" — one along the antidiagonal (offset in both x and y) and one along the opposite antidiagonal (offset x forward, y backward). This is the standard compact cross-derivative approach.

6. **Shock viscosity pipeline (LSHOCK mode):** Three separate GPU kernels run in sequence per RK3 substep: `shock_1_divu` computes divergence, `shock_2_max` finds maximum divergence, `shock_3_smooth` smooths the shock indicator. The shock buffer is swapped in and out between each kernel, interleaved with boundary condition steps. This is a device-specific bypass that doesn't go through the standard `acIntegrate()` path.

7. **Sink particle with mass conservation:** The sink particle accumulates mass from `VTXBUF_ACCRETION` each timestep. `accreted_mass` is monotonically increasing, and `sink_mass = AC_M_sink_init + accreted_mass`. The sink mass is passed to the GPU as a device constant, and the accretion buffer is zeroed each step. A TODO comment notes: "THIS IS A BUG! WILL ONLY SET HOST BUFFER 0!"

8. **Binary mesh format is per-component:** Each vertex buffer is saved as a separate `.mesh` file (`UUX_0.mesh`, `UUZ_0.mesh`, etc.), each containing a timestamp followed by all grid values. This is simple but means a full snapshot of all buffers generates `NUM_VTXBUF_HANDLES` files. The `purge.sh` script generated by `write_mesh_info()` removes all `.mesh` and `.list` files.

9. **Forcing uses Pencil Code Eq. 222:** The helical forcing is directly inspired by the Pencil Code's `forcing_hel_noshear()` function, using complex phasor representation with real and imaginary components, wavevector, and phase. The magnitude is scaled by `sqrt(dt) * cs * sqrt(k_aver * cs)`, following the Pencil Code's energy injection normalization.

10. **Renderer uses partial mesh transfer:** `acStoreWithOffset(dst, num_vertices, mesh)` transfers only the current Z-slice rather than the full 3D mesh, significantly reducing transfer overhead for visualization. This is a performance optimization specific to the renderer mode.

11. **10 initial conditions supported:** RANDOM, XWAVE, GAUSSIAN_RADIAL_EXPL, KICKBALL, ABC_FLOW, SIMPLE_CORE, VEDGE, VEDGEX, RAYLEIGH_TAYLOR, RAYLEIGH_BENARD. The renderer defaults to GAUSSIAN_RADIAL_EXPL (configurable) and cycles through all 10 with spacebar.

12. **Upwind derivatives are separate from compact stencils:** When LUPWD is defined, the CPU model includes additional 6th-order upwind-biased derivatives (`der6x_upwd`, etc.) used only in the continuity equation as a corrective hyperdiffusion term. These are centered-difference with biased coefficients (not compact).

13. **RMS calculations divide by N:** Both `model_reduce_scal` and `model_reduce_vec` compute mean-square (RMS, RMS_EXP) by dividing by `nxyz` after the sum — i.e., they compute `sqrt(Σx²/N)`, not `Σx²`. This is the standard statistical RMS definition.

14. **Config validation via 0xFFFFFFFF sentinel:** After parsing, `load_config()` checks every 32-bit word of the config struct against `0xFFFFFFFF` (the memset fill value). Any match indicates an uninitialized parameter and triggers a WARNING. This is a simple but effective sanity check for config completeness.

15. **CFL timestep uses multiple wave speeds:** `host_timestep()` considers velocity (`umax`), Alfvén speed (`vAmax`), and shock viscosity (`shock_max`) depending on which physics flags are enabled. The reference speed is `max(umax, uu_freefall)` for gravitational runs, or `max(umax, uu_freefall, vAmax)` for MHD runs.

16. **Early termination robustness:** The simulation has five independent early-termination paths: max time reached, dt collapse (dt < dt_typical/1e5 for 10+ steps), NaN detection, STOP file, and LSHOCK shock viscosity limits. All paths perform proper cleanup (boundary conditions, store, save) before breaking.

17. **Renderer bottleneck:** The renderer code contains a comment "Bottleneck is here" on the `renderer_draw(*mesh)` call, indicating the CPU-side per-pixel color mapping and SDL texture upload is the performance limiter in real-time mode.

18. **DEPRECATED function in host_forcing.h:** `DEPRECATED_acForcingVec()` is marked as deprecated in favor of `loadForcingParams`. The `ForcingParams` struct encapsulates all forcing parameters and provides both device and host loading paths.

19. **model_diff.h template stencils:** `model_diff.h` provides `template <AxisType axis>` stencil functions that compile to the appropriate axis at compile time, avoiding branch overhead. These are used for diffusion terms and use a 7-point (r=3) stencil.

20. **Simulation mode defaults to Gaussian radial explosion:** `acmesh_init_to(INIT_TYPE_GAUSSIAN_RADIAL_EXPL, mesh)` is hardcoded as the default initial condition for simulation mode. Other init types (KICKBALL, SIMPLE_CORE) are available via commented-out alternatives in the source.
