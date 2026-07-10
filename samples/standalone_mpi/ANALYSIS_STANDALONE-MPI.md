# ⚠️ WARNING ⚠️
# **This is a LLM based analysis.**
# **THE BRANCH SHOULD NEVER BE MERGED TO master OR develop BRANCHES!**

# LLM Information
- **Model:** Qwen3.6-35B-A3B

# Overview
The `standalone_mpi` sample is Astaroth's **full production-grade simulation driver** (`ac_run_mpi`). It is a complete MPI-enabled HPC application that runs magnetohydrodynamic (MHD) or variant simulations with periodic/complex boundary conditions, adaptive timestepping, diagnostics, snapshot/slice I/O, runtime configuration reloading, user signal files (STOP/RELOAD), optional helical forcing, sink/accretion physics, shock viscosity, and multi-fluid support. It uses the modern `acGrid*` API (not the legacy `acDevice*` API) and the task graph DSL for simulation step definition. It is the primary entry point for running Astaroth simulations on GPU clusters.

# Directory Structure & File Descriptions

| File | Description |
| :--- | :--- |
| `CMakeLists.txt` | Build config; compiles `main.cc` and `simulation_rng.cc` into `ac_run_mpi`, links `astaroth_core` and `astaroth_utils`, includes `include/` and `include/rapidcsv/`. |
| `main.cc` | Main simulation driver (~1933 lines): MPI initialization, CLI parsing, config loading, mesh setup, simulation loop, diagnostics, I/O, event handling. |
| `simulation_rng.cc` | C++11 `std::mt19937` random number generation with `AcReal` wrappers; seeded from a fixed seed. |
| `include/config_loader.h` | Declares `set_extra_config_params()` for recalculating derived config parameters after modifying `nx/ny/nz`. |
| `include/host_forcing.h` / `host_forcing.cc` | Helical forcing generation: wave vector selection, perpendicular vector generation, forcing vector computation (PC Manual Eq. 223). |
| `include/host_memory.h` | Declares `InitType` enum (commented out) and `acmesh_init_to()` for mesh initialization by type. |
| `include/rapidcsv/rapidcsv.h` | Header-only CSV parsing library for reading `snapshots_info.csv` during restart. |
| `include/simulation_rng.h` | Declares RNG API: `seed_rng()`, `get_rng()`, `random_uniform_real_01()`. |
| `simulation_control.h` | Simulation period tracking (`SimulationPeriod`), user signal files (`UserSignalFile`), timestamped logging helpers. |
| `simulation_taskgraphs.h` | Custom task graph builders for non-default physics configurations (shock singlepass, heat duct, boundary test). |
| `stencil_loader.h` | Spectral stencil coefficient loader: first/second order derivatives, cross derivatives, upwinding, shock viscosity stencils. |
| `README.md` | Usage guide for LUMI supercomputer: module loading, CMake build, SLURM job submission. |

# Running

```bash
mpirun -np <num processes> ./ac_run_mpi [--config <path>] [--run-init-kernel <name>] [--init-condition <name>] [--from-pc-varfile] [--from-snapshot <dir>] [--help]
```

## Command-Line Options

| Flag | Short | Description |
| :--- | :--- | :--- |
| `--config` | `-c` | Config file path (default: `AC_DEFAULT_CONFIG`). |
| `--run-init-kernel` | `-k` | Run a kernel to initialize the mesh (e.g., `randomize`). |
| `--init-condition` | `-i` | Run a specific initial condition kernel. |
| `--from-pc-varfile` | `-p` | Load mesh from Pencil Code varfile. |
| `--from-distributed-snapshot` | `-d` | Load from distributed snapshot (one file per process, commented out). |
| `--from-monolithic-snapshot` | `-m` | Load from monolithic snapshot (one file, commented out). |
| `--from-snapshot` | `-s` | Load from monolithic snapshot (folder name). |
| `--help` | `-h` | Print usage. |

## Initial Condition Kernels

| `--init-condition` | Description | Physics Configuration |
| :--- | :--- | :--- |
| `Haatouken` | Kinetic kick / cone-like shock ("punching the air") | `Shock_Singlepass_Solve` |
| `HeatDuct` | Temperature/entropy-driven duct flow | `Hydro_Heatduct_Solve` |
| `ShockTurb` | Shock turbulence | `Shock_Singlepass_Solve` |
| `BoundTest` | Boundary condition test | `Bound_Test_Solve` |
| (default) | Randomize via `--run-init-kernel randomize` | `MHD` |

# Preconditions

| Precondition | Value | Description |
| :--- | :--- | :--- |
| `AC_MPI_ENABLED` | Must be `ON` | The program is MPI-only; without it, `main()` prints an error and exits with `EXIT_FAILURE`. |
| MPI thread level | `MPI_THREAD_MULTIPLE` | Initialized via `MPI_Init_thread`. |

# Data Structures

## Simulation Control

| Structure | Description |
| :--- | :--- |
| `AcMeshInfo` | Runtime configuration (grid dims, physical parameters, MPI communicator). |
| `AcMesh` | Full mesh state (field data, boundary conditions). |
| `AcTaskGraph*` | Pre-compiled simulation step task graph (3 substeps). |
| `Device` | GPU device handle (internal to Astaroth). |
| `SimulationPeriod` | Tracks step/time-based periodic actions (diagnostics, snapshots, slices). |
| `UserSignalFile` | File-based signal detection: checks if file modification time has changed. |
| `ForcingParams` | Helical forcing parameters: magnitude, phase, wave vector, real/imaginary vectors, kaver. |
| `CommandLineArguments` | Parsed CLI arguments. |

## Enums

| Enum | Values | Description |
| :--- | :--- | :--- |
| `Simulation` | `MHD`, `Shock_Singlepass_Solve`, `Hydro_Heatduct_Solve`, `Bound_Test_Solve`, `Default` | Which simulation program to run. |
| `PhysicsConfiguration` | `MHD`, `ShockSinglepass`, `HydroHeatduct`, `BoundTest`, `Default` | User-facing physics selection (maps to `Simulation`). |
| `InitialMeshProcedure` | `InitKernel`, `LoadPC_Varfile`, `LoadDistributedSnapshot`, `LoadMonolithicSnapshot`, `LoadSnapshot`, `InitHaatouken`, `InitBoundTest` | How to set up the initial mesh. |
| `PeriodicAction` | `PrintDiagnostics`, `WriteSnapshot`, `WriteSlices`, `EndSimulation`, `GenerateForcing` | What to do at each period. |
| `SimulationEvent` | `NanDetected=0x001`, `StopSignal=0x002`, `TimeLimitReached=0x004`, `ConfigReloadSignal=0x008` | Bitmask events that can end or modify the simulation. |

## Constants

| Constant | Value | Description |
| :--- | :--- | :--- |
| `ARRAY_SIZE(arr)` | `sizeof(arr)/sizeof(*arr)` | Array element count. |
| `GEN_TEST_FILE` | 0 | Compile-time flag (not used). |
| `snapshot_output_dir` | `"output-snapshots"` | Directory for mesh snapshots. |
| `slice_output_dir` | `"output-slices"` | Directory for 2D slices. |
| `sim_log_msg_len` | 512 | Max length of simulation log message. |

# Key Macros

## fprintf Override

```c
#define fprintf(...) { int tmppid; MPI_Comm_rank(MPI_COMM_WORLD, &tmppid); if (!tmppid) { fprintf(__VA_ARGS__); } }
```

This macro redirects all `fprintf` calls in `main.cc` to **rank 0 only**, avoiding interleaved output from multiple MPI processes.

## Warning/Error Macros

| Macro | Behavior |
| :--- | :--- |
| `WARNING(msg)` | Prints a warning message. |
| `ERROR(msg)` | Prints an error and aborts. |
| `ERRCHK_ALWAYS(expr)` | Checks expression, aborts on failure. |
| `WARNCHK_ALWAYS(expr)` | Warning + continue on failure. |

# Program Flow

## 1. MPI Initialization

```cpp
MPI_Init_thread(NULL, NULL, MPI_THREAD_MULTIPLE, &thread_support_level);
```

Checks that `MPI_THREAD_MULTIPLE` is supported; aborts otherwise.

## 2. Command-Line Parsing

`read_command_line_arguments()` uses `getopt_long()` with these options:
- `-c <config_path>` — config file
- `-k <kernel_name>` — init kernel
- `-i <initcond_name>` — specific initial condition (Haatouken, HeatDuct, ShockTurb, BoundTest)
- `-p` — PC varfile
- `-d` — distributed snapshot (parsed but not yet implemented)
- `-m` — monolithic snapshot (parsed but not yet implemented)
- `-s <folder>` — snapshot folder

Sets `res->initial_mesh_procedure` and `res->simulation_physics`.

## 3. Configuration Loading

```cpp
AcMeshInfo info = load_config_file(cmdline_args.config_path);
```

1. `acLoadConfig(config_path, &info)` — loads from file.
2. Sets `AC_MPI_comm_strategy = AC_MPI_COMM_STRATEGY_DUP_WORLD`.
3. Sets `AC_proc_mapping_strategy = AC_PROC_MAPPING_STRATEGY_MORTON`.
4. Sets `AC_decompose_strategy = AC_DECOMPOSE_STRATEGY_MORTON`.
5. `info.comm->handle = MPI_COMM_WORLD`.
6. `info.runtime_compilation_log_dst = "ac_compilation_log"`.
7. If `AC_RUNTIME_COMPILATION`: `acCompile()`, `acLoadLibrary()`, `acLoadUtils()`.

## 4. Configuration Output

1. `write_info()` — writes `purge.sh` and `mesh_info.list` with all config parameters.
2. `create_output_directories()` — creates `output-snapshots/` and `output-slices/`.
3. `acPrintMeshInfo(*info)` — prints config to stdout.

## 5. Astaroth Initialization

```cpp
acGridInit(info);
```

Allocates mesh, decomposes across MPI ranks, loads config to GPU.

## 6. Mesh Initialization

Switches on `cmdline_args.initial_mesh_procedure`:

| Case | Action |
| :--- | :--- |
| `InitKernel` | Looks up kernel name, launches `acGridLaunchKernel(STREAM_DEFAULT, kernel, dims.n0, dims.n1)`, swaps buffers. |
| `InitHaatouken` | Randomizes all buffers, then launches `haatouken` kernel (cone-like shock). |
| `InitBoundTest` | Launches `constant` and `radial_vec_initcond` kernels, applies periodic BCs. |
| `LoadPC_Varfile` | Calls `read_varfile_to_mesh_and_setup()` with Pencil Code varfile path. |
| `LoadDistributedSnapshot` | Calls `read_file_to_mesh_and_setup(dir, start_step, sim_time, distributed=true)`. |
| `LoadSnapshot` | Calls `read_file_to_mesh_and_setup(dir, start_step, sim_time, distributed=false)`. |

## 7. Initial Diagnostics (if start_step == 0)

1. Open `timeseries.ts` for appending.
2. `print_diagnostics_header_from_root_proc()` — writes CSV header to `timeseries.ts`.
3. `calc_timestep()` — calculates initial `dt` from device outputs.
4. `print_diagnostics()` — computes min/rms/max of velocity, magnetic field, all vertex buffers, and writes to `timeseries.ts`.
5. Checks for NaN values (`found_nan`).
6. `write_slices(pid, 0, 0.0)` — writes initial slices.
7. `save_mesh_mpi_async(info, "output-snapshots", 0, 0, 0.0)` — writes initial snapshot.
8. If forcing is enabled: `generateForcingParams()` and `loadForcingParamsToGrid()`.

## 8. Simulation Loop Setup

### Periodic Actions

| Action | Pre/Post | Period Parameter | Time Parameter |
| :--- | :--- | :--- | :--- |
| `PrintDiagnostics` | Pre-step | `AC_save_steps` | None |
| `WriteSnapshot` | Pre-step | `AC_bin_steps` | `AC_bin_save_t` |
| `WriteSlices` | Pre-step | `AC_slice_steps` | `AC_slice_save_t` |
| `GenerateForcing` | Pre-step | `AC_forcing_period_steps` | `AC_forcing_period_t` |
| `EndSimulation` | Post-step | `AC_max_steps` | `AC_max_time` |

### Signal Files

| File | Event |
| :--- | :--- |
| `"STOP"` | `SimulationEvent::StopSignal` |
| `"RELOAD"` | `SimulationEvent::ConfigReloadSignal` |

### Timestep Calculation

```cpp
acGridExecuteTaskGraph(acGetOptimizedDSLTaskGraph(AC_calc_timestep), 1);
```

```cpp
AcReal calc_timestep(AcMeshInfo info) {
    if (info[AC_additive_timestep])
        return acDeviceGetOutput(acGridGetDevice(), AC_dt_min);
    else {
        // Courant-based: dt = min(advec_dt, diffus_dt, diffus3_dt)
        const AcReal uumax = acDeviceGetOutput(acGridGetDevice(), UU_MAX_ADVEC);
        const AcReal vAmax = acDeviceGetOutput(acGridGetDevice(), ALFVEN_SPEED_MAX);
        const AcReal ad_onefluid = acDeviceGetOutput(acGridGetDevice(), AD_ONE_FLUID_MAX_ADVEC);
        const AcReal shock_max = acDeviceGetOutput(acGridGetDevice(), AC_MAX_SHOCK);
        const long double advec_dt = cdt * dsmin / (fabs(uumax) + ad_onefluid + sqrt(cs2_sound + vAmax * vAmax));
        const long double diffus3_dt = (nu_hyper3 != 0) ? cdt3 * dsmin6 / nu_hyper3 : AC_REAL_MAX;
        const long double diffus_dt = cdtv * dsmin * dsmin / (max(nu_visc, eta) + nu_shock * shock_max);
        return min(advec_dt, min(diffus_dt, diffus3_dt));
    }
}
```

## 9. Main Simulation Loop

```cpp
for (int i = start_step;; ++i) {
```

### Step 9a: Calculate Timestep

```cpp
acGridExecuteTaskGraph(acGetOptimizedDSLTaskGraph(AC_calc_timestep), 1);
const AcReal dt = calc_timestep(info);
```

### Step 9b: Set Device Inputs

```cpp
acDeviceSetInput(acGridGetDevice(), AC_dt, dt);
acDeviceSetInput(acGridGetDevice(), AC_current_time, simulation_time);
```

Optional (if `LSINK`): `acGridLoadScalarUniform(AC_M_sink, sink_mass)`.

### Step 9c: Execute Simulation Step (3 substeps)

```cpp
const int num_substeps = 3;
for (int substep = 0; substep < num_substeps; ++substep) {
    acDeviceSetInput(acGridGetDevice(), AC_SUBSTEP, (AC_SUBSTEP_NUMBER)substep);
    acGridExecuteTaskGraph(acGetOptimizedDSLTaskGraph(AC_rhs_substep), 1);
}
simulation_time += dt;
```

### Step 9d: Pre-Step Periodic Actions

Checks each `pre_step_actions` entry; triggers:
- **PrintDiagnostics**: Computes min/rms/max of all fields, writes to `timeseries.ts`, checks for NaN.
- **WriteSnapshot**: `save_mesh_mpi_async()` — writes mesh snapshots asynchronously.
- **WriteSlices**: `write_slices()` — writes 2D slices asynchronously.
- **GenerateForcing**: `generateForcingParams()` + `loadForcingParamsToGrid()`.

### Step 9e: Post-Step Periodic Actions

- **EndSimulation**: Sets `TimeLimitReached` event if max steps/time reached.

### Step 9f: Check Signal Files

If rank 0 detects file modification:
- `STOP` file modified → `StopSignal` event.
- `RELOAD` file modified → `ConfigReloadSignal` event.

### Step 9g: Broadcast Events

```cpp
MPI_Allreduce(MPI_IN_PLACE, &events, sizeof(uint16_t), MPI_BYTE, MPI_BOR, acGridMPIComm());
```

All processes agree on event flags.

### Step 9h: Event Handling

| Event | Action |
| :--- | :--- |
| `NanDetected` | Logs "FOUND NAN → exiting" |
| `StopSignal` | Logs "Got STOP signal → exiting" |
| `TimeLimitReached` | Logs max steps/time reached |
| `ConfigReloadSignal` | Reloading config file (see below) |

### Step 9i: End Condition Check

```cpp
if (check_event(events, SimulationEvent::EndCondition)) {
    break;
}
events = 0;
```

### Config Reload (ConfigReloadSignal)

1. `MPI_Barrier()` — synchronize all processes.
2. `acLoadConfig()` — reload config file into `new_info`.
3. Compare old vs. new values for all parameter types (`int`, `int3`, `real`, `real3`).
4. Check for:
   - NaN values in new parameters → abort reload.
   - Run constant changes (`AC_start_step`, etc.) → abort reload.
   - No changes → skip reload.
5. If valid: `acHostUpdateParams()`, `acGridDecomposeMeshInfo()`, `acDeviceLoadMeshInfo()`, update `pre_step_actions` and `post_step_actions` periods.

## 10. Cleanup

1. `acGridDiskAccessSync()` — sync all pending disk writes.
2. `acGridQuit()` — tear down GPU resources.
3. `fclose(diag_file)` — close `timeseries.ts`.
4. `ac_MPI_Finalize()` — finalize MPI.
5. Return `EXIT_SUCCESS` or `EXIT_FAILURE` based on error events.

# Key Astaroth APIs Used

## Grid Initialization / Teardown

| Function | Description |
| :--- | :--- |
| `acGridInit(info)` | Initialize Astaroth grid: allocate mesh, decompose, load to GPU. |
| `acGridQuit()` | Tear down GPU resources, free mesh. |

## Mesh Operations

| Function | Description |
| :--- | :--- |
| `acGridLaunchKernel(stream, kernel, n0, n1)` | Launch a DSL kernel on the grid. |
| `acGridSwapBuffers()` | Swap input/output buffers. |
| `acGetMeshDims(local_info)` | Get mesh dimensions (n0, n1, n2). |
| `acGridGetLocalMeshInfo()` | Get local mesh info struct. |
| `acGridGetDevice()` | Get device handle. |

## Task Graph Execution

| Function | Description |
| :--- | :--- |
| `acGridExecuteTaskGraph(task_graph, count)` | Execute a task graph `count` times. |
| `acGetOptimizedDSLTaskGraph(graph_type)` | Get the optimized task graph for a given type. |
| `acGridGetDefaultTaskGraph()` | Get the default (MHD) task graph. |
| `acGridBuildTaskGraph(ops[])` | Build a custom task graph from operations. |
| `acGridDestroyTaskGraph(graph)` | Destroy a custom task graph. |
| `acGraphPrintDependencies(graph)` | Print task graph dependencies (debug). |

## Device Communication

| Function | Description |
| :--- | :--- |
| `acDeviceSetInput(device, param, value)` | Set a device input parameter. |
| `acDeviceGetInput(device, param)` | Get a device input parameter. |
| `acDeviceGetOutput(device, param)` | Get a device output parameter (e.g., `UU_MAX_ADVEC`, `ALFVEN_SPEED_MAX`). |
| `acDeviceLoadMeshInfo(device, info)` | Load mesh info to device. |
| `acDeviceGetLocalConfig(device)` | Get local device config. |

## Boundary Conditions

| Function | Description |
| :--- | :--- |
| `acGridPeriodicBoundconds(stream)` | Apply periodic boundary conditions. |

## Reductions

| Function | Description |
| :--- | :--- |
| `acGridReduceScal(stream, rtype, vhandle, &val)` | Reduce a scalar vertex buffer. |
| `acGridReduceVec(stream, rtype, vx, vy, vz, &val)` | Reduce a vector vertex buffer (3 components). |
| `acGridReduceVecScal(stream, rtype, vx, vy, vz, scalar, &val)` | Reduce Alfvén speed (vector + scalar). |

## I/O

| Function | Description |
| :--- | :--- |
| `acGridWriteMeshToDiskLaunch(job_dir, step)` | Non-blocking snapshot write to disk. |
| `acGridWriteSlicesToDiskLaunch(dir, step, time)` | Non-blocking slice write. |
| `acGridDiskAccessSync()` | Synchronize all pending disk I/O. |
| `acGridAccessMeshOnDiskSynchronous(field, dir, step, ACCESS_READ)` | Synchronous field read from disk. |
| `acGridAccessMeshOnDiskSynchronousDistributed(field, dir, step, ACCESS_READ)` | Distributed synchronous field read. |
| `acGridReadVarfileToMesh(path, fields, count, nn, rr)` | Read Pencil Code varfile to mesh. |
| `acGridLoadStencils(stream, stencils)` | Load spectral stencil coefficients. |
| `acGridSynchronizeStream(stream)` | Synchronize a specific stream. |
| `acGridSynchronizeStream(STREAM_ALL)` | Synchronize all streams. |

## Configuration

| Function | Description |
| :--- | :--- |
| `acLoadConfig(path, &info)` | Load config from file. |
| `acPrintMeshInfo(info)` | Print config to stdout. |
| `acHostUpdateParams(&info)` | Update host-side parameters from config. |
| `acGridDecomposeMeshInfo(info)` | Decompose config for MPI distribution. |
| `acPushToConfig(info, param, value)` | Push a config value. |

## Scalar Uniform Loading

| Function | Description |
| :--- | :--- |
| `acGridLoadScalarUniform(stream, param, value)` | Load a scalar parameter to all device buffers. |

## Logging

| Function | Description |
| :--- | :--- |
| `acLogFromRootProc(pid, fmt, ...)` | Log from any process (rank 0 only). |
| `acVA_LogFromRootProc(pid, msg, args)` | VArgs version of `acLogFromRootProc`. |
| `acVA_DebugFromRootProc(pid, msg, args)` | Debug-level logging (only in `NDEBUG` builds). |

## MPI

| Function | Description |
| :--- | :--- |
| `acGridMPIComm()` | Get Astaroth MPI communicator. |
| `ac_MPI_Finalize()` | Finalize MPI. |

# Key DSL Task Graph Operations

| Operation | Description |
| :--- | :--- |
| `acHaloExchange(fields)` | Exchange halo/ghost cell data. |
| `acBoundaryCondition(boundary, cond, fields)` | Apply boundary condition. |
| `acCompute(kernel, fields)` | Execute a kernel on fields. |
| `acComputeWithParams(kernel, fields, loader)` | Execute kernel with custom parameter loading. |

# Key DSL Kernels Referenced

| Kernel | Description |
| :--- | :--- |
| `randomize` | Random mesh initialization. |
| `constant` | Constant field initialization. |
| `haatouken` | Kinetic kick (cone-like shock). |
| `radial_vec_initcond` | Radial vector field initialization. |
| `beltrami_initcond` | Beltrami flow initialization (commented out in bound test). |
| `scale` | Scale magnetic field. |
| `AC_BUILTIN_RESET` | Reset output buffers to zero/NaN. |
| `KERNEL_shock_1_divu` | Shock viscosity: compute ∇·u. |
| `KERNEL_shock_2_max` | Shock viscosity: compute max. |
| `KERNEL_shock_3_smooth` | Shock viscosity: smooth shock field. |
| `KERNEL_singlepass_solve` | Single-pass integration step. |
| `twopass_solve_intermediate` | Two-pass integration: intermediate substep. |
| `KERNEL_twopass_solve_intermediate` | Two-pass integration (parameterized): intermediate. |
| `twopass_solve_final` | Two-pass integration: final substep. |
| `KERNEL_twopass_solve_final` | Two-pass integration (parameterized): final. |

# Output Files

| File | Description |
| :--- | :--- |
| `timeseries.ts` | Time series diagnostics (min/rms/max per step). |
| `mesh_info.list` | Mesh configuration metadata (written at startup). |
| `purge.sh` | Cleanup script (removes output files). |
| `ac_compilation_log` | Runtime compilation log. |
| `output-snapshots/` | Directory for mesh snapshots (`snapshots_info.csv` + step data). |
| `output-slices/` | Directory for 2D slices (`step_XXXXXXXXXXXX/*`). |

# Notable Observations

1. **Production-grade driver**: This is not a test program — it's the full simulation runner that would be used on production supercomputers (LUMI, etc.).

2. **Adaptive timestepping**: The timestep is computed dynamically from device outputs (`UU_MAX_ADVEC`, `ALFVEN_SPEED_MAX`) using Courant-Friedrichs-Lewy (CFL) conditions for advection, diffusion, and hyper-diffusion.

3. **3-substep Runge-Kutta**: The simulation loop runs 3 substeps per full time step via `AC_rhs_substep`, typical of RK3 methods.

4. **Config reload at runtime**: Users can trigger config reload by touching a "RELOAD" file — the program compares all parameter types and updates without restarting.

5. **File-based signals**: Users can stop simulation by touching "STOP", or reload config by touching "RELOAD". These are checked on rank 0 and broadcast via `MPI_Allreduce`.

6. **Non-blocking I/O**: Snapshots and slices use asynchronous disk writes (`...Launch`) with `acGridDiskAccessSync()` for synchronization, minimizing I/O blocking.

7. **Morton-order decomposition**: Uses Morton (`Z-order`) curve for MPI rank → spatial domain mapping, optimizing locality for halo exchanges.

8. **Spectral stencils**: `stencil_loader.h` loads finite-difference stencil coefficients for first/second derivatives, cross-derivatives, upwinding, and shock viscosity kernels.

9. **Helical forcing**: `host_forcing.cc` implements Pencil Code-style helical forcing with wave vector selection from integer lattice, perpendicular vector generation with random phase, and relative helicity control.

10. **Multiple physics modes**: Supports MHD (default), shock single-pass, hydro/heat-duct two-pass, and boundary condition tests via custom task graphs.

11. **NaN detection**: Every diagnostic call checks for NaN values in min/rms/max; if detected, simulation exits with error.

12. **Sink physics**: If `LSINK` is defined, tracks `sink_mass` and `accreted_mass` per timestep.

13. **Multi-fluid support**: If `LMULTIFLUID` is defined, iterates over `AC_N_SPECIES` velocity fields for diagnostics and uses drag coefficients.

14. **Windowed reductions**: If `LSPECIAL_REDUCTIONS` is defined, computes radial/Gaussian windowed min/sum/max reductions for localized field analysis.

15. **fprintf macro**: The `fprintf` override in `main.cc:90-97` is a pragmatic (if ugly) solution to prevent interleaved output from MPI processes — all `fprintf` calls in the file become rank-0-only.

16. **Dry run**: The `dryrun()` function exists but is commented out (`//dryrun();`). It tests kernel functionality on a non-initialized mesh.

17. **Custom task graphs are cached**: `simulation_taskgraphs.h` uses a `std::map<Simulation, AcTaskGraph*>` to cache built task graphs.

18. **No cleanup of task graphs in normal path**: `free_simulation_graphs()` is defined but commented out at the end of `main()`. Custom task graphs are leaked on normal exit.

19. **Hardcoded output directories**: `output-snapshots` and `output-slices` are hardcoded `static const char*` strings, not configurable.

20. **RapidCSV for restart**: Uses the header-only RapidCSV library to parse `snapshots_info.csv` when restarting from the latest snapshot (`start_step = -1`).
