# ⚠️ WARNING ⚠️
# **This is a LLM based analysis.**
# **THE BRANCH SHOULD NEVER BE MERGED TO master OR develop BRANCHES!**

# LLM Information
- **Service:** Oulu University Lehmus AI
- **Model:** Qwen3.6-35B-A3B

# Overview
The `advection-example` is the "hello world" of Astaroth. It simulates the advection of a scalar concentration field (C) by a prescribed constant velocity vector. The example demonstrates the complete Astaroth workflow: DSL kernel definition, configuration-driven setup, mesh initialization, task graph execution, and slice-based output rendering.

# Directory Structure & File Descriptions

| File | Description |
| :--- | :--- |
| `CMakeLists.txt` | Minimal build configuration; compiles `main.cc` and links against `astaroth_core` and `astaroth_utils`. |
| `DSL/solver.ac` | The Astaroth DSL source defining the field `C`, velocity, Euler update kernel, initial condition kernel, boundary conditions, compute steps, and simulation parameters. |
| `main.cc` | Host-side C++ entry point: initializes info objects, loads config, computes grid spacings, launches task graphs in a time-stepping loop, and writes output slices. |
| `advec.conf` | Configuration file providing grid dimensions, domain length, slice frequency, CFL number, and periodicity flags. |
| `build.sh` | Build script that creates a `build/` directory, invokes CMake with MPI enabled, and compiles the executable. |
| `disbatch.sh` | SLURM batch submission script for cluster execution (configured for LUMI), runs the executable, then post-processes slices into movies via `ac_render_slices`. |
| `README.md` | Documentation with build/run instructions and references to related samples for deeper learning. |

# DSL (`DSL/solver.ac`) Breakdown

## Includes
The DSL includes standard Astaroth library modules:
- `math` — General mathematical functions
- `derivs.h` — Sixth-order finite-difference derivative operators
- `operators.h` — Gradient, Laplacian, Jacobian, etc.
- `grid` — Grid position helpers

## Field & Variables
- `Field C` — The scalar concentration field being advected. Like all Astaroth fields, it has **in** (read) and **out** (write) double buffers.
- `real3 velocity = (1.0, 0.0, 0.0)` — Constant advection velocity in the x-direction.
- `real v_abs_max` — Maximum velocity magnitude, computed from the velocity vector components.
- `real AC_cdt = 0.05` — CFL number (configurable via `advec.conf`, overrides DSL default).
- `real AC_dt` — Computed timestep based on CFL condition: `AC_cdt * (AC_dsmin / v_abs_max)`.
- `real AC_max_time = 2*AC_REAL_PI` — Total simulation time (host-only variable).
- `int AC_slice_steps = 100` — Frequency for writing output slices (host-only variable).

## Kernels
| Kernel | Arguments | Description |
| :--- | :--- | :--- |
| `euler_update_kernel` | `real dt` | Computes the advection term `dot(velocity, gradient(C))` and updates the field via `ite(C, C + dt * rhs)`. |
| `initial_condition_kernel` | None | Sets the initial field to `initial_amplitude * sin(grid_position().x)`, producing a sine wave along the x-axis. |

## Boundary Conditions
```
BoundConds bcs { periodic(BOUNDARY_XYZ) }
```
Periodic boundaries in all three spatial directions.

## ComputeSteps
| ComputeStep | Description |
| :--- | :--- |
| `initial_condition(bcs)` | Executes `initial_condition_kernel` to set up the initial sine wave. |
| `euler_update(bcs)` | Executes `euler_update_kernel(AC_dt)` each timestep. |

# Host (`main.cc`) Workflow

The host program follows this sequence:

1. **`acInitInfo()`** — Creates an info object to hold global variables from the DSL.
2. **`acLoadConfig("advec.conf", &info)`** — Loads runtime parameters from the configuration file.
3. **`acPushToConfig(info, AC_ds, ...)`** — Computes and pushes grid spacings (domain length / grid points) to the info object.
4. **`acHostUpdateParams(&info)`** — Ensures DSL default values fill in any unset variables.
5. **`acGridInit(info)`** — Allocates device buffers, sets up MPI, and initializes the mesh.
6. **`acGetOptimizedDSLTaskGraph(...)`** — Creates optimized task graph handles for both compute steps.
7. **`acGridExecuteTaskGraph(init_graph, 1)`** — Runs the initial condition kernel.
8. **Time-stepping loop:**
   - Every `AC_slice_steps` iterations, writes XY slices to disk via `acGridWriteSlicesToDiskSynchronous`.
   - Executes the Euler update via `acGridExecuteTaskGraph(update_graph, 1)`.
   - Advances `simulation_time` by `AC_dt`.
9. **`acGridQuit()`** — Cleans up device resources and exits.

# Configuration (`advec.conf`)

| Parameter | Value | Description |
| :--- | :--- | :--- |
| `AC_ngrid` | `{256, 8, 8}` | Number of grid points per dimension. |
| `AC_len` | `{6.28318530718, 6.28318530718, 6.28318530718}` | Domain length in each dimension (one full period of 2π). |
| `AC_slice_steps` | `100` | Frequency of slice output. |
| `AC_cdt` | `0.05` | CFL number for timestep calculation. |
| `AC_periodic_grid` | `{True, True, True}` | Periodic boundaries in all directions. |
| `initial_amplitude` | `1.0` | Amplitude of the initial sine wave. |
| `AC_skip_single_gpu_optim` | `T` | Skips single-GPU optimization passes. |

# Running the Example

### Manual execution:
```bash
cd astaroth
pip install -r requirements.txt
. ./sourceme.sh
cd samples/advection-example
./build.sh
./build/advection-example
```

### Cluster execution (SLURM):
```bash
sbatch disbatch.sh
```
Output post-processing produces movies at `output-postprocessed/movies/lines/C.png`.

# Key Dependencies
- `astaroth.h` — Core Astaroth library API and data structures.
- `astaroth_utils.h` — Utility functions including `acInitInfo`, `acLoadConfig`, `acPushToConfig`, `acHostUpdateParams`, `acGridInit`, `acGridExecuteTaskGraph`, `acGridWriteSlicesToDiskSynchronous`, `acGridQuit`.
- `errchk.h` — Error checking macros.
- `MPI` — Parallel communication (the build script enables `-DMPI_ENABLED=ON`).

# Learning Path
The README recommends the following progression for learning Astaroth:
1. **`samples/advection-example/`** — This example: simple, minimal, good starting point.
2. **`samples/standalone_mpi/main.cc`** — A more robust solver implementation.
3. **`acc-runtime/samples/mhd_modular/mhdsolver.ac`** — Complex equation setup solving ideal MHD equations.
4. **`acc-runtime/README.md`** — In-depth DSL documentation.
