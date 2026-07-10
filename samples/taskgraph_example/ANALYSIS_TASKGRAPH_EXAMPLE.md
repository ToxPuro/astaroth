# ⚠️ WARNING ⚠️
# **This is a LLM based analysis.**
# **THE BRANCH SHOULD NEVER BE MERGED TO master OR develop BRANCHES!**

# LLM Information
- **Model:** Qwen3.6-35B-A3B

# Overview
The `taskgraph_example` directory provides a minimal MPI-based demonstration of Astaroth's low-level task graph API — the programmatic (non-DSL) way to construct and execute multi-stage GPU computation pipelines. It builds an `AcTaskGraph` from three operations: halo exchange (for ghost cell synchronization), boundary condition application (periodic on all 3D faces), and a compute kernel (`KERNEL_solve`). The task graph is executed for 3 iterations on 6 vertex buffers (density + 3 velocity components + 3 magnetic vector potentials). The example highlights the difference between the high-level DSL system (`acGetDSLTaskGraph`) and the low-level task graph construction (`acGridBuildTaskGraph`), which gives fine-grained control over task ordering, field dependencies, and subregion decomposition.

# Directory Structure & File Descriptions

| File | Description |
| :--- | :--- |
| `CMakeLists.txt` | Build config: creates `taskgraph_example` executable from `main.cc`, links `astaroth_core` and `astaroth_utils`. |
| `main.cc` | MPI-based task graph demonstration. Initializes MPI, loads config, allocates and randomizes a host mesh, initializes GPU, loads mesh, builds a 3-operation task graph (halo exchange → boundary condition → compute), sets `AC_dt` to `FLT_EPSILON`, executes the graph for 3 iterations, and cleans up. |

# Compile-Time Requirements

| Setting | Value | Description |
| :--- | :--- | :--- |
| MPI | `REQUIRED` | Program requires MPI; falls back to error message if built without MPI support. |
| `AC_MPI_ENABLED` | Config-dependent | If disabled, program prints error and exits. |

Compile options: Inherited from `astaroth_core` (typically `-Wall -Wextra -Werror -Wdouble-promotion -Wfloat-conversion -Wshadow`).

# Compile-Time Options

| Macro | Default | Description |
| :--- | :--- | :--- |
| `AC_MPI_ENABLED` | Config-dependent | If disabled, program prints error and exits. |

# Input Parameters / Command-Line Interface

No command-line arguments. The program uses `AC_DEFAULT_CONFIG` via `acLoadConfig()` for mesh configuration.

Usage: `mpirun -np <num_processes> ./taskgraph_example`

# Program Flow

## 1. MPI Initialization
- `MPI_Init(NULL, NULL)`
- `MPI_Comm_size()` / `MPI_Comm_rank()` — get process count and rank.

## 2. Configuration & Host Mesh Setup (rank 0 only)
- `acLoadConfig(AC_DEFAULT_CONFIG, &info)` — load mesh configuration.
- `acHostMeshCreate(info, &mesh)` — allocate CPU-side `AcMesh`.
- `acHostMeshRandomize(&mesh)` — randomize mesh with pseudo-random values.

## 3. GPU Initialization
- `acGridInit(info)` — initialize GPU subsystem, create global grid variable and default task graph.
- `std::cout << "Loading mesh" << std::endl` — progress logging.

## 4. Mesh Load
- `acGridLoadMesh(STREAM_DEFAULT, mesh)` — transfer randomized mesh to GPU.

## 5. Task Graph Construction
- Define field array: `all_fields[] = {VTXBUF_LNRHO, VTXBUF_UUX, VTXBUF_UUY, VTXBUF_UUZ, VTXBUF_AX, VTXBUF_AY, VTXBUF_AZ}` — 7 fields (density + 3 velocities + 3 magnetic vector potentials, entropy commented out).
- Build task graph with 3 operations:
  ```cpp
  AcTaskGraph* hc_graph = acGridBuildTaskGraph({
      acHaloExchange(all_fields),                        // Step 1: Exchange halo data
      acBoundaryCondition(BOUNDARY_XYZ, BOUNDCOND_PERIODIC, all_fields),  // Step 2: Apply periodic BCs
      acCompute(KERNEL_solve, all_fields)                 // Step 3: Execute solve kernel
  });
  ```
- **Commented alternative:** A `shock_graph` with 5 operations (halo exchange → symmetric BC → shock1 → shock2 → solve) using a subset `shock_fields`.

## 6. Time Delta Setup
- `acGridLoadScalarUniform(STREAM_DEFAULT, AC_dt, FLT_EPSILON)` — set timestep parameter to `FLT_EPSILON`.
- `acGridSynchronizeStream(STREAM_DEFAULT)` — ensure `AC_dt` is loaded before execution.

## 7. Task Graph Execution
- `std::cout << "Executing taskgraph Halo->Compute for 3 iterations" << std::endl`
- `acGridExecuteTaskGraph(hc_graph, 3)` — execute the 3-operation graph 3 times.
- Each iteration: halo exchange → periodic BC → solve kernel, then repeat 3×.

## 8. Cleanup
- `std::cout << "Destroying grid" << std::endl`
- `acGridDestroyTaskGraph(hc_graph)` — free task graph resources.
- `acGridQuit()` — shutdown GPU subsystem.
- `MPI_Finalize()` — shutdown MPI.

# Task Graph Structure

## `hc_graph` (Halo-Compute Graph)

| Step | Operation | Description |
| :--- | :--- | :--- |
| 1 | `acHaloExchange(all_fields)` | Exchange ghost cell data between MPI processes for all 7 fields. |
| 2 | `acBoundaryCondition(BOUNDARY_XYZ, BOUNDCOND_PERIODIC, all_fields)` | Apply periodic boundary conditions on all 3D faces for all 7 fields. |
| 3 | `acCompute(KERNEL_solve, all_fields)` | Execute `KERNEL_solve` GPU kernel on all 7 fields. |

The graph creates tasks for each subregion in the domain and figures out dependencies between tasks. Operations execute in the order listed: halo exchange must complete before boundary conditions, which must complete before the compute kernel.

## `shock_graph` (Commented Alternative)

| Step | Operation | Description |
| :--- | :--- | :--- |
| 1 | `acHaloExchange(all_fields)` | Exchange halo data. |
| 2 | `acBoundaryCondition(BOUNDARY_XYZ, BOUNDCOND_SYMMETRIC, all_fields)` | Apply symmetric BCs (not periodic). |
| 3 | `acCompute(KERNEL_shock1, all_fields)` | Shock detection kernel 1. |
| 4 | `acCompute(KERNEL_shock2, shock_fields)` | Shock detection kernel 2 (subset of fields). |
| 5 | `acCompute(KERNEL_solve, all_fields)` | Main solve kernel. |

This alternative shows how to compose multi-stage pipelines with different boundary conditions and kernels, and how to use a subset of fields for specific kernels.

# API Functions Used

## GPU/Grid API

| Function | Description |
| :--- | :--- |
| `acGridInit(info)` | Initialize GPU subsystem, create global grid and default task graph. |
| `acGridQuit()` | Shutdown GPU subsystem. |
| `acGridLoadMesh(stream, mesh)` | Transfer mesh from host to GPU. |
| `acGridLoadScalarUniform(stream, param, value)` | Set a scalar parameter on GPU. |
| `acGridExecuteTaskGraph(graph, n)` | Execute task graph n iterations. |
| `acGridDestroyTaskGraph(graph)` | Free task graph resources. |
| `acGridSynchronizeStream(stream)` | Wait for stream completion. |
| `acGridBuildTaskGraph(ops[])` | Build task graph from array of `AcTaskDefinition` operations. |
| `acGridBuildTaskGraphWithBounds(ops[], start, end)` | Build task graph with explicit spatial bounds. |
| `acHaloExchange(fields[], n)` | Create halo exchange task definition. |
| `acBoundaryCondition(boundary, kernel, fields[], n)` | Create boundary condition task definition. |
| `acCompute(kernel, fields[], n)` | Create compute kernel task definition. |
| `STREAM_DEFAULT` | Alias for `STREAM_0`. |

## Host API

| Function | Description |
| :--- | :--- |
| `acLoadConfig(path, &info)` | Load configuration from file. |
| `acHostMeshCreate(info, &mesh)` | Allocate CPU-side `AcMesh`. |
| `acHostMeshRandomize(&mesh)` | Fill with pseudo-random values. |
| `acHostMeshDestroy(&mesh)` | Free CPU mesh (not called explicitly in example, relies on process termination). |

# Task Definition Structure (`AcTaskDefinition`)

The `AcTaskDefinition` struct contains all parameters needed to generate GPU tasks for an operation:

| Field | Description |
| :--- | :--- |
| `task_type` | Type of task (halo exchange, boundary condition, compute, ray update, scan, periodic ray). |
| `kernel_enum` | Kernel to execute (for compute tasks). |
| `boundary` | Boundary condition type (e.g., `BOUNDARY_XYZ`, `BOUNDARY_XY`). |
| `fields_in[] / num_fields_in` | Input fields. |
| `fields_out[] / num_fields_out` | Output fields. |
| `profiles_in / profiles_reduce_out / profiles_write_out` | Profile fields for statistical analysis. |
| `parameters[] / num_parameters` | Kernel parameters (e.g., `AC_dt`). |
| `load_kernel_params_func` | Function to load kernel parameters at runtime. |
| `fieldwise` | Whether the task operates field-by-field. |
| `outputs_in / outputs_out` | Reduce output handles. |
| `start / end` | Spatial bounds for the task. |
| `halo_sizes` | Size of halo/ghost zone. |
| `ray_direction` | Ray tracing direction (for ray update tasks). |
| `sending / receiving` | Whether this task sends/receives halo data. |
| `include_boundaries` | Whether to include boundary cells in computation. |
| `red_black_state` | Red-black ordering state (for iterative solvers). |

# Field/Vertex Buffer Handles Used

| Handle | Description |
| :--- | :--- |
| `VTXBUF_LNRHO` | Log density |
| `VTXBUF_UUX` | X velocity |
| `VTXBUF_UUY` | Y velocity |
| `VTXBUF_UUZ` | Z velocity |
| `VTXBUF_AX` | X vector potential (magnetic) |
| `VTXBUF_AY` | Y vector potential (magnetic) |
| `VTXBUF_AZ` | Z vector potential (magnetic) |

Total: 7 fields. `VTXBUF_ENTROPY` is commented out, suggesting this example uses an ideal MHD setup without entropy equation.

# Kernel References

| Kernel | Used In | Description |
| :--- | :--- | :--- |
| `KERNEL_solve` | `hc_graph` step 3 | Main solve kernel. |
| `KERNEL_shock1` | `shock_graph` step 3 (commented) | Shock detection step 1. |
| `KERNEL_shock2` | `shock_graph` step 4 (commented) | Shock detection step 2. |

# Boundary Condition References

| Constant | Description |
| :--- | :--- |
| `BOUNDARY_XYZ` | Apply on all 3D faces (X, Y, Z boundaries). |
| `BOUNDCOND_PERIODIC` | Periodic boundary condition. |
| `BOUNDCOND_SYMMETRIC` | Symmetric boundary condition (in commented shock_graph). |

# DSL vs Low-Level API Comparison

The example comments (lines 42-45) suggest quality-of-life features:
```cpp
// Miikka's note: this would be a good quality of life feature
// VertexBufferHandle all_fields = ALL_VERTEX_BUFFERS;
// or
// VertexBufferHandle all_fields = ALL_FIELDS;
```

This indicates the current API requires manually listing each `VertexBufferHandle`, while the DSL system (`acGetDSLTaskGraph`) abstracts this away.

| Aspect | DSL API (`acGetDSLTaskGraph`) | Low-Level API (`acGridBuildTaskGraph`) |
| :--- | :--- | :--- |
| Field specification | Automatic from DSL file | Manual array of `VertexBufferHandle` |
| Kernel definition | DSL file (`solver.ac`) | C++ `acCompute(kernel, fields)` |
| Boundary conditions | DSL `BoundConds` block | C++ `acBoundaryCondition()` |
| Custom ordering | DSL `ComputeSteps` blocks | Array order in `acGridBuildTaskGraph()` |
| Flexibility | Fixed DSL structure | Arbitrary task compositions |
| Use case | Physics solvers | Custom/prototype pipelines |

# Notable Observations

1. **Minimal example purpose:** This is a tutorial/demonstration program that shows the basic pattern of constructing and executing a task graph. It does not perform any verification or produce output — it only prints progress messages to `std::cout`.

2. **Entropy field is commented out:** The field array includes `VTXBUF_AX`, `VTXBUF_AY`, `VTXBUF_AZ` (Magnetic vector potentials) but explicitly comments out `VTXBUF_ENTROPY` with a trailing comment `//,  VTXBUF_ENTROPY}`. This suggests the example targets ideal MHD (no entropy equation).

3. **Manual field array instead of macro:** The developer comments (lines 42-45) express a desire for `ALL_VERTEX_BUFFERS` or `ALL_FIELDS` macros to avoid manually listing each field. This highlights a gap in the API's ergonomics.

4. **Fixed 3-iteration execution:** `acGridExecuteTaskGraph(hc_graph, 3)` runs exactly 3 iterations with no loop, timestep adaptation, or output. This keeps the example simple but means no useful physics is simulated.

5. **No host mesh cleanup:** Unlike most other samples (`stencil-loader`, `stress`, `standalone`), this program does not call `acHostMeshDestroy(&mesh)`. The host mesh is leaked on process termination.

6. **DT set to FLT_EPSILON:** `acGridLoadScalarUniform(STREAM_DEFAULT, AC_dt, FLT_EPSILON)` sets the timestep to machine epsilon, consistent with testing/verification patterns seen in other samples. No physical timestep is used.

7. **Synchronize after DT load:** `acGridSynchronizeStream(STREAM_DEFAULT)` after setting `AC_dt` ensures the scalar parameter is fully loaded before task graph execution begins. This prevents race conditions where the kernel might read an uninitialized `AC_dt`.

8. **Halo exchange implicitly uses MPI:** The `acHaloExchange()` call internally uses MPI point-to-point communication (with partitioned tag space, as noted in the comment on line 59-60) to exchange ghost cell data between processes.

9. **Subregion decomposition is automatic:** The comment on lines 52-53 states that `acGridBuildTaskGraph()` "generates tasks for each subregions in the domain and figures out the dependencies between the tasks." This means the developer does not manually specify how the domain is partitioned — the grid infrastructure handles this.

10. **Operation ordering is array-order:** The task graph executes operations in the order they appear in the `acGridBuildTaskGraph()` array argument. Halo exchange must come first to ensure ghost cells are valid before boundary conditions are applied, and boundary conditions must precede the compute kernel.

11. **Commented shock graph shows multi-stage composition:** The `shock_graph` example (lines 62-69) demonstrates that task graphs can include multiple compute kernels with different field subsets, mixed boundary conditions, and arbitrary staging — useful for complex physics like shock-capturing methods.

12. **No error checking on task graph build:** The program does not check the return value of `acGridBuildTaskGraph()` (it returns `AcTaskGraph*` which could be NULL on failure). This is a minor robustness gap.

13. **`acHostMeshDestroy` missing but `acGridDestroyTaskGraph` is called:** The cleanup is asymmetric — the host mesh is leaked but the task graph is properly destroyed. This suggests the example prioritizes demonstrating task graph usage over resource management best practices.

14. **STREAM_ALL vs STREAM_DEFAULT:** The example uses `STREAM_DEFAULT` throughout, while other samples (e.g., `stress`) use `STREAM_ALL` for synchronization. This suggests single-stream execution without explicit multi-stream synchronization.

15. **No DSL files:** Unlike `stress/DSL/solver.ac` or `plasma-meets-ai-workshop/`, this example has no DSL files. It is a pure C++ task graph program, demonstrating that the low-level API does not require the DSL compiler.

16. **MPI-only, no non-MPI fallback variant:** The `#if AC_MPI_ENABLED` guard at line 5 wraps the entire program body, with the fallback (lines 93-99) being the standard "rebuild with MPI" error message.

17. **`acGridBuildTaskGraph` uses C-style array syntax:** The function accepts `AcTaskDefinition ops[]` as a C-style array (or C++ std::vector), with the comment on lines 37-38 noting it "only works with C++ at the moment (the interface relies on templates for safety and array type deduction)."

18. **No output or diagnostic writing:** Unlike `stress` (which writes `stress11.dat`) or `standalone` (which writes `timeseries.ts`), this program produces no persistent output. It is a pure computational exercise.

19. **Progress logging via std::cout:** The example uses `std::cout << "..." << std::endl` for progress messages, making it suitable for interactive debugging but not for production use (where structured logging or no output would be preferred).

20. **Task graph lifetime:** The `hc_graph` is created once, executed 3 times, then destroyed. This is the typical pattern for repeated execution — the task graph is built once (which involves graph analysis, dependency resolution, and kernel compilation) and then reused.
