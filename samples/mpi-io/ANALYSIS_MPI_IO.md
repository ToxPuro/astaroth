# ⚠️ WARNING ⚠️
# **This is a LLM based analysis.**
# **THE BRANCH SHOULD NEVER BE MERGED TO master OR develop BRANCHES!**

# LLM Information
- **Model:** Qwen3.6-35B-A3B

# Overview
The `mpi-io` sample is an MPI-IO benchmark for Astaroth's mesh I/O subsystem. It measures the aggregate read and write bandwidth of storing and loading all vertex buffer fields to/from disk using MPI-IO, across configurable 3D grid dimensions. It runs `n` MPI processes (one per MPI rank), each writing its assigned domain chunk. The benchmark supports both centralized I/O (rank 0 handles all file operations) and distributed I/O (each rank writes its own data) modes. Output is a per-process CSV with timing and bandwidth measurements.

# Directory Structure & File Descriptions

| File | Description |
| :--- | :--- |
| `CMakeLists.txt` | Build configuration; compiles `main.c` into the `mpi-io` executable, linked against `astaroth_core` and `astaroth_utils`. Enables position-independent code. |
| `main.c` | MPI-IO benchmark driver (218 lines): initializes MPI and Astaroth grid, configures 3D mesh dimensions, performs optional CPU-GPU verification, creates per-process output directory, writes all fields via MPI-IO, reads them back, optionally verifies correctness, writes CSV results, and cleans up. |

# Compile-Time Requirements

The code has a chain of preprocessor guards:

```
AC_MPI_ENABLED → AC_INTEGRATION_ENABLED → main()
```

| Guard | Behavior if not defined |
| :--- | :--- |
| `AC_MPI_ENABLED` | Prints error message, returns `EXIT_FAILURE`. Astaroth must be built with `cmake -DMPI_ENABLED=ON`. |
| `AC_INTEGRATION_ENABLED` | Prints error message about missing fields (`VTXBUF_UUX`, etc.). Requires DSL source with `hostdefine AC_INTEGRATION_ENABLED`. |

# Compile-Time Options

| Option | Default | Description |
| :--- | :--- | :--- |
| `USE_DISTRIBUTED_IO` | 0 | If 1, uses distributed MPI-IO (each rank writes its own chunk); if 0, uses centralized I/O (rank 0 handles all). |
| `verify` | `false` (hardcoded) | If `true`, enables CPU-GPU load/store verification and disk read/write correctness checks. |

# Input Parameters

| Parameter | Position | Default | Description |
| :--- | :--- | :--- | :--- |
| `nx` | argv[1] | (required) | Grid dimension in x |
| `ny` | argv[2] | (required) | Grid dimension in y |
| `nz` | argv[3] | (required) | Grid dimension in z |
| `job_id` | argv[4] | 0 | Unique job identifier for output files |

Usage: `mpirun -np <num_processes> ./mpi-io <nx> <ny> <nz> [job_id]`

# Program Flow

1. **MPI Init**: `MPI_Init`, get rank (`pid`) and total processes (`nprocs`).

2. **Mesh Configuration**: Load `AC_DEFAULT_CONFIG`, override grid dimensions from command-line args via `acHostUpdateParams`.

3. **Grid Init**: `acGridInit(info)` — initializes the Astaroth grid infrastructure.

4. **Optional Verification** (`verify = true`):
   a. Create two host meshes (`model` and `candidate`), randomize both, apply periodic BCs on `model`.
   b. Load `model` to device, apply periodic BCs, store back to `candidate`.
   c. Verify `model` == `candidate` on rank 0 (`acVerifyMesh("CPU-GPU Load/store")`).

5. **Output Directory**: Each process creates its own directory `mpi-io-tmpdir-{job_id}-{pid}/`.

6. **Write Benchmark**:
   a. Timer starts.
   b. Loop over all `NUM_VTXBUF_HANDLES` vertex buffers, calling `acGridAccessMeshOnDiskSynchronous(handle, job_dir, label, ACCESS_WRITE)` for each field.
   c. Compute elapsed milliseconds, total bytes (`NUM_VTXBUF_HANDLES * acVertexBufferCompdomainSizeBytes(info)`), and bandwidth (bytes/second).

7. **Optional Scramble** (`verify = true`): Randomize `candidate` on device, load, then write to `field-tmp` on disk (overwriting the previous write) to catch false positives if MPI calls fail.

8. **Read Benchmark**:
   a. Timer resets.
   b. Loop over all `NUM_VTXBUF_HANDLES`, calling `acGridAccessMeshOnDiskSynchronous(handle, job_dir, label, ACCESS_READ)` for each field.
   c. Compute elapsed milliseconds and bandwidth.

9. **Optional Verification** (`verify = true`):
   a. Apply periodic BCs, store mesh back from device.
   b. Rank 0 verifies: `acVerifyMesh("MPI-IO disk read/write", model, candidate)`.

10. **CSV Output**: Each process writes its own CSV file `scaling-io-benchmark-job{id}-proc{pid}.csv` with: `pid, nprocs, write_ms, write_bw, read_ms, read_bw, distributed_io, nx, ny, nz`.

11. **Cleanup**: `acGridQuit()`, optionally destroy host meshes, remove output directory via `system("rm -r ...")`, `MPI_Finalize()`.

# Astaroth Grid APIs Used

| Function | Description |
| :--- | :--- |
| `acGridInit(info)` | Initialize grid infrastructure with mesh config. |
| `acGridQuit()` | Shutdown grid infrastructure. |
| `acGridLoadMesh(stream, mesh)` | Load mesh from host to device (for verification). |
| `acGridStoreMesh(stream, &mesh)` | Store mesh from device to host (for verification). |
| `acGridPeriodicBoundconds(stream)` | Apply periodic boundary conditions on device. |
| `acGridAccessMeshOnDiskSynchronous(handle, dir, label, ACCESS_READ/WRITE)` | **Core MPI-IO function**: Synchronously read/write a vertex buffer field to/from disk using MPI-IO. This is the benchmarked operation. |

# Astaroth Host Mesh APIs Used (Verification)

| Function | Description |
| :--- | :--- |
| `acHostMeshCreate(info, &mesh)` | Create host-side mesh. |
| `acHostMeshRandomize(&mesh)` | Randomize mesh field values. |
| `acHostMeshApplyPeriodicBounds(&mesh)` | Apply periodic BCs on host. |
| `acHostMeshDestroy(&mesh)` | Free host mesh. |

# Astaroth Utility APIs Used

| Function | Description |
| :--- | :--- |
| `acLoadConfig(config, &info)` | Load mesh configuration from `AC_DEFAULT_CONFIG`. |
| `acHostUpdateParams(&info)` | Update mesh info from modified `int3_params`. |
| `acVerifyMesh(name, model, candidate)` | Compare two meshes element-by-element. Returns `AC_SUCCESS` on match. |
| `acVertexBufferCompdomainSizeBytes(info)` | Get total bytes for the vertex buffer's computed domain. |

# MPI APIs Used

| Function | Description |
| :--- | :--- |
| `MPI_Init(NULL, NULL)` | Initialize MPI. |
| `MPI_Comm_rank(MPI_COMM_WORLD, &pid)` | Get rank ID. |
| `MPI_Comm_size(MPI_COMM_WORLD, &nprocs)` | Get total number of processes. |
| `MPI_Finalize()` | Shutdown MPI. |

# I/O Access Modes

| Mode | Description |
| :--- | :--- |
| `ACCESS_WRITE` | Write vertex buffer to disk via MPI-IO. |
| `ACCESS_READ` | Read vertex buffer from disk via MPI-IO. |

# Mesh Layout

The mesh dimensions come from command-line args (`nx`, `ny`, `nz`), set into `info.int3_params[AC_ngrid]`. The default config (`AC_DEFAULT_CONFIG`) provides the field definitions, but the grid size is runtime-configurable.

The total data volume per write: `NUM_VTXBUF_HANDLES * acVertexBufferCompdomainSizeBytes(info)`.

# Notable Observations

1. **Per-process output directories**: Each MPI rank creates its own directory `mpi-io-tmpdir-{job_id}-{pid}/`. This is unusual — typically MPI-IO benchmarks use a shared directory or have each rank write to a separate file within a shared directory. The per-process directories suggest that `acGridAccessMeshOnDiskSynchronous` either appends to a rank-specific file or writes to a path relative to the per-process directory.

2. **No rank-0 guard on I/O**: Unlike some MPI benchmarks that guard I/O operations to rank 0 only, this benchmark has all ranks performing the same I/O operations. The `// if (!pid)` comments suggest rank-0-only code was previously used but has been removed — all ranks now write/read their data.

3. **Synchronous I/O**: The function `acGridAccessMeshOnDiskSynchronous` is synchronous — it blocks until the entire field is written/read. This means there's no overlap between I/O and computation, which is appropriate for a pure I/O benchmark.

4. **Field iteration**: The benchmark iterates over `NUM_VTXBUF_HANDLES` fields, writing each with a label `"field-{i}"`. This measures the aggregate throughput of writing all vertex buffers (all fields) that a simulation would typically produce.

5. **Write then read (no concurrent I/O)**: The benchmark does a full write sweep followed by a full read sweep. There's no attempt to overlap reads and writes or to simulate a real workload pattern.

6. **Scalability testing**: The CSV output format includes `nx, ny, nz` and `nprocs`, enabling scaling analysis across different grid sizes and process counts. The file naming `scaling-io-benchmark-job{id}-proc{pid}.csv` suggests the intent is to compare scaling behavior.

7. **Bandwidth calculation**: Uses total bytes (`NUM_VTXBUF_HANDLES * compdomain size`) divided by total seconds. This gives aggregate bandwidth across all fields per process.

8. **Scramble step for verification**: If `verify=true`, the benchmark writes to `field-tmp` as a "scramble" step to catch false-positive MPI calls. This overwrites the previously written data on disk, ensuring the subsequent read actually reads from disk rather than from a cache.

9. **Cleanup via system()**: Uses `system("rm -r ...")` to remove the output directory after the benchmark. This runs on all ranks (no rank-0 guard), which could cause race conditions if directories overlap — but each rank has a unique directory name, so this is safe.

10. **`verify` is hardcoded false**: The `static const bool verify = false;` means correctness checking is disabled by default. To enable it, the source must be modified and recompiled.

11. **No error checking on MPI calls**: Unlike other Astaroth samples that use `ERRCHK_*` macros, MPI calls (`MPI_Init`, `MPI_Comm_rank`, etc.) have no error checking. MPI failures would produce undefined behavior rather than clear error messages.

12. **File naming for results**: Each rank produces a separate CSV file named by rank ID (`proc{pid}.csv`). This allows post-processing to collect all ranks' data. There's no aggregation step within the benchmark — results are per-process.

13. **Uses `acGridAccessMeshOnDiskSynchronous` instead of `acGridStoreFieldToFile`/`acGridLoadFieldToFile`**: The commented-out calls to `acGridStoreFieldToFile` and `acGridLoadFieldFromFile` suggest these are older APIs. The current `acGridAccessMeshOnDiskSynchronous` appears to be a more general interface that supports distributed I/O.

14. **Distributed vs. centralized I/O flag**: `USE_DISTRIBUTED_IO` is written to the CSV but has no functional impact on the I/O operations themselves (the I/O code path doesn't check this flag). It appears to be metadata for post-hoc analysis of what mode was configured.

15. **No timing for verification**: The CPU-GPU load/store and disk read/write verification steps are not timed — only the `acGridAccessMeshOnDiskSynchronous` operations are benchmarked.

16. **Single MPI communicator**: Uses `MPI_COMM_WORLD` for all communication. No custom communicators or subcommunicators for domain decomposition.

17. **Directory-based path separation**: Uses a directory `job_dir` rather than a single filename for I/O. This suggests `acGridAccessMeshOnDiskSynchronous` may write one file per field within the directory (e.g., `job_dir/field-0`, `job_dir/field-1`, etc.).
