# ⚠️ WARNING ⚠️
# **This is a LLM based analysis.**
# **THE BRANCH SHOULD NEVER BE MERGED TO master OR develop BRANCHES!**

# LLM Information
- **Model:** Qwen3.6-35B-A3B

# Overview
The `multikerneltest` sample is a minimal device-level kernel execution test for Astaroth's DSL (Domain Specific Language) runtime. It implements a 3D Fibonacci cellular automaton across a 64³ grid with NGHOST ghost cells, compiled from a DSL source file (`fibonacci.ac`) at compile time. The test launches three DSL kernels — `clear` (set all values to 0), `set` (set all values to 1), and `step` (each cell adds the value of its own neighbor, creating a Fibonacci-like propagation) — and runs 20 iterations of the `step` kernel, printing the minimum value after each iteration. The sample is designed to be built standalone (without MPI, without other samples or utils) by pointing `DSL_MODULE_DIR` and `PROGRAM_MODULE_DIR` to this directory.

# Directory Structure & File Descriptions

| File | Description |
| :--- | :--- |
| `CMakeLists.txt` | Build configuration; compiles `main.cc` into the `multikerneltest` executable, linked against `astaroth_core` only (no `astaroth_utils`). Does not enable MPI or utils. |
| `main.cc` | Device-level test driver (71 lines): creates a standalone GPU device, configures a 64³ grid, initializes DSL kernels (`clear`, `set`, `step`), runs 20 iterations, prints minimum per iteration, cleans up. |
| `fibonacci.ac` | DSL source file (33 lines): defines a single `ScalarField VTXBUF_FIBO` with four kernels — `clear` (zero initialize), `set` (initialize to 1), `step` (Fibonacci propagation: `fibo_out += value(fibo_in)`), and `solve` (dummy kernel). |

# Build Instructions

The sample must be built with custom CMake options to point the DSL compiler at this directory:

```bash
cmake -DDSL_MODULE_DIR=samples/multikerneltest \
      -DPROGRAM_MODULE_DIR=samples/multikerneltest \
      -DBUILD_SAMPLES=OFF -DBUILD_UTILS=OFF -DMPI_ENABLED=OFF ..
make -j
./multikerneltest
```

| Option | Value | Description |
| :--- | :--- | :--- |
| `DSL_MODULE_DIR` | `samples/multikerneltest` | Directory containing the `.ac` DSL source files. |
| `PROGRAM_MODULE_DIR` | `samples/multikerneltest` | Directory containing DSL program definitions. |
| `BUILD_SAMPLES` | `OFF` | Disable building other samples (this sample builds independently). |
| `BUILD_UTILS` | `OFF` | Disable building utility libraries. |
| `MPI_ENABLED` | `OFF` | No MPI support — single-device, no MPI. |

# Compile-Time Requirements

| Guard | Behavior if not defined |
| :--- | :--- |
| `AC_MPI_ENABLED` | Not checked. This sample is designed for non-MPI, single-device builds. |

# Compile-Time Options

None defined in source. Grid size is hardcoded as `static const int nn = 64`.

# Input Parameters / Command-Line Interface

| Parameter | Position | Default | Description |
| :--- | :--- | :--- | :--- |
| (none) | — | — | No command-line arguments. Grid size and iteration count are hardcoded. |

Usage: `./multikerneltest`

# Program Flow

## 1. Grid Configuration
a. `nn = 64` — global grid dimension (hardcoded).
b. `mm = nn + 2 * NGHOST` — extended grid dimension including ghost cells on both sides.
c. Set `info.int_params[AC_nx] = info.int_params[AC_ny] = info.int_params[AC_nz] = nn`.
d. `acHostUpdateParams(&info)` — process mesh configuration parameters.

## 2. Device Creation
a. `acDeviceCreate(0, info, &device)` — create a standalone GPU device on GPU ID 0. This is the key difference from grid-based samples: it uses the lower-level `acDevice` API directly, bypassing MPI and multi-node infrastructure.
b. `acDevicePrintInfo(device)` — print device configuration to stdout.

## 3. DSL Kernel Initialization
a. `acDevice_clear(device, STREAM_DEFAULT, start, end)` — execute DSL `clear()` kernel: set all `VTXBUF_FIBO` values to 0. Region: `(0,0,0)` to `(mm,mm,mm)`.
b. `acDeviceSwapBuffer(device, VTXBUF_FIBO)` — swap input/output buffers for `VTXBUF_FIBO`.
c. `acDevice_set(device, STREAM_DEFAULT, start, end)` — execute DSL `set()` kernel: set all `VTXBUF_FIBO` values to 1.
d. `acDeviceSwapBuffer(device, VTXBUF_FIBO)` — swap buffers again.

The sequence `clear → swap → set → swap` initializes the grid: first zero all memory, then set to 1. The swaps after each kernel are unusual — they ensure the output buffer becomes the input buffer for subsequent operations.

## 4. Iteration Loop (20 steps)
For each of 20 iterations:
a. `fibostep(device)` — defined as a local static function:
   1. `acDevice_step(device, STREAM_DEFAULT, start, end)` — execute DSL `step()` kernel: `fibo_out += value(fibo_in)` (Fibonacci propagation).
   2. `acDeviceSwapBuffer(device, VTXBUF_FIBO)` — swap input/output buffers.
   3. `acDeviceReduceScal(device, STREAM_DEFAULT, RTYPE_MIN, VTXBUF_FIBO, &val)` — compute minimum value across the entire grid.
   4. `printf("%f\n", (double)val)` — print the minimum to stdout.

## 5. Cleanup
`acDeviceDestroy(device)` — free device memory and destroy the device.

# DSL Source (fibonacci.ac) Breakdown

| Section | Description |
| :--- | :--- |
| `uniform ScalarField VTXBUF_FIBO` | Declares a uniform (constant spatial layout) scalar field named `VTXBUF_FIBO`. |
| `in ScalarField fibo_in(VTXBUF_FIBO)` | Declares `fibo_in` as an input field bound to `VTXBUF_FIBO`. |
| `out ScalarField fibo_out(VTXBUF_FIBO)` | Declares `fibo_out` as an output field bound to `VTXBUF_FIBO`. |
| `value(in ScalarField vertex)` | Helper function returning `vertex[vertexIdx]` — retrieves the scalar value at the current grid cell's index. |
| `Kernel void clear()` | Sets `fibo_out = 0` — zero-initializes the field. |
| `Kernel void set()` | Sets `fibo_out = 1` — initializes the field to 1. |
| `Kernel void step()` | `fibo_out += value(fibo_in)` — each cell adds its own current value to the output (Fibonacci-like: values propagate outward from their sources). |
| `Kernel void solve()` | Empty dummy kernel — commented in main as "hack" for integration. |

# Device API (acDevice*)

| Function | Description |
| :--- | :--- |
| `acDeviceCreate(id, info, &device)` | Create a standalone GPU device at the specified GPU ID with the given mesh config. Returns a `Device` handle. |
| `acDeviceDestroy(device)` | Free device memory and destroy the device. |
| `acDevicePrintInfo(device)` | Print device configuration (grid dims, fields, memory layout) to stdout. |
| `acDevice_clear(device, stream, start, end)` | Execute the DSL `clear()` kernel over the specified volume. |
| `acDevice_set(device, stream, start, end)` | Execute the DSL `set()` kernel over the specified volume. |
| `acDevice_step(device, stream, start, end)` | Execute the DSL `step()` kernel over the specified volume. |
| `acDevice_solve(device, stream, start, end)` | Execute the DSL `solve()` kernel (commented out, unused). |
| `acDeviceSwapBuffer(device, handle)` | Swap the input and output buffers for the specified vertex buffer handle. |
| `acDeviceSwapBuffers(device)` | Swap all input/output buffer pairs. |
| `acDeviceReduceScal(device, stream, reduction, vtxbuf, result)` | Compute a scalar reduction across all elements of a vertex buffer on this device. |
| `acDeviceReduceScalNoPostProcessing(device, stream, reduction, vtxbuf, result)` | Same as above but without post-processing (e.g., no sqrt for RMS). |

# Astaroth APIs Used

| Function | Description |
| :--- | :--- |
| `acHostUpdateParams(&info)` | Process mesh configuration parameters from `int_params` into the mesh info struct. |

# Preprocessor Constants / Definitions

| Constant | Value | Description |
| :--- | :--- | :--- |
| `nn` | 64 | Global grid dimension in each axis. |
| `mm` | `nn + 2 * NGHOST` | Extended grid dimension including ghost cells. |
| `NGHOST` | (DSL-defined) | Number of ghost cells per side used for stencil computations. |
| `STREAM_DEFAULT` | `STREAM_0` | Default CUDA/HIP stream. |
| `VTXBUF_FIBO` | DSL field handle | Single scalar field used for the Fibonacci automaton. |
| `RTYPE_MIN` | Reduction type | Minimum value reduction across the grid. |
| `start` | `(int3){0, 0, 0}` | Start of the compute volume. |
| `end` | `(int3){mm, mm, mm}` | End of the compute volume (includes ghost cells). |
| 20 | Iterations | Number of `fibostep` iterations. |

# Mesh Layout

The grid is `nn × nn × nn` = `64 × 64 × 64` = 262,144 cells, with `NGHOST` ghost cells on each side in each dimension. The extended compute volume is `mm = nn + 2 * NGHOST` per axis. All kernels operate on the full extended volume (start to end includes ghost cells), which ensures stencil computations have valid neighbor data at boundaries.

The grid is **not decomposed** across MPI ranks — this is a single-device test. All 262,144 cells reside on GPU 0.

# Output

After 20 iterations, the sample prints 20 lines to stdout, one minimum value per iteration:

```
1.000000
1.000000
1.000000
...
```

The minimum value represents the smallest value in the entire `VTXBUF_FIBO` field after each `step` kernel execution. Since the field is initialized to 1 and the `step` kernel adds the current value to the output (`fibo_out += value(fibo_in)`), values grow exponentially in a Fibonacci-like pattern, and the minimum should remain 1 (the cells that were never "reached" by the propagation keep their initial value of 1, or the swap pattern preserves at least one cell with value 1).

# Notable Observations

1. **Device API vs Grid API**: This sample uses the lower-level `acDevice*` API directly, bypassing the higher-level `acGrid*` API. This means no MPI, no domain decomposition, no halo exchange. It's the simplest possible entry point into Astaroth's DSL runtime — a single GPU device with no network communication.

2. **Standalone build**: The sample is designed to be built without `astaroth_utils` or MPI. It links only against `astaroth_core`. This makes it suitable for isolated kernel testing and development, without requiring a full MPI environment.

3. **DSL compilation at build time**: The `.ac` source file (`fibonacci.ac`) is compiled by Astaroth's DSL compiler during the build process (triggered by `DSL_MODULE_DIR` and `PROGRAM_MODULE_DIR` CMake options). The generated C++ code for `acDevice_clear`, `acDevice_set`, `acDevice_step`, etc., is linked into the executable. No runtime compilation occurs.

4. **Double buffer swap pattern**: After every kernel, `acDeviceSwapBuffer(device, VTXBUF_FIBO)` is called. This swaps the input and output buffers for the field. The `step` kernel reads from `fibo_in` and writes to `fibo_out` — after the swap, what was the output becomes the input for the next iteration. This is the standard dual-buffer pattern for stencil computations that need to avoid in-place overwrites.

5. **`clear → swap → set → swap` initialization**: The initialization sequence is unusual. `clear` sets everything to 0, then a swap makes the zeroed buffer the input. Then `set` sets everything to 1 (writing to the output buffer), then another swap makes the ones buffer the input. The net effect is the same as just calling `set` once — the initial `clear` is unnecessary unless there's memory contamination from a previous run (but this is a standalone process, so no such contamination exists).

6. **`VTXBUF_FIBO` as both input and output**: The DSL declares `fibo_in(VTXBUF_FIBO)` and `fibo_out(VTXBUF_FIBO)` bound to the **same** vertex buffer handle. This means the field is used as both input and output, requiring the buffer swap to differentiate read/write phases. If both pointed to different handles, no swap would be needed.

7. **No ghost cell boundary logic**: The DSL kernels (`clear`, `set`, `step`) operate on the full extended volume including ghost cells. There's no explicit boundary condition logic in the DSL code — the ghost cells are presumably maintained by other Astaroth infrastructure (halo exchange, boundary condition kernels). This test doesn't exercise any BC logic.

8. **`solve()` is a dummy**: The `solve()` kernel is empty and commented out in main.cc. The comment says "dummy kernel for integration, hack" — it was likely added to satisfy some DSL compilation requirement that all referenced kernels must be defined, even if unused.

9. **`acDeviceReduceScal` after every iteration**: After each `step` kernel, the sample computes the minimum value across the grid and prints it. This provides a simple correctness signal — if the Fibonacci propagation is working correctly, the minimum should remain stable. A crashing kernel or buffer corruption would cause the minimum to become NaN or an extreme value.

10. **No error checking**: There are no `WARNCHK_ALWAYS` or `ERRCHK_ALWAYS` calls on any of the `acDevice*` API calls. A failed kernel launch or reduction would produce undefined behavior rather than a clear error message.

11. **Hardcoded grid size**: `nn = 64` is a `static const` — there's no way to change the grid size at runtime. This limits testing to a single resolution. For a more flexible test, this could be made a command-line argument.

12. **`mm` extends beyond `nn`**: The compute volume goes from `0` to `mm` in each axis, where `mm = nn + 2 * NGHOST`. This means the test exercises stencil operations in the ghost cell regions. If the DSL kernels don't properly handle ghost cells, this would be caught here.

13. **No host-device data transfer**: The sample never copies data from GPU to host (except via the scalar reduction, which returns a single `AcReal`). There's no `acDeviceStoreMesh` or equivalent — the entire test runs GPU-side.

14. **20 iterations is arbitrary**: The iteration count is a hardcoded `20`. This is enough to observe exponential growth in a Fibonacci-like pattern, but there's no stated criterion for "correct" output values.

15. **No comparison against CPU reference**: Unlike `mpitest`, this sample has no CPU-side reference implementation to compare against. It relies solely on the printed minimum values as a correctness signal.

16. **`acHostUpdateParams` without `acLoadConfig`**: The sample calls `acHostUpdateParams(&info)` directly after setting `info.int_params`, without first calling `acLoadConfig`. This works because `acHostUpdateParams` only processes parameters that are already set — it doesn't require the full config from the DSL. This is appropriate for a minimal standalone test.

17. **`fibostep` is a free function, not a method**: `fibostep` is a plain C-style function defined at file scope, not a class method or lambda. It takes the `Device` as a parameter. This is consistent with the procedural design of the `acDevice*` API.

18. **`printf` with `(double)` cast**: The reduction result (`AcReal`, which may be `float` or `double` depending on `DOUBLE_PRECISION` build option) is explicitly cast to `double` before `printf`. This is correct practice for `printf` — `%f` expects a `double`, and passing a `float` directly would produce undefined output due to default argument promotion.

19. **No cleanup on error**: If `acDeviceCreate` fails, there's no cleanup — the program simply continues with an uninitialized `device` pointer. This is a potential crash vector.

20. **`NGHOST` from DSL compilation**: The value of `NGHOST` is defined by the DSL compiler, not in this source file. The sample implicitly depends on the DSL compiler defining `NGHOST` (e.g., typically 1 or 2 for a first-order stencil). The exact value determines the extended grid size `mm` and thus the total number of elements processed by each kernel.
