# ⚠️ WARNING ⚠️
# **This is a LLM based analysis.**
# **THE BRANCH SHOULD NEVER BE MERGED TO master OR develop BRANCHES!**

# LLM Information
- **Model:** Qwen3.6-35B-A3B

# Overview
The `benchmark-thrust` sample is a CUDA Thrust library performance benchmark embedded within the Astaroth ecosystem. It serves a dual purpose:

1. **Thrust baseline benchmark** — Measures the raw performance of `thrust::reduce` on `thrust::device_vector` arrays, providing a CUDA-level reference for reduction throughput on the target GPU hardware.
2. **Astaroth buffer array cross-reduce benchmark** — Benchmarks Astaroth's `acMapCrossReduce` operation over a `VertexBufferArray` (VBA) and `ProfileBufferArray` (PBA), comparing it against the raw Thrust reduction performance.

The sample is written in **CUDA C++** (`.cu` file), directly linking to Thrust for low-level GPU array operations alongside the Astaroth runtime.

# Directory Structure & File Descriptions

| File | Description |
| :--- | :--- |
| `CMakeLists.txt` | Build configuration; compiles `main.cu` into the `benchmark-thrust` executable, linked against `astaroth_core` and `astaroth_utils`. |
| `build.sh` | Build script that sets Astaroth-specific CMake options including `CUDA_ARCHITECTURES=61`, enables MPI, disables single-pass integration, and configures DSL module directories. |
| `main.cu` | The benchmark implementation: Thrust `device_vector` reduction benchmark + Astaroth `acMapCrossReduce` benchmark with profiler instrumentation. |

# Build Script (`build.sh`)

The build script enforces several configuration options:

| Option | Value | Description |
| :--- | :--- | :--- |
| `OPTIMIZE_MEM_ACCESSES` | `ON` | Optimize memory access patterns. |
| `MPI_ENABLED` | `ON` | Enable MPI support. |
| `SINGLEPASS_INTEGRATION` | `OFF` | Disable single-pass integration mode. |
| `BUILD_STANDALONE` | `OFF` | Do not build standalone sample. |
| `BUILD_MHD_SAMPLES` | `OFF` | Do not build MHD samples. |
| `BUILD_SAMPLES` | `OFF` | Do not build other samples. |
| `DSL_MODULE_DIR` | `$ASTAROTH/samples/tfm/mhd` | Path to DSL modules for MHD. |
| `PROGRAM_MODULE_DIR` | `$ASTAROTH/samples/benchmark-thrust` | Program module directory. |
| `CUDA_ARCHITECTURES` | `61` | Target GPU architecture (Volta V100, etc.). |

Prerequisite: The `ASTAROTH` environment variable must point to the Astaroth source root.

# Environment Variables & Compile-Time Constants

| Constant | Description |
| :--- | :--- |
| `STENCIL_ORDER` | Astaroth stencil order, used to compute padding radius: `radius = STENCIL_ORDER / 2`. |

# Program Flow

## 0. Profiler Setup
`cudaProfilerStop()` — Ensure NVIDIA Nsight profiler is disabled during benchmarking.

## 1. Parse Arguments

| Argument | Default | Description |
| :--- | :--- | :--- |
| `<nx> <ny> <nz>` | `32, 32, 32` | Grid dimensions. |

The padded dimensions are computed as:
```
radius = STENCIL_ORDER / 2
mx = nx + 2 * radius
my = ny + 2 * radius
mz = nz + 2 * radius
```

These padded dimensions are printed and used for both benchmarks.

## 2. Thrust Reduction Benchmark

### 2a. Setup
1. `cudaDeviceSynchronize()` — Clear any prior GPU work.
2. For each of the `mz` "slices" (k-dimension):
   - Allocate a `thrust::device_vector<double>` of size `mx * my`, initialized to `1.0`.
3. Allocate a `thrust::device_vector<double> results` of size `mz`.
4. Perform one reduction per slice before timing:
   ```cpp
   for (size_t k = 0; k < mz; ++k)
       results[k] = thrust::reduce(inputs[k].begin(), inputs[k].end());
   ```
   This ensures kernel launches are warmed up and device memory is allocated.

### 2b. Timing
1. `cudaDeviceSynchronize()` — Synchronize before timing.
2. `timer_reset(&t)` — Start high-resolution timer.
3. For each of the `mz` slices:
   ```cpp
   results[k] = thrust::reduce(inputs[k].begin(), inputs[k].end());
   ```
4. `cudaDeviceSynchronize()` — Synchronize after timing.
5. `timer_diff_print(t)` — Print elapsed time.

Each reduction computes the sum of all elements in a `mx * my`-sized array on the GPU using Thrust's parallel reduce. With `nx=ny=nz=32` and `STENCIL_ORDER=6` (typical), `mx*my = 44 * 44 = 1936` elements per slice, and `mz = 44` slices.

## 3. Astaroth Buffer Array Benchmark

### 3a. Device & Mesh Setup
1. `cudaSetDevice(0)` — Select the first GPU device.
2. `acLoadConfig(AC_DEFAULT_CONFIG, &info)` — Load mesh configuration.
3. `acSetMeshDims(nx, ny, nz, &info)` — Set mesh dimensions.
4. `acLoadMeshInfo(info, 0)` — Load mesh information for device 0.

### 3b. Random Number Initialization
1. `seed = 12345`, `pid = 0`.
2. `count = acVertexBufferCompdomainSize(info)` — Get the number of elements in the computation domain.
3. `acRandInitAlt(seed, count, pid)` — Initialize Astaroth's random number generator.

### 3c. Mesh Dimensions & Buffer Arrays
1. `dims = acGetMeshDims(info)` — Get mesh dimensions.
2. `vba = acVBACreate(mx, my, mz)` — Create a Vertex Buffer Array with padded dimensions.
3. `acVBAReset(0, &vba)` — Reset the buffer array to zero.
4. `acLaunchKernel(randomize, 0, dims.n0, dims.n1, vba)` — Launch the `randomize` kernel to fill the VBA with random data (the kernel is expected to be provided by the compiled DSL library).
5. `acVBASwapBuffers(&vba)` — Swap in/out buffers.
6. `pba = acPBACreate(mz)` — Create a Profile Buffer Array of size `mz`.
7. `scratchpad = acBufferArrayCreate(14, mx * my)` — Create a scratchpad buffer array with 14 buffers of size `mx * my` each.

### 3d. Cross-Reduce Benchmark
1. `cudaProfilerStart()` — Enable CUDA profiling.
2. `cudaDeviceSynchronize()` — Clear GPU.
3. `timer_reset(&t)` — Start timer.
4. `acMapCrossReduce(vba, 0, scratchpad, pba)` — Perform the cross-reduce operation:
   - Iterates over the z-dimension of the VBA.
   - For each slice, extracts a cross-section (mapped region) into the scratchpad.
   - Computes a reduction over the mapped region and writes the result to `pba`.
5. `cudaDeviceSynchronize()` — Wait for completion.
6. `timer_diff_print(t)` — Print elapsed time.
7. `cudaProfilerStop()` — Disable profiling.
8. `cudaDeviceSynchronize()` — Final synchronization.

The `acMapCrossReduce` operation is the Astaroth-level equivalent of the Thrust `reduce` benchmark above, but includes:
- **Buffer mapping** (`acMapCross` internally) — selecting a 2D cross-section from the 3D VBA.
- **Scratchpad buffers** — intermediate storage for mapped data.
- **Reduction into profile buffers** — results written to the PBA rather than a device vector.

### 3e. Error Check & Cleanup
1. `ERRCHK_CUDA_KERNEL_ALWAYS()` — Check for CUDA kernel errors.
2. `acRandQuit()` — Destroy the random number generator.
3. `acBufferArrayDestroy(&scratchpad)` — Free scratchpad buffers.
4. `acPBADestroy(&pba)` — Destroy profile buffer array.
5. `acVBADestroy(&vba)` — Destroy vertex buffer array.

# Key Functions & Operations

## Thrust Operations

| Function | Description |
| :--- | :--- |
| `thrust::device_vector<T>` | CUDA device-resident vector container. |
| `thrust::reduce(first, last)` | Parallel sum reduction over a device range. |
| `thrust::square<T>` | Unary functor that returns `x * x` (defined but **not used** in the current code). |

## Astaroth Buffer Array Operations

| Function | Description |
| :--- | :--- |
| `acVBACreate(mx, my, mz)` | Create a vertex buffer array with padded dimensions. |
| `acVBAReset(index, &vba)` | Reset buffer array to zero. |
| `acLaunchKernel(kernel, stream, nx, ny, vba)` | Launch a DSL kernel with the VBA context. |
| `acVBASwapBuffers(&vba)` | Swap in/out buffers. |
| `acPBACreate(mz)` | Create a profile buffer array of size `mz`. |
| `acBufferArrayCreate(count, size)` | Create a buffer array with `count` buffers of `size` elements each. |
| `acMapCrossReduce(vba, index, scratchpad, pba)` | Map 2D cross-sections from the VBA, reduce each, and write results to PBA. |
| `acBufferArrayDestroy(&buf)` | Destroy a buffer array. |
| `acPBADestroy(&pba)` | Destroy a profile buffer array. |
| `acVBADestroy(&vba)` | Destroy a vertex buffer array. |

# Data Layout

```
VBA: (mx, my, mz) padded vertex buffers
         ┌──────────────────────────┐
  k=0    │  mx × my  random values  │
  k=1    │  mx × my  random values  │
  ...    │  ...                     │
  k=mz-1 │  mx × my  random values  │
         └──────────────────────────┘

PBA: (mz) reduced values — one per slice

Scratchpad: 14 × (mx × my) temporary buffers for mapped cross-sections
```

The Thrust benchmark operates on flat `mx * my` device vectors (one per z-slice), while the Astaroth benchmark operates on the structured 3D VBA with explicit cross-section mapping and scratchpad-based intermediate storage.

# Timing Mechanism

Both benchmarks use the same high-resolution timer (`timer_hires.h`) wrapped around CUDA synchronization:
- `cudaDeviceSynchronize()` — Ensures GPU completion before and after timing.
- `timer_reset(&t)` / `timer_diff_print(t)` / `timer_diff_nsec(t)` — Timer manipulation.

# Output

The benchmark prints two timing results:
1. **Thrust reduce** — Elapsed time for `mz` reductions over `mx * my` elements each.
2. **Astaroth `acMapCrossReduce`** — Elapsed time for the same logical operation through Astaroth's buffer array abstraction.

This allows direct comparison between raw CUDA Thrust performance and the overhead introduced by Astaroth's VBA/PBA layer.

# Key Dependencies
- `acc_runtime.h` — ACC runtime API (for `acLaunchKernel`, `acVBACreate`, etc.).
- `timer_hires.h` — High-resolution timer.
- `thrust/device_vector.h` — Thrust device vector container.
- `thrust/functional.h` — Thrust functors (`square`).
- `thrust/transform_reduce.h` — Thrust transform-reduce algorithms.
- `astaroth_utils.h` — Astaroth utility functions (`acLoadConfig`, `acSetMeshDims`, `acRandInitAlt`, etc.).
- `errchk.h` — CUDA error-checking macros.
- CUDA runtime (`cudaProfilerStop`, `cudaSetDevice`, `cudaDeviceSynchronize`).
