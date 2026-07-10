# ⚠️ WARNING ⚠️
# **This is a LLM based analysis.**
# **THE BRANCH SHOULD NEVER BE MERGED TO master OR develop BRANCHES!**

# LLM Information
- **Model:** Qwen3.6-35B-A3B

# Overview
The `microbenchmark` sample is a GPU microbenchmark suite for measuring memory bandwidth in 1D stencil computations. It measures the effective memory bandwidth of a generic `y[i] = sum(x[i + r])` stencil (sum of radius `r` neighborhood) across varying stencil radii (0 to 1024), problem sizes (default 128M elements), and implementation strategies. It features an autotuning phase that sweeps threadblock sizes to find optimal configurations, a verification phase against a CPU model kernel, and produces CSV output suitable for plotting bandwidth vs. working-set-size curves. Supports both NVIDIA CUDA and AMD HIP (via Roctracer), with single/double precision via `AC_DOUBLE_PRECISION`.

# Directory Structure & File Descriptions

| File | Description |
| :--- | :--- |
| `CMakeLists.txt` | Build configuration; compiles `microbenchmark.cu` into the `microbenchmark` executable, linked against `acc-runtime-headers` and `ac_cuda_wrappers`. Configures `USE_SMEM`, `MAX_THREADS_PER_BLOCK`, and `MB_IMPLEMENTATION` compile options. HIP mode adds `roctracer64`. |
| `microbenchmark.cu` | Main benchmark driver (734 lines): device info printing, autotuning, verification, and benchmarking with CSV output. |
| `array.h` | Lightweight array abstraction: creates host/device arrays, randomizes data, destroys. Uses `AC_DOUBLE_PRECISION` to select `float` or `double`. |

# Data Structures

| Structure | Description |
| :--- | :--- |
| `Array` | Opaque array wrapper: `{ length, data, on_device }`. `data` is `real*` (float or double). `on_device` determines allocation/frees (`cudaMalloc`/`cudaFree` vs `malloc`/`free`). |
| `KernelConfig` | Autotuned kernel launch parameters: `{ array_length, domain_length, pad, radius, stride, tpb, bpg, smem }`. |
| `Timer` | Wrapper around `struct timespec` from `<time.h>`, used for high-resolution CPU timing. |

# Compute Implementations

The benchmark supports three implementation modes controlled by `MB_IMPLEMENTATION`:

| Mode | Description | Templated? | EPT |
| :--- | :--- | :--- | :--- |
| 1 | Classic 1 element per thread (EPT=1) | No | 1 |
| 2 | 4 elements per thread (EPT=4) | No | 4 |
| 3 | Templated unrolling (compile-time `#pragma unroll`) | Yes | 1 |

Mode 2 is designed to "get around compute bound on NVIDIA" by increasing arithmetic intensity per thread.

## Stencil Kernel Variants (two styles)

### 1. Shared Memory (USE_SMEM=1)

Threads cooperatively load the halo region into `extern __shared__ real smem[]`, then each thread accesses the shared memory stencil region. With EPT > 1, each thread manages a vector `real tmp[ELEMS_PER_THREAD]`.

```
base_idx = blockIdx.x * blockDim.x * EPT + pad - radius
For each tile sid in [threadIdx.x, blockDim.x*EPT, stride blockDim.x):
    smem[sid] = in.data[sid + base_idx]    // Load halo + interior
__syncthreads()
For i in [-radius, radius, stride]:
    For each block in [0, EPT):
        tmp[block] += smem[radius + threadIdx.x + i + block * blockDim.x]
Store: out.data[pad + base_tid + block * blockDim.x] = tmp[block]
```

Shared memory size: `smem = (tpb * EPT + 2 * radius) * sizeof(real)`

### 2. Direct Global Memory (USE_SMEM=0)

Threads directly load from global memory for each stencil tap. No shared memory overhead. Same EPT vectorization pattern.

```
base_tid = threadIdx.x + blockIdx.x * blockDim.x * EPT
For i in [-radius, radius, stride]:
    For each block in [0, EPT):
        tmp[block] += in.data[pad + base_tid + block * blockDim.x + i]
Store: out.data[pad + base_tid + block * blockDim.x] = tmp[block]
```

Shared memory size: 0

### 3. Templated Unrolling (MB_IMPLEMENTATION=3, USE_SMEM=0)

C++ template parameterizes radius at compile time. Each stencil coefficient is a separate template instantiation. The switch statement dispatches to the correct `<radius, stride>` specialization. Uses `#pragma unroll` for the stencil loop.

Supported radii: 0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024 (2048 and 4096 commented out).

```cpp
template <int radius, int stride>
__global__ void kernel(const size_t domain_length, const size_t pad, const Array in, Array out) {
    const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    real tmp = 0.0;
#pragma unroll
    for (int i = -radius; i <= radius; i += stride)
        tmp += in.data[tid + pad + i];
    out.data[tid + pad] = tmp;
}
```

## CPU Model Kernel

Reference CPU implementation used for verification:

```c
void model_kernel(size_t domain_length, size_t pad, int radius, int stride,
                  Array in, Array out) {
    for (int tid = 0; tid < domain_length; ++tid) {
        real tmp = 0.0;
        for (int i = -radius; i <= radius; i += stride)
            tmp += in.data[tid + pad + i];
        out.data[tid + pad] = tmp;
    }
}
```

# Benchmark Pipeline

The execution follows a 4-phase pipeline:

1. **Device Info Print**: Calls `printDeviceInfo(0)` to show GPU properties (name, compute capability, clock rates, memory size, cache info, L2 size, shared memory per block, warp size).

2. **Autotuning** (`autotune()`): Sweeps threadblock sizes `tpb` from minimum transaction alignment (32/sizeof(real)) to `maxThreadsPerBlock` (optionally capped by `MAX_THREADS_PER_BLOCK`), computes `bpg = ceil(domain_length / tpb / EPT)`, validates `bpg * tpb * EPT == domain_length` (exact tiling requirement), and skips if shared memory exceeds device limits. Each configuration runs 3 kernel launches timed with CUDA events. Records fastest config.

3. **Verification** (`verify()`): Runs `model_kernel` on CPU host, launches GPU kernel with autotuned config, copies result back to host, and compares element-by-element. Allows up to 100 failures before exiting.

4. **Benchmarking** (`benchmark()`): Runs `num_samples` iterations with a POSIX high-resolution timer (`timer_diff_nsec` from `clock_gettime`). Computes effective bandwidth: `bytes / seconds` where `bytes = sizeof(real) * (2 * domain_length + 2 * radius / stride)`. Outputs CSV with per-iteration statistics. Profiling enabled via `cudaProfilerStart()`/`Stop()` (or `hip` conditional).

# Benchmark Parameters

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `domain_length` | 128M | Computational domain size in elements (128 * 1024 * 1024 / sizeof(real)) |
| `radius` | 1 | Stencil half-width (0, 1, 2, 4, 8, ..., 1024) |
| `stride` | 1 | Stencil step (must be 1, enforced by `ERRCHK_ALWAYS`) |
| `jobid` | 0 | Job identifier for output CSV filename |
| `num_samples` | 100 | Number of benchmark iterations |
| `salt` | 42 | Random seed modifier |

## Derived Parameters

| Parameter | Formula | Description |
| :--- | :--- | :--- |
| `pad` | First `radius + n` where `(radius + n) * sizeof(real) % 256 == 0` | 256-byte alignment padding (capped at 10000) |
| `array_length` | `pad + domain_length + radius` | Total array length including padding on both sides |
| `seed` | `12345 + salt + (1 + domain_length + radius + stride + jobid + num_samples) * time(NULL)` | Random seed for array initialization |
| `minimum_transaction_size_in_elems` | `32 / sizeof(real)` | Minimum `tpb` alignment (32 bytes = minimum L1 cache transaction) |

## Compile-Time Options

| Option | Default | Description |
| :--- | :--- | :--- |
| `USE_SMEM` | 0 | Enable shared memory implementation |
| `MAX_THREADS_PER_BLOCK` | 0 | Override device max threads per block (0 = device default) |
| `MB_IMPLEMENTATION` | 1 | Implementation mode (1, 2, or 3) |
| `AC_DOUBLE_PRECISION` | 0 | Single precision (float) if 0, double precision (double) if 1 |

# Profiling

### CUDA (NVIDIA)

```bash
# Enable nvprof/cuda profiler in benchmark
# Output stored via cudaProfilerStart()/cudaProfilerStop()
./microbenchmark 33554432 0   # r=0 bandwidth test

# Or use nvprof externally:
nvprof ./microbenchmark 33554432 0
```

### HIP/ROCm (AMD)

```bash
cmake -DUSE_HIP=ON .. && make -j
rocprof --trace-start off -i rocprof-input-metrics.txt ./microbenchmark
```

`rocprof-input-metrics.txt` format:
```
# Perf counters group 1
pmc : Wavefronts VALUInsts SALUInsts SFetchInsts
# Perf counters group 2
pmc : TCC_HIT[0], TCC_MISS[0], TCC_HIT_sum, TCC_MISS_sum
# Perf counters group 3
pmc: L2CacheHit MemUnitBusy LDSBankConflict
# Filter by dispatches range, GPU index and kernel names
range: 0 : 16
gpu: 0 1 2 3
kernel: singlepass_solve
```

# Bandwidth Calculation

Effective bandwidth formula:

```
bytes = sizeof(real) * (2 * domain_length + 2 * radius / stride)
seconds = milliseconds / 1000
bandwidth = bytes / seconds
```

- `2 * domain_length`: one read of the input array + one write of the output array
- `2 * radius / stride`: extra reads for the stencil halo region (read-only, not written)
- Total memory traffic per kernel launch: `bytes`
- For `num_samples` iterations: total = `bytes * num_samples`

Example (double precision, r=1):
- `sizeof(real) = 8 bytes`
- `bytes = 8 * (2 * 134217728 + 2 * 1) = 2,147,483,664 bytes ≈ 2 GiB`

# Key Astaroth C APIs Used

| Function | Description |
| :--- | :--- |
| `acDeviceGetAttribute(&val, attr, device_id)` | Query device attributes (clock rate, compute mode) via Astaroth wrapper. |
| `ac_cuda_wrappers` | Linked library providing `acDeviceGetAttribute` and other CUDA API wrappers. |

# Standard C/CUDA APIs Used

| Function | Description |
| :--- | :--- |
| `cudaMalloc`, `cudaFree` | Device memory allocation/deallocation. |
| `cudaMemcpy` | Host-device data transfer. |
| `cudaMemcpyHostToDevice`, `cudaMemcpyDeviceToHost` | Explicit transfer direction. |
| `cudaGetDeviceProperties` | Query device properties (name, caps, memory, cache, shared mem, warp size). |
| `cudaGetDevicePCIBusId` | Get PCI bus ID string. |
| `cudaMemGetInfo` | Query free/total device memory. |
| `cudaEventCreate`, `cudaEventRecord`, `cudaEventSynchronize`, `cudaEventElapsedTime`, `cudaEventDestroy` | CUDA event-based timing. |
| `cudaDeviceSynchronize` | Block until all launched kernels complete. |
| `cudaProfilerStart`, `cudaProfilerStop` | Start/stop NVIDIA Nsight profiler. |
| `cudaGetLastError`, `cudaDeviceGetAttribute` | Error checking and attribute queries. |
| `hip/hip_runtime.h`, `roctracer/*` | AMD HIP/ROCm profiling (conditional on `AC_USE_HIP`). |

# Timing Methods

Two timing methods are present in the code (one active, one commented):

1. **Active: POSIX high-resolution timer** (`timer_hires.h`):
   ```c
   Timer t;
   timer_reset(&t);
   kernel<<<...>>>();
   cudaDeviceSynchronize();
   const long double milliseconds = timer_diff_nsec(t) / 1e6l;
   ```
   Uses `clock_gettime(CLOCK_REALTIME)` for sub-microsecond resolution.

2. **Commented out: CUDA events**:
   ```c
   cudaEventRecord(tstart);
   kernel<<<...>>>();
   cudaEventRecord(tstop);
   cudaEventSynchronize(tstop);
   cudaEventElapsedTime(&milliseconds, tstart, tstop);
   ```

# Notable Observations

1. **Exact tiling requirement**: The autotuner enforces `bpg * tpb * EPT == domain_length` — the grid must perfectly cover the domain with no partial work-items. This simplifies boundary handling but constrains valid `tpb` values.

2. **256-byte alignment padding**: The `get_pad()` function ensures `pad * sizeof(real) % 256 == 0`, aligning the computational domain to 256-byte boundaries for optimal coalesced memory accesses.

3. **Stride enforced to 1**: Despite accepting `stride` as a parameter, the code enforces `stride == 1` (`ERRCHK_ALWAYS(stride == 1)`), skipping every other stencil coefficient. This appears to be a planned feature not yet implemented.

4. **Shared memory with EPT**: When `USE_SMEM=1` and `ELEMS_PER_THREAD > 1`, shared memory is sized as `(tpb * EPT + 2 * radius) * sizeof(real)`. The extra `EPT` factor accounts for the expanded thread work per block. Without this, shared memory bank conflicts would be severe.

5. **AMD performance pitfall note**: Template-based implementations (MB_IMPLEMENTATION=3) are specifically noted as having "abysmal performance on AMD" — the commented-out global `#define USE_TEMPLATED_IMPLEMENTATION` avoids this. Mode 2 (EPT=4, non-templated) is the recommended workaround.

6. **Warmup kernel**: A single kernel launch is executed before timing begins to avoid cold-start overhead (cache warming, GPU clock ramp-up).

7. **Profiling overhead**: The benchmark starts the CUDA profiler (`cudaProfilerStart()`) during the benchmarking phase. Profiling adds overhead, so results with profiling enabled are likely not representative of raw performance.

8. **Default problem size**: The default `domain_length` is `128M / sizeof(real)` elements — 128 million `float` values or 64 million `double` values. This ensures the working set is large enough to exercise GPU cache hierarchies effectively.

9. **Memory bandwidth reference**: The `printDeviceInfo` function computes "Peak Memory Bandwidth" as `2 * memClockRate * memoryBusWidth / (8 * GiB)`, which is the standard double-data-rate calculation. This provides a useful baseline for comparing effective bandwidth.

10. **CSV output format**: Output filename is `microbenchmark-{jobid}-{seed}.csv` with columns: `implementation, maxthreadsperblock, domainlength, radius, stride, milliseconds, effectivebandwidth, tpb, jobid, seed, iteration, double_precision, mbimplementation`.

11. **Array randomization**: Uses a simple `(real)rand() / RAND_MAX` function, noted as "Not suitable for generating full-precision f64 randoms." This is acceptable for bandwidth benchmarking where exact values don't matter, only memory access patterns.

12. **Failure limit in verification**: The verification step allows up to 100 floating-point mismatches before bailing. This accounts for potential floating-point non-associativity between the CPU model and GPU kernel, though for a simple sum stencil, results should be identical.

13. **No MPI**: This is a single-GPU benchmark — the entire 128M-element array fits on modern GPUs (especially with float data).
