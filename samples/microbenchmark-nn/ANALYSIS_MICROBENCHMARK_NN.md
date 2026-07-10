# ⚠️ WARNING ⚠️
# **This is a LLM based analysis.**
# **THE BRANCH SHOULD NEVER BE MERGED TO master OR develop BRANCHES!**

# LLM Information
- **Model:** Qwen3.6-35B-A3B

# Overview
The `microbenchmark-nn` sample is a GPU microbenchmark suite that measures the effective memory bandwidth of **NVIDIA cuDNN** and **AMD MIOpen** convolution backends for 1D convolutions (equivalent to a stencil sum). It maps a 1D stencil `y[i] = sum(x[i+r])` onto the convolution API using a kernel of all ones, enabling direct comparison between hand-written CUDA kernels (in the `microbenchmark` sample) and vendor-optimized deep learning library implementations. Supports both NVIDIA (cuDNN) and AMD (MIOpen) backends via conditional compilation. Produces CSV output with per-iteration bandwidth measurements.

# Directory Structure & File Descriptions

| File | Description |
| :--- | :--- |
| `CMakeLists.txt` | Build configuration; conditionally compiles against either `cudnn` (NVIDIA) or `MIOpen` + `roctracer` (AMD/HIP). Includes `acc-runtime/api` for device attribute queries. HIP mode sets all `.cu` files to HIP language and links `roctracer64` + `MIOpen`. |
| `microbenchmark-nn.cu` | Main benchmark driver (305 lines): device info, CPU model verification, backend-based benchmarking with POSIX timer and CSV output. |
| `array.h` | Array type declaration: `{ length, bytes, data, on_device }`. Declares `arrayCreate`, `arrayDestroy`, `randd`, `arrayRandomize`. |
| `array.cu` | Array implementation: device/host allocation, randomization. Same interface as `microbenchmark/array.h` but as separate compilation unit with `bytes` field. |
| `backend.h` | Abstract backend interface: `backendInit`, `backendQuit`, `backendConvolutionFwd`, `backendGetInputTensor`, `backendGetOutputTensor`. |
| `backend-cudnn.cu` | NVIDIA cuDNN 7.x implementation: sets up NCHW 4D tensors, uses `cudnnFindConvolutionForwardAlgorithm` for auto-tuning, cross-correlation mode. |
| `backend-miopen.cu` | AMD MIOpen implementation: equivalent setup with `miopen` API calls, prints selected algorithm (GEMM/Direct/FFT/Winograd). |

# Data Structures

| Structure | Description |
| :--- | :--- |
| `Array` | Tensor memory wrapper: `{ length, bytes, data, on_device }`. `bytes` is precomputed as `length * sizeof(real)`. `data` is `real*` (float or double, but double precision is explicitly rejected by both backends). |

# Backend Interface

| Function | Description |
| :--- | :--- |
| `backendInit(domain_length, radius, stride)` | Initialize backend: create cuDNN/MIOpen handle, descriptors, allocate input/filter/output/workspace tensors, select convolution algorithm. |
| `backendQuit()` | Destroy descriptors, free tensors, destroy handle. |
| `backendConvolutionFwd()` | Execute one forward convolution pass: `output = alpha * (input ⊛ filter) + beta * output`, with `alpha=1, beta=0`. |
| `backendGetInputTensor()` | Returns the input `Array` for host data copying. |
| `backendGetOutputTensor()` | Returns the output `Array` for host data copying. |

# Backend Implementations

## cuDNN (`backend-cudnn.cu`)

| Aspect | Value |
| :--- | :--- |
| **Library** | NVIDIA cuDNN 7.x (`<cudnn.h>`) |
| **Data type** | `CUDNN_DATA_FLOAT` (double precision explicitly rejected: `ERRCHK_ALWAYS(sizeof(real) == sizeof(float))`) |
| **Format** | `CUDNN_TENSOR_NCHW` |
| **Convolution type** | `CUDNN_CROSS_CORRELATION` |
| **Algorithm selection** | `cudnnFindConvolutionForwardAlgorithm` (one algorithm) |
| **Workspace** | Auto-computed via `cudnnGetConvolutionForwardWorkspaceSize` |

Tensor descriptors:
| Descriptor | N | C | H | W | Size |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Input | 1 | 1 | 1 | `domain_length` | `domain_length` |
| Filter | 1 | 1 | 1 | `2 * radius + 1` | `2 * radius + 1` |
| Output | `fn_out` | `fc_out` | `fh_out` | `fw_out` | Same |

Convolution parameters:
| Parameter | Value | Description |
| :--- | :--- | :--- |
| `pad_h` | `(gh - 1) / 2` | 0 (since gh=1) |
| `pad_w` | `(gw - 1) / 2` | `radius` (half-width padding for "same" output) |
| `stride_h` | 1 | No stride |
| `stride_w` | 1 | No stride |
| `dilation_h` | 1 | No dilation |
| `dilation_w` | 1 | No dilation |

Filter is initialized to all ones — this makes the cuDNN convolution equivalent to the stencil sum operation.

## MIOpen (`backend-miopen.cu`)

| Aspect | Value |
| :--- | :--- |
| **Library** | AMD MIOpen (`<miopen/miopen.h>`) |
| **Data type** | `miopenFloat` (double precision explicitly rejected) |
| **Convolution type** | `miopenConvolution` |
| **Algorithm selection** | `miopenFindConvolutionForwardAlgorithm` with workspace, brute-force mode enabled |
| **Workspace** | Pre-computed via `miopenConvolutionForwardGetWorkSpaceSize` |

Algorithm selection prints the chosen algorithm:
| Algorithm ID | Name | Description |
| :--- | :--- | :--- |
| 0 | `miOpenConvolutionAlgoGEMM` | Im2col + GEMM |
| 1 | `miopenConvolutionAlgoDirect` | Direct convolution |
| 2 | `miopenConvolutionAlgoFFT` | FFT-based convolution |
| 3 | `miopenConvolutionAlgoWinograd` | Winograd minimal filtering |
| 5 | `miopenConvolutionAlgoImplicitGEMM` | Implicit GEMM |

Note: Algorithm ID 4 is skipped (reserved).

## Tensor Shape Mapping (Stencil → Convolution)

The 1D stencil is mapped to a 1×1×W convolution:

```
Stencil:  y[i] = Σ x[i + r] for r in [-radius, radius]

Convolution:
  Input tensor:  N=1, C=1, H=1, W=domain_length
  Filter tensor: N=1, C=1, H=1, W=2*radius+1  (all ones)
  Padding: W=(gw-1)/2 = radius   → "same" convolution → output width = domain_length
```

The 4D tensor layout `NCHW` is used even for 1D data — H=1 (height dimension is trivial).

# Benchmark Pipeline

1. **Device Info Print**: Calls `printDeviceInfo(0)` showing GPU properties (same format as `microbenchmark` sample).

2. **Verification** (`verify()`):
   - CPU model: `model_kernel` performs `y[i] = Σ x[i+r]` with boundary check `idx >= 0 && idx < domain_length` (implicit zero padding).
   - GPU candidate: Host input → `cudaMemcpy` → `backendConvolutionFwd()` → `cudaMemcpy` → compare element-by-element (100 failure limit).

3. **Benchmarking** (`benchmark()`):
   - 10-dryrun warmup iterations before profiling.
   - `num_samples` timed iterations with POSIX high-resolution timer.
   - Profiling via `cudaProfilerStart()`/`Stop()`.
   - CSV output with per-iteration: milliseconds, effective bandwidth.

# Benchmark Parameters

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `domain_length` | 128K | Computational domain size in elements (128 * 1024 / sizeof(real)) — note: smaller than `microbenchmark` (128M) |
| `radius` | 1 | Stencil half-width / filter half-width |
| `stride` | 1 | Stencil/convolution stride (enforced to 1) |
| `jobid` | 0 | Job identifier for output CSV filename |
| `num_samples` | 100 | Number of benchmark iterations |
| `salt` | 42 | Random seed modifier |

## Derived Parameters

| Parameter | Formula |
| :--- | :--- |
| `seed` | `12345 + salt + (1 + domain_length + radius + stride + jobid + num_samples) * time(NULL)` |

## Compile-Time Options

| Option | Default | Description |
| :--- | :--- | :--- |
| `USE_HIP` | 0 | Compile for AMD HIP/MIOpen if 1, NVIDIA/cuDNN if 0 |
| `AC_DOUBLE_PRECISION` | 0 | Single precision (float) if 0, double precision (double) if 1 (but both backends reject double) |

# Bandwidth Calculation

Effective bandwidth formula (same as `microbenchmark`):

```
bytes = sizeof(real) * (2 * domain_length + 2 * radius / stride)
seconds = milliseconds / 1000
bandwidth = bytes / seconds
```

- `2 * domain_length`: one read of input tensor + one write of output tensor (each of length `domain_length`)
- `2 * radius / stride`: extra reads for the filter stencil (the filter of size `2*radius+1` is read `domain_length` times, but this is already partially counted in the input read — the formula appears to over-count by including filter reads)

**Note**: The bandwidth formula here does not account for the filter tensor reads. The actual memory traffic is:
- Input: `domain_length * sizeof(real)` bytes read
- Filter: `domain_length * (2*radius+1) * sizeof(real)` bytes read (filter is reused across all spatial positions)
- Output: `domain_length * sizeof(real)` bytes written

Total actual: `(2 * domain_length + domain_length * (2*radius+1)) * sizeof(real)`

The benchmark formula `2 * domain_length + 2 * radius` is likely a simplification that doesn't account for the filter memory traffic, which becomes significant at large radii. This makes direct comparison with the `microbenchmark` sample's bandwidth misleading at larger radii.

# Profiling

### CUDA (NVIDIA)

```bash
# cuDNN backend (default)
./microbenchmark-nn 33554432 0
```

### HIP/ROCm (AMD)

```bash
cmake -DUSE_HIP=ON .. && make -j
rocprof --trace-start off -i rocprof-input-metrics.txt ./microbenchmark-nn
```

Same `rocprof-input-metrics.txt` format as `microbenchmark` sample (Wavefronts, VALUInsts, TCC_HIT/MISS, L2CacheHit, etc.).

# CSV Output Format

Output file: `microbenchmark-nn-{jobid}-{seed}.csv`

| Column | Description |
| :--- | :--- |
| `implementation` | `"cudnn"` or `"miopen"` |
| `maxthreadsperblock` | -1 (no threadblock config; backend-internal) |
| `domainlength` | Domain length in elements |
| `radius` | Stencil/filter radius |
| `stride` | Convolution stride |
| `milliseconds` | Iteration time in ms |
| `effectivebandwidth` | Bandwidth in bytes/second |
| `tpb` | -1 (no threadblock config) |
| `jobid` | Job identifier |
| `seed` | Random seed |
| `iteration` | Iteration index |
| `double_precision` | 0 or 1 |

# Notable Observations

1. **No autotuning**: Unlike `microbenchmark`, this sample skips the threadblock-size autotuning phase. cuDNN/MIOpen internally select the optimal algorithm and launch configuration, so threadblock tuning is not exposed.

2. **Smaller default problem size**: Default `domain_length` is `128K` (128 * 1024) vs `128M` in `microbenchmark`. This is likely because cuDNN's algorithm search may be slower for very large inputs, or the convolution overhead is more pronounced at small sizes.

3. **Double precision rejected**: Both `backend-cudnn.cu` and `backend-miopen.cu` have `ERRCHK_ALWAYS(sizeof(real) == sizeof(float))` — double precision convolution algorithms are not supported or tested. The `AC_DOUBLE_PRECISION` compile option is effectively ignored.

4. **Filter initialized to ones**: The convolution kernel is a 1×1×(2r+1) tensor of all ones. This makes the cuDNN/MIOpen convolution mathematically equivalent to the stencil sum, enabling direct comparison. The filter is allocated on-device and initialized from a host buffer.

5. **"Same" convolution padding**: The padding `(gw-1)/2 = radius` ensures the output has the same width as the input (`domain_length`). This matches the stencil behavior of the `microbenchmark` sample (which uses explicit padding via `pad`).

6. **Boundary handling difference**: The CPU model uses **implicit zero padding** (`if idx >= 0 && idx < domain_length`), while the convolution uses **explicit symmetric padding**. For a ones filter with zero-padded input edges, these are equivalent — the edges of the output will be lower values since they sum fewer input elements due to the zero-padding.

7. **cuDNN algorithm auto-tuning**: `cudnnFindConvolutionForwardAlgorithm` runs all supported algorithms on the input data and selects the fastest. This search can be expensive (several seconds) but only runs once during `backendInit`.

8. **MIOpen algorithm with workspace**: `miopenFindConvolutionForwardAlgorithm` takes a workspace buffer and brute-force flag (`true`), meaning it tests all supported algorithms including potentially slow ones. The selected algorithm is printed for debugging.

9. **cuDNN uses workspace buffer**: The convolution forward call passes `workspace.data` and `workspace.bytes` — the workspace size is computed at init time and may be substantial for large inputs.

10. **10-dryrun warmup**: The benchmark runs 10 warmup iterations before starting timed measurements. This likely serves to trigger cuDNN's internal caching and ensure all algorithm code is compiled (cuDNN may JIT-compile kernels on first use).

11. **Profiled during benchmarking**: Like `microbenchmark`, the CUDA profiler is started before benchmark iterations and stopped after. Profiling overhead will affect measured times.

12. **Backend abstraction is minimal**: `backend.h` provides a thin interface — the entire cuDNN/MIOpen complexity is hidden behind `backendInit`, `backendConvolutionFwd`, and `backendQuit`. This makes the main benchmark driver backend-agnostic.

13. **No shared memory configuration**: Unlike `microbenchmark` which has `USE_SMEM` and `MAX_THREADS_PER_BLOCK` options, the backend-based approach delegates all memory hierarchy optimization to the vendor library.

14. **Bandwidth measurement methodology**: The bandwidth formula `2 * domain_length + 2 * radius` does not account for filter reads (`domain_length * (2*radius+1)`), which are a significant part of the actual memory traffic for large radii. This makes the reported bandwidth likely inflated compared to the true memory bandwidth utilization.

15. **NCHW format for 1D data**: Using 4D tensor format `N=1, C=1, H=1, W=domain_length` for a 1D problem is a common pattern in deep learning frameworks, but adds a layer of abstraction overhead compared to direct 1D array operations.

16. **cuDNN error handling**: Uses a custom `CUDNN_ERRCHK` macro that prints file/line and a timestamp, with `abort=true` (always prints, doesn't exit).

17. **MIOpen does not use tensor format**: `miopenSet4dTensorDescriptor` doesn't take a format parameter (unlike cuDNN's `cudnnSetTensor4dDescriptor` which takes `CUDNN_TENSOR_NCHW`). MIOpen uses a different descriptor API.

18. **No MPI**: Single-GPU benchmark, no domain decomposition or multi-device communication.
