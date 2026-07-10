# ⚠️ WARNING ⚠️
# **This is a LLM based analysis.**
# **THE BRANCH SHOULD NEVER BE MERGED TO master OR develop BRANCHES!**

# LLM Information
- **Model:** Qwen3.6-35B-A3B

# Overview
The `benchmark-device` sample is a per-kernel-level GPU benchmarking tool that directly benchmarks Astaroth device (GPU) kernels, as opposed to the higher-level `samples/benchmark/` which benchmarks full integration steps via task graphs. It measures individual kernel launch performance (kernel runtime in milliseconds) and reports optimal thread block dimensions (TPB). It also includes an optional CPU-vs-GPU verification path and a separate Python-based nonlinear MHD benchmarking script for comparison against PyTorch, TensorFlow, and JAX.

# Directory Structure & File Descriptions

| File | Description |
| :--- | :--- |
| `CMakeLists.txt` | Build configuration; compiles `main.cc` into the `benchmark-device` executable with position-independent code, linked against `astaroth_core` and `astaroth_utils`. |
| `main.cc` | Device-level kernel benchmark: direct kernel launch via `acDeviceLaunchKernel`, timing via high-resolution timer, optional CPU reference verification, and per-iteration CSV output. |
| `mhd.py` | Standalone Python script that benchmarks a nonlinear MHD forward model across three ML frameworks (PyTorch, TensorFlow, JAX) and verifies correctness against a NumPy/SciPy reference implementation. |

# Precondition Check

The executable requires the integration DSL to be enabled:

| Condition | Macro | Error Message |
| :--- | :--- | :--- |
| Integration DSL | `AC_INTEGRATION_ENABLED` | "AC_INTEGRATION was not enabled, cannot run benchmark-device" |

If the macro is not defined, `main()` returns `EXIT_FAILURE`.

# Command-Line Interface

The executable takes 7 positional arguments:

| Argument | Default | Description |
| :--- | :--- | :--- |
| `<nx> <ny> <nz>` | `256, 256, 256` | Grid dimensions. |
| `<jobid>` | `0` | Job identifier, used in the output filename. |
| `<num_samples>` | `100` | Number of benchmark iterations. |
| `<verify>` | `0` | Set to `1` to enable CPU-vs-GPU verification (load/store, boundconds, integration). |
| `<salt>` | `42` | Salt for random seed generation. |

Usage: `./benchmark-device <nx> <ny> <nz> <jobid> <num_samples> <verify> <salt>`

# Compile-Time Constants

| Macro | Description |
| :--- | :--- |
| `IMPLEMENTATION` | Build implementation identifier (written to CSV). |
| `MAX_THREADS_PER_BLOCK` | Maximum threads per block on the target device. |
| `AC_DOUBLE_PRECISION` | Double precision flag (`1` or `0`). |

# Program Flow

## 1. Initialization
1. `acProfilerStop()` — Disable the profiler.
2. Parse command-line arguments and compute a random seed: `seed = 12345 + salt + (1 + nx + ny + nz + jobid + num_samples + verify) * time(NULL)`.
3. Print configuration constants: `IMPLEMENTATION`, `MAX_THREADS_PER_BLOCK`, `DOUBLE_PRECISION`.

## 2. Mesh & Device Setup
1. `acInitInfo()` / `acLoadConfig(AC_DEFAULT_CONFIG, &info)` / `acHostUpdateParams(&info)` — Load mesh configuration.
2. `acSetLocalMeshDims(nx, ny, nz, &info)` — Set local (per-process) mesh dimensions.
3. Configure decomposition:
   - `AC_PROC_MAPPING_STRATEGY_LINEAR` — Linear process mapping.
   - `AC_DECOMPOSE_STRATEGY_MORTON` — Morton curve decomposition.
   - `AC_MPI_COMM_STRATEGY_DUP_WORLD` — Duplicate MPI_COMM_WORLD (if `AC_MPI_ENABLED`).
4. (Optional) Runtime compilation and dynamic library loading if `AC_RUNTIME_COMPILATION` is defined.
5. `acPrintMeshInfo(info)` — Print mesh information to stderr.
6. `acDeviceCreate(0, info, &device)` — Create the device (GPU) object; device ID 0 is hardcoded.
7. `acDevicePrintInfo(device)` — Print device information.

## 3. Random Number Generation
- `acRandInitAlt(seed, count, pid)` — Initialize Astaroth's device random number generator.
- `srand(seed)` — Initialize host-side `rand()`.

## 4. Optional Verification (`verify == 1`)

Three independent verification stages run sequentially, each comparing GPU results against a CPU reference:

### 4a. Load/Store Verification
1. Randomize host meshes `model` and `candidate`.
2. `acDeviceLoadMesh(device, STREAM_DEFAULT, model)` — Upload `model` to the GPU.
3. `acDeviceStoreMesh(device, STREAM_DEFAULT, &candidate)` — Download to `candidate`.
4. `acDeviceSynchronizeStream(device, STREAM_DEFAULT)` — Wait for completion.
5. `acVerifyMesh("Load/Store", model, candidate)` — Compare.

### 4b. Boundary Condition Verification
1. Randomize `model` and `candidate`, load to device.
2. `acDevicePeriodicBoundconds(device, STREAM_DEFAULT, dims.m0, dims.m1)` — Apply GPU periodic boundaries.
3. Store to `candidate`, synchronize, apply host periodic bounds, compare via `acVerifyMesh("Boundconds", ...)`.

### 4c. Integration Verification
1. Randomize `model` and `candidate`, load to device.
2. Apply periodic boundary conditions.
3. 5 verification steps, each performing:
   - 3 integration substeps via `acDeviceIntegrateSubstep(device, STREAM_DEFAULT, i, dims.n0, dims.n1, dt)`.
   - `acDeviceSwapBuffers(device)` and `acDevicePeriodicBoundconds(device, ...)` after each substep.
   - `acHostIntegrateStep(model, dt)` — CPU reference step.
   - `acHostMeshApplyPeriodicBounds(&model)` — CPU boundary application.
4. Store final device mesh to `candidate`, synchronize, compare via `acVerifyMesh("Integration", ...)`.

## 5. Benchmark Loop

The benchmark focuses on two kernels: `singlepass_solve` (the main compute kernel) and `randomize`.

Each iteration performs:

### Warmup / Reset
1. `acDeviceGetKernelInputParamsObject(device)` — Obtain the kernel input parameters object.
2. Set parameters: `step_num = AC_SUBSTEP_NUMBER(2)`, `dt`, `current_time`.
3. `acDeviceLaunchKernel(device, STREAM_DEFAULT, singlepass_solve, dims.n0, dims.n1)` — Launch the target kernel.
4. `acDeviceResetMesh(device, STREAM_DEFAULT)` — Reset the mesh buffers.
5. `acDeviceLaunchKernel(device, STREAM_DEFAULT, randomize, dims.n0, dims.n1)` — Randomize the mesh.
6. `acDeviceSwapBuffers(device)` — Swap in/out buffers.
7. `acDeviceSynchronizeStream(device, STREAM_ALL)` — Wait for completion.

### Timing
1. `timer_reset(&t)` — Start timer.
2. `acDeviceLaunchKernel(device, STREAM_DEFAULT, singlepass_solve, dims.n0, dims.n1)` — Launch and time the kernel.
3. `acDeviceSynchronizeStream(device, STREAM_ALL)` — Wait for completion.
4. `timer_diff_nsec(t) / 1e6` — Convert elapsed nanoseconds to milliseconds.
5. `acKernelLaunchGetLastTPB()` — Retrieve the actual thread block dimensions used for the last launch.

### CSV Output
Each iteration writes a row to `benchmark-device-{jobid}-{seed}.csv`:
```
implementation, maxthreadsperblock, nx, ny, nz, milliseconds, tpbx, tpy, tpbz, jobid, seed, iteration, double_precision
```

After the final iteration, rank 0 also prints the last iteration's timing and optimal TPB to stdout.

## 6. Profiling
If requested, a single kernel launch is profiled with `acProfilerStart()` / `acProfilerStop()`.

## 7. Cleanup
1. Close the CSV file.
2. `acDeviceDestroy(&device)` — Destroy the device.
3. `acHostMeshDestroy(&model)` / `acHostMeshDestroy(&candidate)` — Destroy host meshes.

# The Target Kernel: `singlepass_solve`

The benchmark directly launches the `singlepass_solve` kernel (a kernel symbol expected to be defined by the compiled DSL library). This kernel represents a single-pass RHS evaluation with substep parameterization. The kernel input parameters object (`AcKernelInputParams`) is configured with:
- `singlepass_solve.step_num` — The substep number (0-2).
- `singlepass_solve.time_params.dt` — The timestep value.
- `singlepass_solve.time_params.current_time` — The simulation time.

The kernel launch dimensions are `(dims.n0, dims.n1)` — the local grid extent.

# CSV Output File

The output file is named `benchmark-device-{jobid}-{seed}.csv` and contains one row per iteration with columns:

| Column | Description |
| :--- | :--- |
| `implementation` | Compile-time `IMPLEMENTATION` constant. |
| `maxthreadsperblock` | Compile-time `MAX_THREADS_PER_BLOCK` constant. |
| `nx, ny, nz` | Input grid dimensions. |
| `milliseconds` | Kernel launch duration for `singlepass_solve` only. |
| `tpbx, tpy, tpbz` | Actual thread block dimensions used (from `acKernelLaunchGetLastLastTPB()`). |
| `jobid` | Job identifier argument. |
| `seed` | Computed random seed. |
| `iteration` | Iteration index (0-based). |
| `double_precision` | 1 if `AC_DOUBLE_PRECISION`, 0 otherwise. |

# Python MHD Benchmark (`mhd.py`)

This standalone Python script benchmarks a nonlinear MHD forward model across three ML frameworks. It is **not** linked to the Astaroth C library — it provides an independent reference for comparing convolution-based stencil computations on GPU against GPU-accelerated ML frameworks.

## Dependencies
- `numpy`, `scipy`, `pandas`, `findiff`
- One of: `pytorch`, `tensorflow`, `jax` (selected via `--library`)

## CLI Arguments

| Argument | Default | Description |
| :--- | :--- | :--- |
| `--dims` | `(128, 128, 128)` | Computational domain dimensions. |
| `--device` | `gpu` | Device: `cpu` or `gpu`. |
| `--radius` | `1` | Stencil radius (determines finite-difference order: `2*radius`). |
| `--dtype` | `fp32` | Data precision: `fp32` or `fp64`. |
| `--library` | required | ML framework: `pytorch`, `tensorflow`, or `jax`. |
| `--verify` | `1` | Verify results against NumPy reference. |
| `--jobid` | `0` | Job identifier. |
| `--salt` | `12345` | Random salt. |
| `--nsamples` | `100` | Number of benchmark samples. |
| `--visualize` | — | Generate MHD evolution animation. |

## MHD Model

The forward model computes the time derivatives of 8 MHD fields:

| Field | Index | Description |
| :--- | :--- | :--- |
| `lnrho` | 0 | Log density |
| `ux` | 1 | x-velocity |
| `uy` | 2 | y-velocity |
| `uz` | 3 | z-velocity |
| `ax` | 4 | x-vector potential |
| `ay` | 5 | y-vector potential |
| `az` | 6 | z-vector potential |
| `ss` | 7 | Entropy variable |

## Stencil Operations

| Stencil | Description |
| :--- | :--- |
| `value` | Kronecker delta (identity). |
| `ddx, ddy, ddz` | First-order derivatives. |
| `d2dx2, d2dy2, d2dz2` | Second-order derivatives. |
| `d2dxdy, d2dxdz, d2dydz` | Mixed second-order derivatives. |

## Physical Constants

| Constant | Value | Description |
| :--- | :--- | :--- |
| `cs2_sound` | `1.0` | Sound speed squared coefficient. |
| `nu_visc` | `1e-3` | Dynamic viscosity. |
| `zeta` | `0.01` | Bulk viscosity. |
| `gamma` | `0.5` | adiabatic index. |
| `lnrho0` | `1.3` | Reference log density. |
| `cp_sound` | `1` | Specific heat at constant pressure. |
| `mu0` | `1.4` | Magnetic permeability. |
| `eta` | `1e-2` | Resistivity. |
| `lnT0` | `1.2` | Reference log temperature. |
| `K_heatcond` | `1e-3` | Thermal conductivity coefficient. |

## Framework-Specific Implementations

### PyTorch
- Uses `torch.nn.functional.conv3d` for stencil convolutions.
- Enables cuDNN autotuning (`torch.backends.cudnn.benchmark = True`).
- Disables tensor cores to ensure precision consistency.
- Benchmarks via `torch.jit.trace` and `torch.utils.benchmark`.
- CUDA events record precise GPU-side timing.

### TensorFlow
- Uses `tf.nn.convolution` for stencil convolutions.
- JIT-compiled via `@tf.function(jit_compile=True)`.
- Benchmarks via `tf.test.Benchmark.run_op_benchmark`.

## Verification

When `--verify` is set, the script:
1. Computes the forward model result with the NumPy reference implementation.
2. Runs the framework-specific forward model on the same random input.
3. Compares results with relative/absolute tolerance based on dtype (fp32 or fp64 epsilon * 100).
4. If verification fails, prints largest absolute error and failure indices.
5. Also verifies each individual convolution kernel output against the NumPy reference.

## Visualization

When `--visualize` is set, the script generates a 2-panel animation (100 frames) of all 8 MHD field slices, saved as `nonlinear-mhd-{library}.mp4`. The time integration uses a 3-stage substep scheme with alpha/beta coefficients.

# Key Dependencies

## C++ (`main.cc`)
- `astaroth.h` — Core device API (`acDeviceCreate`, `acDeviceLaunchKernel`, `acDeviceIntegrateSubstep`, `acDeviceSwapBuffers`, etc.).
- `astaroth_utils.h` — Utility functions.
- `errchk.h` — Error-checking macros.
- `timer_hires.h` — High-resolution timer.
- `MPI` — Optional, if `AC_MPI_ENABLED`.

## Python (`mhd.py`)
- `numpy` — Base array operations and random number generation.
- `scipy.ndimage` — Reference convolution via `scipy.ndimage.correlate`.
- `findiff` — Finite-difference coefficient computation.
- `torch` / `tensorflow` / `jax` — ML framework backends.
- `pandas` — CSV output.
