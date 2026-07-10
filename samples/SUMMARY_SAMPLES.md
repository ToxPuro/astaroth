# Summary of Sample Directories — Simulation Applications

This document summarizes all `ANALYSIS_*.md` files for sample directories in the **Simulation Applications** category. Each entry is 2–4 sentences.

## les

The `les` (Large Eddy Simulation) sample implements a compressible Navier-Stokes solver with Smagorinsky subgrid-scale turbulence model, using 6th-order compact finite-difference stencils and a 3-substep strong-stability-preserving RK3 time integrator. It runs 2499 time steps on a 32³ periodic grid, computing stress tensors, momentum advection, viscous diffusion, and continuity evolution across two physics variants (`main.ac` for incompressible-style and `main.ac.compressible` for fully compressible flow). The C driver performs verification tests (load/store, read/write, boundary conditions), Nsight Compute profiling integration, and saves field slices every 25 steps for Python-based matplotlib visualization. Notable features include an AMD performance workaround for RK3 conditionals and a dual-kernel-per-substep approach separating stress tensor computation from the time-stepping solve.

## standalone

> **⚠️ DEPRECATED:** This directory is superseded by `standalone_mpi`.

The `standalone` directory provides a CPU-based reference implementation and multi-tool executable (`ac_run`) for Astaroth, consisting of a comprehensive `long double` CPU solver that re-implements the full hydrodynamic/MHD pipeline for GPU-CPU numerical verification, plus four operational modes: autotest (GPU vs CPU comparison), benchmark (CFL-limited performance profiling), simulation (production time-advance with I/O and diagnostics), and real-time rendering (SDL2-based X-Y slice visualization). The CPU model in the `model/` subdirectory implements RK3 time integration, boundary conditions, diffusion stencils, helical forcing, and sink particle physics, all serving as a high-precision reference against which GPU results are compared using a tight 30 ULP tolerance.

## standalone_mpi

The `standalone_mpi` sample is Astaroth's full production-grade simulation driver (`ac_run_mpi`), an MPI-enabled HPC application that runs magnetohydrodynamic or variant simulations with adaptive timestepping, periodic boundary conditions, DSL task graph execution, snapshot/slice I/O, runtime config reloading, helical forcing, sink physics, and shock viscosity. It uses the modern `acGrid*` API with Morton-order domain decomposition and supports multiple physics configurations (MHD, shock single-pass, hydro/heat-duct two-pass, boundary condition tests) via custom task graph builders. The program features file-based control signals (STOP/RELOAD), non-blocking async I/O, and per-step diagnostics output to `timeseries.ts`, making it the primary entry point for running Astaroth on GPU clusters.

## stress

The `stress` directory provides an MPI-based 2D elastic wave stress simulation using the Astaroth DSL task graph system, modeling dynamic stress propagation in a steel-like material (E = 200 GPa, Poisson's ratio 1/3) on a 168×168 grid with a 5-kernel pipeline: initial condition, displacement calculation, Hooke's law stress computation, momentum balance with damping, and velocity smoothing with coordinate update. The simulation runs 1000 timesteps with an extremely small effective timestep (`eldt = 2×10⁻¹²`) and strong damping (`10¹⁵`), writing the final `STRESS11` component values to a single output file. Notable quirks include an inconsistent Verlet-like position update paired with Euler velocity integration, a duplicate X-coordinate column in the output, and the absence of a CPU reference for DSL kernel verification.

## tfm

The `tfm` directory is the most physics-rich sample in the suite, implementing a comprehensive Test Field Method (TFM) MHD simulator for studying small-scale dynamo and mean-field astrophysics. It provides four executables (`tfm`, `tfm_mpi`, `tfm_pipeline`, `tfm_standalone`), an 1116-line DSL physics file, INI configuration files, and Python visualization scripts. The TFM methodology solves for four pairs of magnetic vector potential test fields advected by a hydrodynamic velocity field, computing mean-field coefficients (alpha, beta/turbulent diffusivity) from correlations between fluctuating fields and mean profiles. Key features include configurable SOCA and magnetic Laplace diffusion variants, helical forcing based on Pencil Code, and a 3-substep RK3 integrator with 28 boundary condition calls per substep.

## tfm-mpi

The `tfm-mpi` sample is the MPI-accelerated Test Field Method simulator with a hand-coded C++ pipeline (1920 lines in `tfm.cc`) instead of the DSL task graph builder. It implements the full simulation loop with segmented computation (halo vs inner domain), two independent halo exchange batches for hydro and TFM fields, LUMI topology-aware GPU mapping, and periodic test field resets. The sample includes extensive Python analysis scripts for computing alpha and eta tensors from EMF profiles, timeseries plotting, slice animation, and production run result archives. Notable implementation details include a dry-run initialization in the `rev::Grid` constructor, async profile writes, dual-buffer snapshot rotation for restart support, and Mahti compatibility workarounds for CUDA-aware MPI IO.

# Summary of Sample Directories — Learning & Onboarding

This document summarizes all `ANALYSIS_*.md` files for sample directories in the **Learning & Onboarding** category. Each entry is 2–4 sentences.

## ac-interpreter

The `ac-interpreter` is a command-line REPL utility for interactive testing and debugging of the Astaroth library, providing manual control over core library functions including mesh initialization, kernel execution, data I/O, and field reductions. It processes a set of interactive commands (`init`, `load_random`, `read`, `write`, `launch`, `reduce`, `reduce_all`, `exit`) via standard input, using `acDevice*` APIs for device management and `acHostMesh*` APIs for host-side mesh operations. The tool is designed for exploratory debugging and rapid prototyping, allowing developers to launch individual kernels and inspect field reductions without writing full simulation drivers.

## advection-example

The `advection-example` is Astaroth's "hello world" — the simplest end-to-end simulation that advects a scalar concentration field by a constant velocity vector using the Astaroth DSL and task graph system. It demonstrates the complete workflow: DSL kernel definition (Euler update and initial condition kernels with periodic boundaries), configuration-driven setup via `advec.conf`, mesh initialization, task graph optimization and execution in a time-stepping loop, and slice-based output rendering. With a minimal 256×8×8 grid and a CFL number of 0.05, it serves as the recommended starting point in Astaroth's own learning path, preceding more complex samples like `standalone_mpi` and the full MHD DSL.

## blur

The `blur` sample is Astaroth's minimal stencil-based image blurring example, demonstrating the DSL's `Stencil` construct — a named weighted convolution pattern applicable to any field. It includes three DSL implementations: a reference 7-point 3D cross stencil (`1/7` weight on center + 6 face neighbors), a completed 9-point 2D box blur (`1/9` on 3×3 x-y cross), and an incomplete student exercise with only 3 of 9 coefficients filled. The C host program creates a uniform mesh (all cells set to `1.0`), loads it to GPU, launches the blur kernel, and prints the resulting grid to verify that constant fields are preserved (stencil weights sum to `1.0`).

## plasma-meets-ai-workshop

The `plasma-meets-ai-workshop` is an educational workshop package containing a progressive series of exercises from a simple 2D blur filter to a full 3D magnetohydrodynamics solver with Smagorinsky subgrid-scale turbulence modeling. Each exercise builds on the previous one, using the standalone `acDevice*` API with a 256×256×1 thin grid to keep memory usage low for workshop participants. The package includes incomplete student exercises, completed reference solutions in `blur-demo/`, and pre-completed model examples in `model-examples/`. Key physics features include 7th-order compact finite differences, a 3-substep RK3 integrator, annular ring vortex forcing, and a 7×7×7 uniform box smoothing filter, while the MHD exercise is intentionally left incomplete as a final challenge.

# Summary of Sample Directories — Correctness & Verification Tests

This document summarizes all `ANALYSIS_*.md` files for sample directories in the **Correctness & Verification Tests** category. Each entry is 2–4 sentences.

## boundcond_test

The `boundcond_test` is Astaroth's comprehensive boundary condition correctness validation suite, testing all ghost-zone BC implementations (periodic, symmetric, antisymmetric, relative antisymmetry, prescribed derivative) by decomposing the 3D mesh boundary into 26 regions (6 faces, 12 edges, 8 corners). For each region and every vertex buffer field, it executes a task graph (halo exchange followed by directional BC applications) and compares device-computed ghost values against analytical kernel expectations with double-precision tolerance (`epsilon = 1e-14`). The test also includes a periodic boundary query test that validates `acGridTaskGraphHasPeriodicBoundcondsX/Y/Z()` across all 8 periodicity combinations, and provides detailed per-field error reporting for failed regions.

## convection_kramers

The `convection_kramers` sample is a boundary condition verification test structurally nearly identical to `boundcond_test` but with two distinguishing additions: a simplified kernel function signature (drops the `int3 normal` parameter) and a pilot MHD BCs test that exercises `acSpecialMHDBoundaryCondition` for entropy blackbody radiation and entropy prescribed heat flux. It tests the same four standard BC types (symmetric, antisymmetric, prescribed derivative with value 6.0, relative antisymmetry) across all 26 ghost zone regions, using only `AC_dsx` (isotropic grid assumption) instead of direction-specific grid spacings. The pilot MHD task graph is constructed but not executed, serving as a development draft for special MHD boundary condition testing.

## cpptest

The `cpptest` is Astaroth's simplest integration test — a single-functionality C++ executable that validates the `AC_SINGLEPASS_INTEGRATION` mode by creating a host mesh, randomizing it, applying periodic boundaries, loading to the device, performing one integration step with `dt = FLT_EPSILON`, and printing min/max ranges for all vertex buffers. It uses the legacy `acInit`/`acQuit` API (not the modern `acGrid*` task-graph API) and requires no MPI, making it a single-processor smoke test for the single-pass integration infrastructure. The program includes a load/store round-trip verification and prints "cpptest complete." on success.

## ctest

The `ctest` is the C-language counterpart to the C++ `cpptest`, functionally identical but with an added startup GPU device availability check via `acCheckDeviceAvailability()` and a hardcoded error message bug (still references "cpptest" instead of "ctest"). It compiles `main.c` as pure C with `errchk.h` included, uses `POSITION_INDEPENDENT_CODE`, and follows the same single-pass integration validation flow: device check → config → mesh creation → randomize → load → store → integrate → print vertex buffer ranges → cleanup. Like `cpptest`, it requires no MPI and is a single-processor test.

## devicetest

The `devicetest` is Astaroth's comprehensive GPU device correctness validator, comparing device-side integration results against a host-side reference implementation using ULP (units in the last place) error tolerance. It runs three validation phases: boundary condition verification, full integration comparison (default 100 steps, configurable grid), and all reduction type testing (scalar, vector, and Alfvén reductions with 5 reduction types each). Written in C++ using the legacy `acDevice*` API, it runs via `mpirun` but assumes rank 0. Two bugs were identified in the vector and Alfvén reduction minimum calculations where `v1` is used twice instead of `v2`.

## dconst-race-condition-test

The `dconst-race-condition-test` is an exploratory CUDA diagnostic that investigates whether race conditions can occur when writing to a device constant memory symbol (`__device__ __constant__`) from two concurrent streams via `cudaMemcpyToSymbolAsync`. It launches a 1-billion-iteration delay kernel on one stream to ensure concurrency, while the other stream writes a competing value to the same constant symbol. The test checks whether the kernel on stream0 always reads the value written on its own stream, or whether it can observe the interleaved write from stream1. It is a standalone `.cu` file compiled directly with `nvcc` and has no CMake integration.

## fortrantest

The `fortrantest` is Astaroth's simplest Fortran integration test, a 23-line executable that validates the Fortran interface to the core library. It creates a 128³ host mesh configuration, updates built-in parameters, creates and prints device info, then destroys the device. The test uses Astaroth's Fortran API via an `include "astaroth.f90"` file with `iso_c_binding` for C interoperability. It exercises only the device creation/teardown lifecycle without any actual simulation or integration steps.

## mpi_fullgriderror_test

The `mpi_fullgriderror_test` is Astaroth's most concise MPI correctness diagnostic, running a single GPU integration step (`dt = FLT_EPSILON`) across an MPI-distributed grid and writing per-element differences to a binary file (`full_grid_error.out`) for numerical diagnosis of GPU vs CPU discrepancies. By using the minimal time step, it isolates floating-point ordering differences (reduction order, rounding) from physical integration drift. The error dump is written only by rank 0, covering only rank 0's local grid chunk, making it a targeted diagnostic tool rather than a comprehensive correctness validator.

## mpitest

The `mpitest` is Astaroth's most comprehensive MPI integration test at 423 lines, verifying the full GPU-accelerated grid stack: mesh load/store, disk I/O, periodic boundary conditions (via both legacy and DSL task graph APIs), dual integration verification paths (legacy `acGridIntegrate` and DSL task graph with 3 substeps), and all three reduction types (scalar, vector, Alfvén). It uses `dt = FLT_EPSILON` for integration and configurable ULP error tolerance, with domain decomposition using Morton-curve strategy across ranks. The sample also supports runtime DSL compilation (`AC_RUNTIME_COMPILATION`) and provides per-substep timing instrumentation via `MPI_Wtime()`.

## multikerneltest

The `multikerneltest` is a minimal standalone device-level kernel execution test for Astaroth's DSL runtime, implementing a 3D Fibonacci cellular automaton on a 64³ grid using three DSL kernels (`clear`, `set`, `step`) compiled at build time from `fibonacci.ac`. It uses the lower-level `acDevice*` API directly, bypassing MPI and multi-node infrastructure, and is designed to be built independently without `astaroth_utils` by pointing CMake options to this directory. The test runs 20 iterations of the `step` kernel, computing and printing the minimum grid value after each iteration as a simple correctness signal.

## stencil-loader

The `stencil-loader` is a minimal MPI-based correctness verification tool that loads GPU stencil derivative coefficients (1st/2nd order, cross derivatives, and 6th-order upwind derivatives) from host CPU code and verifies the resulting GPU computation against a CPU reference across four test phases: boundary conditions, time integration, scalar reductions, and vector reductions. It uses random initial conditions with a fixed seed and `FLT_EPSILON` timesteps to exercise the stencil pipeline while minimizing error accumulation. The program demonstrates both bulk stencil upload (`acGridLoadStencils`) and per-stencil upload patterns, though only 9 stencils are tested and a bug was identified where `VTXBUF_UUX` is used instead of `VTXBUF_UUZ` in the minimum magnitude vector reduction test.

# Summary of Sample Directories — Performance Benchmarks

This document summarizes all `ANALYSIS_*.md` files for sample directories in the **Performance Benchmarks** category. Each entry is 2–4 sentences.

## benchmark

The `benchmark` sample is Astaroth's core performance benchmarking tool for measuring full integration step execution time across varying problem sizes and processor counts to evaluate strong and weak scaling characteristics. It supports both strong scaling (fixed total grid, decreasing per-process workload) and weak scaling (fixed per-process grid, growing total domain), running 5 warmup + 100 timed integration steps each consisting of 3 RK3 substeps. An optional CPU-vs-GPU verification path compares host-side `acHostIntegrateStep` results against GPU task graph execution with configurable ULP tolerance. Results are appended to `scaling-benchmark.csv` with percentile statistics (50th and 90th) and per-kernel sanity checks for boundary conditions, integration, and reductions.

## benchmark-device

The `benchmark-device` sample is a per-kernel-level GPU benchmarking tool that directly benchmarks individual Astaroth device kernels via `acDeviceLaunchKernel`, as opposed to the higher-level `samples/benchmark/` which measures full integration steps through task graphs. It measures `singlepass_solve` kernel launch performance, reports optimal thread block dimensions (TPB), and includes a three-stage CPU-vs-GPU verification path (load/store, boundary conditions, integration) with configurable grid dimensions and random seeds. A companion Python script (`mhd.py`) benchmarks a nonlinear MHD forward model across PyTorch, TensorFlow, and JAX against a NumPy/SciPy reference, providing an independent ML-framework comparison for convolution-based stencil computations.

## benchmark-node

The `benchmark-node` sample is a single-node, single-process GPU benchmarking tool that exercises Astaroth's Node API (`acNode*`), a higher-level wrapper encapsulating device management, stream operations, and kernel launches within a single-node context. Written entirely in C (unlike the C++ `benchmark` and `benchmark-device` samples), it benchmarks full 3-substep integration via `acNodeIntegrateSubstep`, reports percentile statistics (min, median, 90th percentile, max), and includes load/store and boundary condition verification stages. Unlike the other benchmarks, it passes explicit volume bounds (`n_min`, `n_max`) to substep calls rather than relying on implicit kernel launch dimensions.

## benchmark-thrust

The `benchmark-thrust` sample is a CUDA Thrust library performance benchmark that provides a dual-purpose comparison between raw `thrust::reduce` throughput and Astaroth's `acMapCrossReduce` operation over vertex and profile buffer arrays. Written in CUDA C++ (`.cu` file), it first benchmarks Thrust parallel reductions on `thrust::device_vector` arrays across multiple z-slices, then runs the same logical operation through Astaroth's VBA/PBA layer with explicit cross-section mapping and scratchpad buffers. The comparison isolates the overhead introduced by Astaroth's buffer array abstraction, including 2D cross-section extraction from 3D arrays and reduction into profile buffers.

## bwtest-mpi

The `bwtest-mpi` is a standalone MPI/CUDA bandwidth benchmark measuring communication and data transfer performance across different memory allocation strategies (standard `malloc`, `cudaMalloc`, pinned device/host memory) and send/recv patterns (blocking, non-blocking, `MPI_Sendrecv`, all-to-all multiple). It uses a ring topology with a 12 MiB block size and runs 10 warmup + 100 timed iterations, reporting bandwidth in GiB/s for host-host, device-device, and device-host transfers in both unidirectional and bidirectional modes. The tool documents key performance findings: pinned memory is essential for RDMA (40 GiB/s internode vs 5–6 GiB/s standard device memory), and both sender and receiver buffers must be pinned for internode optimization.

## cubtest

The `cubtest` is a standalone CUDA C++ benchmark/utility for NVIDIA CUB and AMD HIP/CUB segmented reduction operations, containing two programs: `cubtest.cu` (a large-scale 256³ × 32 double benchmark allocating ~4.3 GiB with 8,192 segments) and `cubtest-basic-working.cu` (a small 10-element correctness test with hardcoded values). Both test `cub::DeviceSegmentedReduce::Sum` using CUB's two-pass pattern (query workspace size, then execute with allocated temporary storage), with the large benchmark running 10 iterations for performance measurement and assertion-based verification. The code supports cross-platform compilation via `USE_HIP` preprocessor flag, targeting AMD MI200 (`gfx90a`) through `hipcc`, and uses only the standalone `timer_hires.h` utility from Astaroth without linking the core library.

## genbenchmarkscripts

The `genbenchmarkscripts` is a C utility that generates SLURM batch scripts (`benchmark_*.sh`) for running Astaroth benchmarks on HPC clusters like the Puhti system. It produces 7 scripts for processor counts of 1, 2, 4, 8, 16, 32, and 64, each containing SLURM directives, module loading commands, and 22 benchmark run commands spanning domain decomposition, mesh size scaling, stencil orders, communication timings, and weak scaling tests. The scripts are tightly coupled to the target cluster's hardware configuration, including hardcoded GPU types (V100), node exclusion lists, and UCX environment tuning for multi-node GPU direct communication.

## heat-equation

The `heat-equation` is a comprehensive cross-framework benchmarking suite for stencil-based heat equation solvers, comparing Astaroth's performance against PyTorch, TensorFlow, JAX, cuDNN, and AMD MIOpen on the same mathematical problem — second-order spatial derivative (Laplacian) discretization via finite differences. Its C driver (`main.c`) benchmarks Astaroth kernels with stencils of orders 0, 2, 4, 6, and 8 in 1D, 2D, and 3D, with optional correctness verification against a host-side finite-difference solver. Python scripts (`heat-equation.py`, `heat-equation-separable.py`) benchmark deep learning framework equivalents with SciPy reference verification, while standalone `cudnn-heat-equation/` and `miopen-heat-equation/` subdirectories provide native CUDA and AMD HIP convolution benchmarks.

## microbenchmark

The `microbenchmark` is a GPU memory bandwidth measurement suite for 1D stencil sum operations, testing effective bandwidth across varying stencil radii (0 to 1024), problem sizes (default 128M elements), and three implementation strategies (single-element-per-thread, 4-element vectorization, and templated compile-time unrolling). It supports both NVIDIA CUDA and AMD HIP (via Roctracer), with single/double precision selectable at compile time. The execution pipeline includes device info printing, threadblock-size autotuning, CPU model verification, and CSV output of per-iteration bandwidth measurements. Key optimizations include 256-byte alignment padding and shared memory tiling.

## microbenchmark-nn

The `microbenchmark-nn` is a GPU microbenchmark that maps 1D stencil sum operations onto NVIDIA cuDNN and AMD MIOpen deep learning library convolution APIs, enabling direct comparison between hand-written CUDA kernels (in the `microbenchmark` sample) and vendor-optimized implementations. It uses a filter tensor of all ones to make the convolution equivalent to a stencil sum, with cuDNN using NCHW 4D tensors and MIOpen supporting algorithm selection (GEMM, Direct, FFT, Winograd, ImplicitGEMM). The benchmark defaults to a smaller 128K domain size and explicitly rejects double precision for both backends, producing CSV output with backend-identification, milliseconds, and effective bandwidth per iteration.

## mpi-io

The `mpi-io` is an MPI-IO benchmark for Astaroth's mesh I/O subsystem, measuring aggregate read and write bandwidth of vertex buffer fields across configurable 3D grid dimensions. It runs `n` MPI processes, each writing its assigned domain chunk via `acGridAccessMeshOnDiskSynchronous`, supporting both centralized I/O (rank 0 handles all file operations) and distributed I/O (each rank writes its own data) modes. The benchmark performs a full write sweep followed by a read sweep, with optional CPU-GPU verification of load/store and disk round-trip correctness. Output is per-process CSV with timing and bandwidth measurements.

## mpi-io-multithreaded

The `mpi-io-multithreaded` extends the `mpi-io` sample by using `MPI_Init_thread` with `MPI_THREAD_MULTIPLE` support, enabling concurrent compute and asynchronous disk I/O on separate threads per MPI rank. It validates correctness through CPU-GPU load/store checks, integration step verification, and an asynchronous I/O test where 10 integration steps are run while an async mesh write completes in the background — testing I/O-hide-behind-compute overlap. The sample produces timing printed to stdout rather than CSV files, and its async I/O mechanism (`acGridWriteMeshToDiskLaunch` + `acGridDiskAccessSync`) is the primary active benchmark, with synchronous I/O commented out.

## mpi_reduce_bench

The `mpi_reduce_bench` measures the latency of scalar and vector MPI reduction operations across an MPI-distributed grid, reporting the 90th percentile of 100 iterations as the metric. It benchmarks five reduction types (MAX, MIN, SUM, RMS, RMS_EXP) on both single fields (VTXBUF_UUX) and three-field vectors (VTXBUF_UUX/UUY/UUZ), appending results to a CSV file with benchmark label, test label, process count, and latency in milliseconds. The benchmark is designed to profile reduction cost scaling with process count and includes an SLURM submission script (`mpibench.sh`) for HPC cluster execution on V100-based GPU nodes.

# Summary of Sample Directories — Development & Debugging Tools

This document summarizes all `ANALYSIS_*.md` files for sample directories in the **Development & Debugging Tools** category. Each entry is 2–4 sentences.

## pc-varfile-import

The `pc-varfile-import` is an MPI-accelerated mesh initialization utility that reads initial condition data from an external varfile format and loads it into Astaroth's GPU grid infrastructure for large-scale cosmological or MHD simulations. It reads all specified fields from a single binary file (`var.dat`), populates the GPU mesh, computes per-field statistics (min/RMS/max), writes Astaroth-compatible snapshots to disk, and exports per-slice data for visualization. The sample defaults to a 4096³ grid and supports an optional `LMAGNETIC` compile-time flag to include magnetic vector potential fields alongside the 4 hydrodynamic fields. Notable issues include a hardcoded HPC cluster path, no command-line arguments, and statistics computed for all DSL fields including uninitialized ones.

## taskgraph_example

The `taskgraph_example` is a minimal MPI-based demonstration of Astaroth's low-level task graph API, the programmatic (non-DSL) way to construct and execute multi-stage GPU computation pipelines. It builds an `AcTaskGraph` from three operations — halo exchange for ghost cell synchronization, periodic boundary condition application on all 3D faces, and a compute kernel — executed for exactly 3 iterations on 7 vertex buffers (density + 3 velocity components + 3 magnetic vector potentials). The example contrasts with the DSL system (`acGetDSLTaskGraph`) by showing the fine-grained control over task ordering, field dependencies, and subregion decomposition that the lower-level API provides.

## taskgraph_print

At 40 lines, `taskgraph_print` is the smallest sample in the suite — an MPI-based introspection utility that retrieves the default task graph via `acGridGetDefaultTaskGraph()` and writes its dependency structure to per-process text files (`dependencies_pid_<rank>.txt`). It does not execute any physics or transfer mesh data to the GPU, making it purely a development and debugging tool. A commented-out alternative shows the capability to also dump the full task list including kernel names, fields, and parameters.

## taskgraph_test

The `taskgraph_test` is an MPI-based diagnostic utility nearly identical in structure to `taskgraph_print` but differs by printing the default task graph's dependency structure to stdout instead of writing to files, using `acGraphPrintDependencies()` rather than `acGraphWriteDependencies()`. Unlike `taskgraph_print`, it creates, randomizes, and loads a full host mesh to GPU, exercising the complete initialization pipeline even though the task graph is only introspected and never executed. Its design makes it suitable for interactive debugging and CI/CD verification where immediate console feedback is preferred over file inspection.

## taskgraph_trace

The `taskgraph_trace` is an MPI-based performance profiling utility that records per-kernel execution timing data from Astaroth's task graph execution. It creates two randomized meshes but loads only one to GPU, runs 100 warm-up iterations to trigger JIT compilation and auto-optimizations, then enables tracing and executes one measured iteration with results written to per-process trace files. The 100 warm-up iterations ensure all kernel compilation, auto-tuning, and memory allocation overhead is excluded from the measured trace data, and the per-process file naming enables correlation with the corresponding dependency structure files.
