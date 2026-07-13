# ACC-Runtime Analysis

## Overview

**acc-runtime** is the GPU runtime and standard library for **Astaroth**, a domain-specific language (DSL) and source-to-source compiler that converts DSL kernels into CUDA/HIP GPU kernels. It is a high-performance computing framework designed for stencil computations in computational physics.

- **License**: GPL v3+
- **Authors**: Johannes Pekkila, Miikka Vaisala, Touko Puro (2014-2026)
- **Languages**: C (compiler), CUDA, C++ (runtime API), C (DSL host code)
- **Target GPUs**: NVIDIA (CUDA) and AMD (HIP)
- **Build System**: CMake 3.21+

---

## Purpose

Astaroth DSL compiles to GPU kernels that perform stencil-based computations on structured grids. The runtime provides:

1. **Memory management** on the GPU for fields, arrays, and profiles
2. **Kernel execution** infrastructure (launch, synchronization, streams)
3. **Communication** via MPI for multi-node, multi-GPU simulations
4. **Reduction operations** (min, max, sum) across subdomains
5. **Boundary condition** handling
6. **Task graph** scheduling for dependent kernel sequences
7. **Raytracing** for sequential-dependency computations
8. **Numerical methods** library (derivatives, operators, solvers)

---

## Directory Structure

```
acc-runtime/
|-- CMakeLists.txt              # Top-level build configuration
|-- README.md                   # Extensive DSL documentation
|-- LICENCE.md                  # GPL v3+ license
|-- standalone_params.h         # Standalone build parameters
|
|-- acc/                        # DSL compiler (source-to-source)
|   |-- ac.l                    # Flex lexer (tokenizer)
|   |-- ac.y                    # Bison parser (grammar)
|   |-- ast.h                   # Abstract syntax tree node definitions
|   |-- codegen.c/.h            # GPU code generation
|   |-- implementation.c/.h     # Kernel implementation logic
|   |-- stencilgen.c            # Stencil generation
|   |-- tinyexpr.c/.h           # Expression parser library
|   |-- hash.h, hashtable.h     # Hash table utilities
|   |-- vecs.h, string_vec.h    # Dynamic vector utilities
|   |-- warp_reduce.h           # CUDA warp-level reduction primitives
|   |-- 3d_caching_implementations.h  # 3D caching strategies
|   |-- create_node.h/.h        # AST node creation helpers
|   |-- mem_access_helper_funcs.h     # Memory access helpers
|   |-- CMakeLists.txt
|
|-- api/                        # Runtime API (device + host code)
|   |-- acc_runtime.h/.cu       # Main runtime API header/implementation
|   |-- datatypes.h             # Core data type definitions
|   |-- host_datatypes.h        # Host-side data types
|   |-- device_headers.h        # CUDA/HIP device header abstraction
|   |-- device_details.h        # GPU device configuration details
|   |-- errchk.h                # Error checking macros
|   |-- func_attributes.h       # Function attribute declarations
|   |-- mapreduce.cuh           # CUDA map-reduce kernels
|   |-- fft.cc                  # GPU FFT implementation
|   |-- arrays.cc               # Array memory management
|   |-- vba.cc                  # Vertex buffer array management
|   |-- random.cu/.cuh          # GPU random number generation
|   |-- common_kernels.cc/.h    # Common GPU kernels
|   |-- math_utils.h/.h         # Mathematical utilities
|   |-- static_analysis.cc/.h   # Compile-time analysis
|   |-- astaroth_analysis.h     # Analysis infrastructure
|   |-- CMakeLists.txt
|
|-- built-in/                   # Built-in DSL types and functions
|   |-- typedefs.h              # Type definitions (real2, real3, Field, etc.)
|   |-- variables.h             # Built-in variable declarations
|   |-- functions.h             # Built-in function declarations
|   |-- intrinsics.h            # DSL intrinsics
|   |-- kernels.h               # Built-in kernel declarations
|   |-- CMakeLists.txt
|
|-- stdlib/                     # Numerical methods library
|   |-- operators.h             # Gradient, divergence, curl, laplace
|   |-- derivs.h                # Derivative stencils (various orders)
|   |-- general_derivs.h        # General derivative operators
|   |-- general_operators.h     # General operators
|   |-- integrators.h           # Numerical integration methods
|   |-- optimized_integrators.h # Optimized integration kernels
|   |-- bc.h                    # Boundary condition functions
|   |-- poisson.h               # Poisson equation solvers
|   |-- compact_poisson_operators.h # Compact stencil Poisson operators
|   |-- geometric_multigrid.h   # Geometric multigrid methods
|   |-- geometric_multigrid_core.h  # Multigrid core implementation
|   |-- cg.h                    # Conjugate gradient solver
|   |-- bicgstab.h              # BiCGSTAB solver
|   |-- average.h               # Averaging operators
|   |-- smooth_max.h            # Smooth max operators
|   |-- slope_limited_diffusion.h   # Slope-limited diffusion
|   |-- shock.h                 # Shock-capturing methods
|   |-- radiation_ray.h         # Radiation ray tracing
|   |-- sink_particle.h         # Sink particle methods
|   |-- spherical_harmonics.h   # Spherical harmonic transforms
|   |-- colors.h                # Color functions
|   |-- units.h                 # Physical units
|   |-- map.h                   # Data mapping utilities
|   |-- grid_extension.h        # Grid extension functions
|   |-- grid_transfer_functions.h   # Grid transfer operators
|   |-- grid/                   # Grid-specific utilities
|   |-- general_grid/           # General grid utilities
|   |-- math/                   # Mathematical utilities (FFT, stencils, interpolation, planes)
|   |-- utils/                  # General utility functions and intrinsics
|   |-- CMakeLists.txt
|
|-- Pencil/                     # Pencil communication infrastructure
|   |-- prerequisites.h
|   |-- PC_modulepardecs.h
|
|-- samples/                    # Example DSL programs
|   |-- mhd/                    # MHD (magnetohydrodynamics) solver
|   |-- mhd_modular/            # Modular MHD solver
|   |-- mhd_4thord/             # 4th-order MHD solver
|   |-- conv-slab/              # Convection slab solver
|   |-- planes/                 # 2D planes solver
|   |-- planes-2d/              # 2D planes solver
|   |-- blur/                   # Image blur example
|   |-- random-walker/          # Random walker simulation
|   |-- test/                   # Test solver
|   |-- gputest/                # GPU testing solver
|
|-- tests/                      # Test DSL source files
    |-- 0.ac - 6.ac             # Basic test programs
    |-- mhd.ac                  # MHD test
    |-- overload-test/          # Function overloading test
    |-- syntaxtest.sh           # Syntax test runner
```

---

## DSL Compiler (`acc/`)

The `acc` directory contains the **Astaroth DSL compiler**, a source-to-source compiler that transforms `.ac` (ASTAROTH Kernel) files into CUDA/HIP kernels.

### Lexer (`ac.l`)
- Flex-based lexer tokenizing the DSL language
- Recognizes: types (`Field`, `real`, `int`, `bool`, `Matrix`, `Tensor`, `Profile<...>`), qualifiers (`Kernel`, `dconst`, `gmem`, `elemental`, `communicated`, `auxiliary`, `output`, `global`), keywords (`if`, `for`, `while`, `return`), stencil directives (`Stencil`, `Sum`, `Max`), raytracing (`Raytrace`), and computation constructs (`ComputeSteps`, `BoundConds`)

### Parser (`ac.y`)
- Bison-based parser for the DSL grammar
- Constructs an AST from the tokenized input

### AST (`ast.h`)
- Abstract syntax tree node definitions for all DSL constructs
- Supports: fields, stencils, kernels, functions, expressions, loops, control flow

### Code Generation (`codegen.c`)
- Transforms the DSL AST into CUDA/HIP source code
- Handles memory allocation, kernel launch, halo communication

### Implementation (`implementation.c`)
- Kernel implementation logic, including field swapping, stencil application, and reduction operations

### Stencil Generation (`stencilgen.c`)
- Generates GPU-optimized stencil code with proper caching strategies

### Expression Parsing (`tinyexpr.c`)
- Integrated tinyexpr library for parsing mathematical expressions in DSL

---

## Runtime API (`api/`)

The `api` directory provides the **C/C++ runtime interface** for hosting and executing Astaroth-generated kernels.

### Core Types
- `AcMeshInfo`: Configuration structure holding grid dimensions, decomposition, parameters, and runtime settings
- `VertexBufferArray`: GPU memory buffers for fields (input/output, single/half precision, complex)
- `ProfileBufferArray`: Memory for profile data (1D/2D slices of fields)
- `AcKernel`: Handle to a compiled kernel
- `KernelReduceOutput`: Reduction output metadata
- `AcCompInfo`: Configuration parameter info (config, loaded, default value states)

### Key API Functions
| Function | Purpose |
|----------|---------|
| `acInitializeRuntimeMPI` | Initialize runtime with MPI communication |
| `acRuntimeInit` / `acRuntimeQuit` | Runtime lifecycle |
| `acVBACreate` / `acVBADestroy` | Vertex buffer array management |
| `acAllocateArrays` / `acFreeArrays` | GPU memory allocation |
| `acLoadRealUniform` ... `acStoreRealUniform` | Host-to-device / device-to-host data transfer |
| `acLoadStencil` / `acStoreStencil` | Runtime stencil coefficient loading |
| `acLaunchKernel` | Execute a GPU kernel |
| `acLoadMeshInfo` / `acVerifyMeshInfo` | Mesh configuration management |
| `acGetKernels` / `acGetOptimizedKernel` | Kernel retrieval and optimization |

### Device Code
- `acc_runtime.cu`: CUDA implementation of the runtime API
- `mapreduce.cuh`: CUDA map-reduce operations
- `random.cu`: GPU random number generation using cuRAND
- `fft.cc`: FFT operations using cuFFT

### Caching & Optimization
- `static_analysis.cc`: Compile-time field usage analysis for dead field detection
- `common_kernels.cc`: Shared utility kernels (flush, reset, etc.)

---

## Built-in DSL Definitions (`built-in/`)

### Types (`typedefs.h`)
- **Vector types**: `real2`, `real3`, `real4`, `real5`, `complex`, `complex_float`
- **Field types**: `Field`, `Field2`, `Field3`, `Field4`, `FieldSymmetricTensor`
- **Profile types**: `Profile<Z>`, `VecZProfile`, etc.
- **Coordinate systems**: `AcCoordinateSystem` (Cartesian, Spherical, Cylindrical)
- **Boundary conditions**: `AcBoundary` (bitmask for X/Y/Z top/bottom faces)
- **Reductions**: `AcReductionPostProcessingOp` (none, RMS, sqrt, radial window RMS)
- **Precision**: `AcPrecision` (real, single, half)
- **Red-black coloring**: `AcRedBlackState` (none, red, black)
- **Decomposition**: `AcDecomposeStrategy`, `AcProcMappingStrategy`, `AcMPICommStrategy`

### Functions & Variables
- `functions.h`: Built-in function signatures
- `variables.h`: Built-in global variables (grid dimensions, decomposition info)
- `intrinsics.h`: DSL intrinsic operations
- `kernels.h`: Built-in kernel declarations

---

## Standard Library (`stdlib/`)

The `stdlib` provides **reusable numerical methods** for computational physics simulations.

### Differential Operators
- `derivs.h`: 4th, 5th, 6th order derivative stencils (derx, dery, derz, derxx, etc.)
- `general_derivs.h`: General derivative operator definitions
- `operators.h`: gradient, divergence, curl, laplace, hessian, traceless strain, rate-of-strain
- `general_operators.h`: Dimension-aware operator implementations (handles inactive dimensions)
- `pc_derivs.h`: Petrov-Galerkin derivative operators

### Linear Solvers
- `poisson.h`: Poisson equation solvers
- `compact_poisson_operators.h`: Compact stencil Poisson operators
- `cg.h`: Conjugate gradient solver
- `bicgstab.h`: BiCGSTAB solver
- `geometric_multigrid.h`: Geometric multigrid V-cycle / F-cycle
- `geometric_multigrid_core.h`: Multigrid core (smoothing, restriction, prolongation)

### Integration
- `integrators.h`: Numerical integration along rays and paths
- `optimized_integrators.h`: Optimized integration kernels

### Boundary Conditions
- `bc.h`: Dirichlet, Neumann, symmetry, periodic, outflow boundary condition functions

### Specialized Methods
- `shock.h`: Shock detection and capturing
- `slope_limited_diffusion.h`: Slope-limited diffusion operators
- `smooth_max.h`: Smooth maximum operators
- `radiation_ray.h`: Radiation transport ray tracing
- `sink_particle.h`: Sink particle methods (astrophysical simulations)
- `spherical_harmonics.h`: Spherical harmonic transforms
- `average.h`: Cell-averaging operators
- `units.h`: Physical unit conversion
- `grid_extension.h`: Grid extension/manipulation
- `grid_transfer_functions.h`: Multigrid grid transfer operators

### Math Subdirectory (`stdlib/math/`)
- `stencils.h`: Mathematical stencil definitions
- `interpolation.h`: Interpolation functions
- `fft.h`: FFT utilities
- `funcs.h`: General mathematical functions
- `planes/`: 2D plane operations

### Utils Subdirectory (`stdlib/utils/`)
- `intrinsics.h`, `funcs.h`, `kernels.h`: General utility primitives

---

## Key DSL Concepts

### Fields
Scalar or vector arrays mapped to GPU memory buffers. Declared as `Field name` or `Field3 name` (vector field). Automatically managed with input/output buffer swapping.

### Stencils
Named kernel operations that access neighboring field values:
```
Stencil derx {
    [0][0][-3] = -AC_ds.x*DER1_3,
    [0][0][-1] = -AC_ds.x*DER1_1,
    [0][0][1]  =  AC_ds.x*DER1_1,
    [0][0][3]  =  AC_ds.x*DER1_3
}
```

### Kernels
GPU-parallel functions defined in DSL:
```
Kernel my_kernel() {
    write(ux, derx(uy))
}
```

### ComputeSteps
Sequence of kernel calls with automatic dependency analysis, halo exchange, and boundary condition application.

### Profiles
1D/2D arrays that depend on fewer dimensions than full 3D fields.

### Raytracing
For computations with sequential dependencies along specific directions.

### Reductions
Global or subdomain-wide min/max/sum operations on field values.

---

## Build Configuration

### CMake Options
| Option | Description | Default |
|--------|-------------|---------|
| `USE_HIP` | Use AMD HIP instead of CUDA | OFF |
| `DOUBLE_PRECISION` | Use double precision for `real` | ON |
| `AC_STENCIL_ORDER` | Stencil radius (halo width) | 3 |
| `AC_RUNTIME_COMPILATION` | Enable runtime recompilation | OFF |
| `AC_MPI_ENABLED` | Enable MPI communication | ON |
| `OPTIMIZE_FIELDS` | Dead field elimination optimization | ON |
| `OPTIMIZE_ARRAYS` | Dead array elimination | ON |

### GPU Architectures
- CUDA default: `80` (Ampere), or `61;70;80` for older architectures
- HIP default: `gfx90a;gfx908` (AMD MI200/MI100)

### Build Flow
1. User writes `.ac` files in a directory
2. Compiler (`acc/`) preprocesses and generates CUDA/HIP code
3. Runtime (`api/`) compiles GPU code and links with host API
4. Result: shared library (`libkernels.so`) with generated kernels

---

## Sample Programs

| Sample | Description |
|--------|-------------|
| `mhd/`, `mhd_modular/` | Magnetohydrodynamics solver (full MHD system) |
| `mhd_4thord/` | 4th-order MHD solver |
| `conv-slab/` | Convection slab test case |
| `planes/`, `planes-2d/` | 2D plane wave / perturbation solver |
| `blur/` | Image blur kernel (simple stencil example) |
| `random-walker/` | Random walker diffusion simulation |

---

## Testing

Tests are `.ac` DSL source files (`tests/*.ac`) that exercise various language features:
- `0.ac` - `6.ac`: Progressive feature tests
- `mhd.ac`: MHD solver test
- `overload-test/`: Function overloading tests
- `syntaxtest.sh`: Shell script to run syntax validation

---

## Technical Characteristics

### Performance Features
- **Source-to-source compilation**: Generates native CUDA/HIP, no JIT interpretation overhead
- **Intelligent caching**: Automatic shared memory caching of field values based on stencil access patterns
- **3D caching strategies**: Configurable caching optimizations for different access patterns
- **Warp-level primitives**: Efficient intra-thread-block reductions
- **Multi-precision**: Support for double, single, and half precision fields
- **Task graph**: Automatic dependency resolution and overlap of communication with computation

### Architecture
- **Stream processing model**: Each GPU thread processes one grid vertex
- **Stencil-based computation**: Regular memory access patterns enabling efficient caching
- **MPI parallelism**: Domain decomposition across MPI ranks with halo exchange
- **Field buffer swapping**: Double-buffered fields for in-place updates

### Generated Code
Intermediate files produced during compilation:
- `user_kernels.ac.pp_stage*`: Preprocessed DSL sources
- `user_kernels.inc.raw`: Unformatted generated CUDA
- `user_kernels_backup.inc`: Formatted generated CUDA/CPU code
- `user_defines.inc`: Project-wide defines from DSL
- `user_kernels.inc`: Final compiled kernels
