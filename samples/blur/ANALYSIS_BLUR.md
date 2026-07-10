# ⚠️ WARNING ⚠️
# **This is a LLM based analysis.**
# **THE BRANCH SHOULD NEVER BE MERGED TO master OR develop BRANCHES!**

# LLM Information
- **Model:** Qwen3.6-35B-A3B

# Overview
The `blur` sample is Astaroth's minimal stencil-based image/video blurring example. It demonstrates the Astaroth DSL's **Stencil** construct — a named, weighted convolution pattern that can be applied to any field. The primary sample (`samples/blur/`) launches a blur kernel on a device, stores the result to host, and prints the resulting grid to stdout for visual inspection.

The blur concept appears in multiple locations across the repository, each serving a different purpose:
- **`acc-runtime/samples/blur/blur.ac`** — The reference implementation: a 7-point 3D blur stencil (weight `1/7` on the center point and its 6 face-adjacent neighbors).
- **`samples/blur/`** — The primary C host program that launches the DSL blur kernel and prints results.
- **`samples/plasma-meets-ai-workshop/blur/`** — An in-progress workshop exercise with an incomplete 2D blur stencil (only 3 of 9 coefficients filled, kernel body empty).
- **`samples/plasma-meets-ai-workshop/model-examples/blur/`** — A completed 9-point 2D blur stencil (weight `1/9` on a 3×3 x-y cross pattern), used as a model example with a working C host that generates 20 output snapshots.

# Directory Structure & File Descriptions

## Primary Sample (`samples/blur/`)

| File | Description |
| :--- | :--- |
| `CMakeLists.txt` | Build configuration; compiles `main.c` into the `blur` executable with position-independent code, linked against `astaroth_core` and `astaroth_utils`. |
| `main.c` | C host program: creates a mesh, initializes all cells to `1.0`, loads to GPU, launches the blur kernel, stores back to host, and prints the resulting grid. |

## DSL Implementations

| File | Description |
| :--- | :--- |
| `acc-runtime/samples/blur/blur.ac` | Reference 7-point 3D blur stencil (`COEFF = 1/7`). Uses `Field IMAGE0` and a `Stencil blur` definition. |
| `samples/plasma-meets-ai-workshop/blur/blur.ac` | **Incomplete exercise** — 2D blur with `COEFF = 1/9`, only 3 of 9 stencil coefficients defined; kernel body is a TODO. |
| `samples/plasma-meets-ai-workshop/model-examples/blur/blur.ac` | **Completed** 9-point 2D blur stencil (`COEFF = 1/9`) — full 3×3 x-y cross pattern. Uses `Field field0` and a `Stencil filter` definition. |

## Workshop C Programs

| File | Description |
| :--- | :--- |
| `samples/plasma-meets-ai-workshop/blur/blur.c` | Workshop exercise C program — kernel launch is commented out as an exercise for participants. Runs a 20-step blur loop with periodic boundary conditions. |
| `samples/plasma-meets-ai-workshop/model-examples/blur/blur.c` | Completed C program — runs the blur kernel 19 times (steps 1–19), writing each step's mesh to a file (`astaroth_0.dat` through `astaroth_19.dat`). |

## Build Scripts

| File | Description |
| :--- | :--- |
| `test-builds/blur/build.sh` | Build script that configures CMake with `PROGRAM_MODULE_DIR=samples/blur`, `DSL_MODULE_DIR=acc-runtime/samples/blur`, MPI enabled, and memory access optimizations disabled. |

# Build Configuration

## Primary Sample
```bash
cmake -B build -S $AC_HOME \
  -DOPTIMIZE_MEM_ACCESSES=OFF \
  -DMPI_ENABLED=ON \
  -DBUILD_STANDALONE=OFF \
  -DBUILD_MHD_SAMPLES=OFF \
  -DPROGRAM_MODULE_DIR=${AC_HOME}/samples/blur \
  -DDSL_MODULE_DIR=${AC_HOME}/acc-runtime/samples/blur \
  && cmake --build build -j
```

Key CMake variables:
| Variable | Value | Description |
| :--- | :--- | :--- |
| `PROGRAM_MODULE_DIR` | `samples/blur` | Directory containing the C host program (`main.c`). |
| `DSL_MODULE_DIR` | `acc-runtime/samples/blur` | Directory containing the DSL source (`blur.ac`). |
| `OPTIMIZE_MEM_ACCESSES` | `OFF` | Memory access optimizations disabled (for clarity/debugging). |
| `MPI_ENABLED` | `ON` | MPI support enabled. |
| `BUILD_STANDALONE` | `OFF` | Standalone sample excluded. |
| `BUILD_MHD_SAMPLES` | `OFF` | MHD samples excluded. |

## Workshop Variants
Both workshop CMakeLists.txt files use:
```cmake
set(BUILD_SAMPLES OFF CACHE BOOL "Turn off samples" FORCE)
set(DSL_MODULE_DIR ../samples/plasma-meets-ai-workshop/blur CACHE BOOL "" FORCE)
add_executable(blur blur.c)
target_link_libraries(blur astaroth_core astaroth_utils)
```

# DSL Breakdown

## Reference Implementation: `blur.ac` (7-point 3D)

```dsl
Field IMAGE0

#include "../../stdlib/map.h"

#define COEFF (1. / 7.)

Stencil blur {
  [-1][0][0] = COEFF,
  [1][0][0]  = COEFF,
  [0][-1][0] = COEFF,
  [0][1][0]  = COEFF,
  [0][0][-1] = COEFF,
  [0][0][1]  = COEFF,
  [0][0][0]  = COEFF
}

Kernel blur_kernel() {
  write(IMAGE0, blur(IMAGE0))
}
```

### Stencil Geometry
The `blur` stencil is a **7-point 3D cross** pattern, applying equal weight (`1/7`) to the center voxel and its 6 face-adjacent neighbors:

```
       [0][0][1]  = 1/7    (top neighbor)
          |
[−1][0][0] — [0][0][0] — [1][0][0]   (x-axis neighbors + center)
          |
       [0][0][−1] = 1/7    (bottom neighbor)

      [0][−1][0] = 1/7    (y-axis neighbor)
      [0][1][0]  = 1/7    (y-axis neighbor)
```

This is equivalent to a 3×3×3 box blur where only the center cross is included (not corners or edges).

### Operation
The kernel `blur_kernel()` calls `write(IMAGE0, blur(IMAGE0))`, which:
1. Applies the `blur` stencil convolution to `IMAGE0` at every grid point.
2. Writes the result to the **out** buffer of `IMAGE0`.

## Completed 2D Variant: `model-examples/blur/blur.ac` (9-point x-y)

```dsl
Field field0

hostdefine STENCIL_ORDER (2)

#define COEFF (1. / 9.)
Stencil filter {
    [0][-1][-1] = COEFF,
    [0][-1][0] = COEFF,
    [0][-1][1] = COEFF,
    [0][0][-1] = COEFF,
    [0][0][0] = COEFF,
    [0][0][1] = COEFF,
    [0][1][-1] = COEFF,
    [0][1][0] = COEFF,
    [0][1][1] = COEFF
}

Kernel blur() {
    write(field0, filter(field0))
}
```

### Stencil Geometry
The `filter` stencil is a **9-point 2D box blur** in the x-y plane:

```
z-fixed plane:
  [0][1][−1]  [0][1][0]  [0][1][1]    = 1/9 each
  [0][0][−1]  [0][0][0]  [0][0][1]    = 1/9 each
  [0][−1][−1] [0][−1][0] [0][−1][1]  = 1/9 each
```

This is a standard 3×3 averaging filter. Only the z-index `[0]` is used, making it a 2D operation applied across all z-slices.

## Incomplete Workshop Variant: `plasma-meets-ai-workshop/blur/blur.ac`

```dsl
Field field0

hostdefine STENCIL_ORDER (2)

#define COEFF (1. / 9.)
Stencil filter {
    [0][-1][-1] = (1.0 / 9.0),
    [0][-1][0] = (1.0 / 9.0),
    // EXERCISE: fill in the proper coefficients for a 2D blur filter in the x-y dimension
}

Kernel blur() {
    // EXERCISE: use function `write(Field, AcReal)` to write the result of the filter operation
    // to the output buffer
    //write(..., ...)
}
```

The user is expected to:
1. Fill in the remaining 6 stencil coefficients.
2. Implement the kernel body with `write(field0, filter(field0))`.

# Host Program: `samples/blur/main.c`

## Command-Line Interface

| Argument | Default | Description |
| :--- | :--- | :--- |
| `<nx> <ny>` | Required | Grid dimensions in x and y. z is hardcoded to `1`. |

Usage: `./blur <nx> <ny>`

## Program Flow

### 1. Configuration
1. `AcMeshInfo info` — Stack-allocated mesh info.
2. Parse `nx` and `ny` from arguments, set `AC_nlocal = (nx, ny, 1)`.
3. `acHostUpdateParams(&info)` — Compute derived parameters.

### 2. Host Mesh Creation & Initialization
1. `acHostMeshCreate(info, &mesh)` — Allocate host mesh.
2. `acHostMeshSet(1, &mesh)` — Set **all cells** of the mesh to `1.0`.
   - This is the key test input: a uniform field of `1.0` values. After blurring, every cell should still be `1.0` (the stencil weights sum to `7 × 1/7 = 1.0` for the 3D variant).

### 3. Device Setup
1. `acDeviceCreate(0, info, &device)` — Create device (GPU 0).
2. `acDevicePrintInfo(device)` — Print device information.
3. `acDeviceLoadMesh(device, STREAM_DEFAULT, mesh)` — Upload the mesh to the GPU.

### 4. Blur Kernel Launch
1. Compute launch bounds:
   - `nn_min = (AC_nmin.x, AC_nmin.y, AC_nmin.z)` — Lower bound of the computation region.
   - `nn_max = (AC_nlocal_max.x, AC_nlocal_max.y, AC_nlocal_max.z)` — Upper bound.
2. `acDeviceLaunchKernel(device, STREAM_DEFAULT, blur_kernel, Volume(n_min), Volume(n_max))` — Launch the blur kernel over the specified volume.
3. `acDeviceSwapBuffers(device)` — Swap in/out buffers (the blur kernel writes to the **out** buffer).

### 5. Store & Print Results
1. `acDeviceStoreMesh(device, STREAM_DEFAULT, &mesh)` — Download results to host.
2. Print `"Store complete"`.
3. Iterate over all vertex buffer handles (`NUM_VTXBUF_HANDLES`) and all z-slices:
   - Print depth header: `"==== DEPTH %d ====="`.
   - If `print_bounds == true`: iterate over the full local mesh dimensions (`AC_mlocal`).
   - Otherwise: iterate over the computation bounds (`nn_min` to `nn_max`).
   - Print each value with `%.3g` format.

### 6. Cleanup
1. `acDeviceDestroy(&device)` — Destroy the device.
2. `acHostMeshDestroy(&mesh)` — Free host mesh.

## Output Example

For a `blur 4 4` invocation, the output would look like:

```
Store complete
==== DEPTH 0 ====
1 1 1 1
1 1 1 1
1 1 1 1
1 1 1 1
```

Since the input is uniform `1.0` and the stencil weights sum to `1.0`, every output value remains `1.0`. This serves as a simple correctness check: the blur should be identity on constant fields.

# Workshop C Program: `model-examples/blur/blur.c`

This completed variant generates a sequence of 20 snapshots:

1. Create device and mesh.
2. Randomize the mesh (`acHostMeshRandomize(&mesh)`).
3. Load to device, apply periodic boundaries.
4. Write initial snapshot (`astaroth_0.dat`).
5. Loop 19 times (steps 1–19):
   - Launch blur kernel.
   - Swap buffers, apply periodic boundaries, synchronize.
   - Store to host and write to file (`astaroth_i.dat`).
6. Cleanup.

The output files can be viewed as a time-evolving video of the blur diffusion process.

# Key Astaroth APIs Used

| Function | Description |
| :--- | :--- |
| `acHostMeshCreate(info, &mesh)` | Allocate host-side mesh buffers. |
| `acHostMeshSet(value, &mesh)` | Set all mesh values to a scalar. |
| `acHostMeshRandomize(&mesh)` | Fill mesh with random values. |
| `acHostMeshWriteToFile(mesh, step)` | Write mesh to a data file. |
| `acHostMeshDestroy(&mesh)` | Free host mesh. |
| `acDeviceCreate(device_id, info, &device)` | Create a GPU device context. |
| `acDevicePrintInfo(device)` | Print device information. |
| `acDeviceLoadMesh(device, stream, mesh)` | Upload host mesh to device. |
| `acDeviceStoreMesh(device, stream, &mesh)` | Download device mesh to host. |
| `acDeviceLaunchKernel(device, stream, kernel, volume_start, volume_end)` | Launch a DSL kernel over a 3D volume. |
| `acDeviceSwapBuffers(device)` | Swap in/out field buffers. |
| `acDevicePeriodicBoundconds(device, stream, m0, m1)` | Apply periodic boundary conditions. |
| `acDeviceSynchronizeStream(device, stream)` | Wait for GPU stream completion. |
| `acDeviceDestroy(&device)` | Destroy device context. |
| `acVertexBufferIdx(i, j, k, info)` | Compute linear index from 3D coordinates. |

# Key DSL Constructs

| Construct | Description |
| :--- | :--- |
| `Field NAME` | Declares a 3D discretized field with in/out double buffers. |
| `Stencil NAME { ... }` | Defines a named weighted convolution pattern with relative offset→coefficient mappings. |
| `Kernel NAME() { ... }` | Defines a kernel that runs over all grid points; uses `write(Field, value)` to update. |
| `write(Field, value)` | Writes `value` to the **out** buffer of `Field` at all grid points. |
| `StencilName(Field)` | Applies a named stencil convolution to a field, returning the result at each grid point. |
| `hostdefine MACRO value` | Defines a compile-time macro accessible from both host and device code. |
| `#include "stdlib/map.h"` | Includes the map module with field operations (`ac_map_get_value`, etc.). |

# Stencil Weight Sum Check

| Stencil | Points | Weight | Sum |
| :--- | :--- | :--- | :--- |
| 3D cross (`blur.ac`) | 7 | `1/7` | `7 × 1/7 = 1.0` |
| 2D box (`model-examples/blur/blur.ac`) | 9 | `1/9` | `9 × 1/9 = 1.0` |

Both stencils have weights that sum to `1.0`, ensuring they are **averaging** operations that preserve constant fields (no gain or attenuation of uniform values).

# Key Dependencies
- `astaroth.h` — Core API (Device, mesh, kernel launch).
- `astaroth_utils.h` — Utility functions (mesh creation, host updates).
- `timer_hires.h` — High-resolution timer (declared but not actively used in the primary `main.c`).
- `<float.h>` — `FLT_EPSILON` (included but unused in primary `main.c`).
- `<stdlib.h>` — Standard library.
