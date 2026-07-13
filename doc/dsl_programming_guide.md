# Astaroth DSL Programming Guide

A beginner-friendly introduction to the Astaroth Domain-Specific Language (DSL), for writing GPU-accelerated stencil kernels for structured-grid computational physics.

## Table of Contents

1. [What Is Astaroth?](#1-what-is-astaroth)
2. [Your First Kernel](#2-your-first-kernel)
3. [Core Concepts](#3-core-concepts)
   - [Fields](#31-fields)
   - [Stencils](#32-stencils)
   - [Kernels](#33-kernels)
   - [Field Access](#34-field-access)
4. [Data Types](#4-data-types)
   - [Primitives](#41-primitives)
   - [Vectors and Matrices](#42-vectors-and-matrices)
   - [Fields](#43-fields)
   - [Arrays](#44-arrays)
   - [Structs and Enums](#45-structs-and-enums)
5. [Variables and Constants](#5-variables-and-constants)
   - [Global Variables](#51-global-variables)
   - [Type Qualifiers](#52-type-qualifiers)
   - [Local Variables](#53-local-variables)
   - [Host Defines and Preprocessor](#54-host-defines-and-preprocessor)
6. [Math and Functions](#6-math-and-functions)
   - [Arithmetic](#61-arithmetic)
   - [Vector Operations](#62-vector-operations)
   - [Math Functions](#63-math-functions)
   - [Casting](#64-casting)
   - [Functions](#65-functions)
   - [Elemental Functions](#66-elemental-functions)
7. [Control Flow](#7-control-flow)
   - [Conditionals](#71-conditionals)
   - [Loops](#72-loops)
8. [Built-in Variables](#8-built-in-variables)
   - [Grid and Vertex](#81-grid-and-vertex)
   - [Coordinate Functions](#82-coordinate-functions)
   - [Constants](#83-constants)
   - [Printing](#84-printing)
9. [Boundary Conditions and ComputeSteps](#9-boundary-conditions-and-computesteps)
   - [Boundary Conditions](#91-boundary-conditions)
   - [ComputeSteps](#92-computesteps)
10. [Reductions](#10-reductions)
11. [Profiles](#11-profiles)
12. [Complete Example: A Simple Physics Kernel](#12-complete-example-a-simple-physics-kernel)
13. [Tips and Best Practices](#13-tips-and-best-practices)
14. [Further Reading](#14-further-reading)

---

## 1. What Is Astaroth?

Astaroth is a source-to-source compiler for **stencil computations** on GPUs. You write your physics in a high-level `.ac` file using the Astaroth DSL, and the `acc/` compiler translates it into CUDA or HIP kernel code. The runtime handles GPU memory, halo exchange between subdomains, boundary conditions, and multi-process synchronization.

The DSL is inspired by C syntax and stream-processing models: each kernel runs on every grid vertex in parallel, reading and writing **fields** using **stencils** that reach out to neighboring vertices.

### Quick Start

```bash
mkdir build && cd build
cmake -DDSL_MODULE_DIR=<path to directory containing .ac files> ..
make -j
```

The DSL files must use the `.ac` extension. Each directory should contain one `.ac` file.

---

## 2. Your First Kernel

Here is the simplest useful kernel — a 7-point average of a scalar field:

```c
Field IMAGE0

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

Three pieces:

1. **Field** — `IMAGE0` declares a 3D scalar field (a distributed array on the GPU).
2. **Stencil** — `blur` defines a weighted sum over the 7 neighbors (center + 6 face-adjacent).
3. **Kernel** — `blur_kernel` is the GPU program. `write(IMAGE0, blur(IMAGE0))` applies the blur at every vertex.

---

## 3. Core Concepts

### 3.1 Fields

A `Field` is a scalar array stored on the GPU, sized to the local subdomain plus halo cells. Vector fields combine three scalar fields:

```c
Field ux, uy, uz
const Field3 velocity = Field3(ux, uy, uz)
```

`Field2`, `Field3`, and `Field4` are struct types with `.x`, `.y`, `.z` (and `.w`) components.

To read a field's value at the current vertex, use `value()`:

```c
real density = value(VTXBUF_LNRHO)
```

You can also use a `Field` directly in expressions — `ux + 1.0` is automatically converted to `value(ux) + 1.0`.

**Double-buffering**: Fields have two buffers — current (input) and previous (output from the last time step). Use `previous(field)` to read the old time step and `write(field, ...)` to write the new one.

### 3.2 Stencils

A stencil is a weighted sum over neighboring vertices. It is the **only** way to access field values at positions other than the current vertex.

```c
// Central first derivative (4th order, 6-point)
Stencil derx {
    [-3][0][0] = -DER1_3,
    [-2][0][0] = -DER1_2,
    [-1][0][0] = -DER1_1,
    [1][0][0]  =  DER1_1,
    [2][0][0]  =  DER1_2,
    [3][0][0]  =  DER1_3
}
```

The offset `[z][y][x]` is relative to the current vertex. Stencils act as functions: `derx(ux)` applies the x-derivative stencil to the `ux` field.

By default, stencil reduction is summation. Use `Max Stencil` for element-wise maximum:

```c
Max Stencil max5 {
    [1][0][0] = 1,
    [-1][0][0] = 1,
    [0][1][0] = 1,
    [0][-1][0] = 1,
    [0][0][1] = 1,
    [0][0][-1] = 1,
    [0][0][0] = 1
}
```

### 3.3 Kernels

A kernel is a GPU-executed function called from the host:

```c
Kernel solve() {
    write(lnrho, rk3(previous(lnrho), value(lnrho), continuity()))
    write(ux, rk3(previous(ux), value(ux), mom.x))
    write(uy, rk3(previous(uy), value(uy), mom.y))
    write(uz, rk3(previous(uz), value(uz), mom.z))
}
```

Kernels can accept parameters:

```c
Kernel solve_with_dt(int step_num, real dt) {
    write(lnrho, rk3(previous(lnrho), value(lnrho), continuity(), step_num, dt))
}
```

Utility kernels are skipped during field liveness analysis:

```c
utility Kernel reset() {
    for field in 0:NUM_FIELDS {
        write(Field(field), 0.0)
    }
}
```

### 3.4 Field Access

| Function | Meaning |
|----------|---------|
| `write(field, expr)` | Write `expr` to `field` at the current vertex |
| `value(field)` | Read the current (input-buffer) value |
| `current(field)` | Alias for `value(field)` |
| `previous(field)` | Read the value from the previous time step |

---

## 4. Data Types

### 4.1 Primitives

| Type | Description |
|------|-------------|
| `real` | Floating-point (double by default, single if `DOUBLE_PRECISION=OFF`) |
| `int` | Integer |
| `bool` | Boolean |
| `float`, `double`, `long`, `long long` | Standard C/C++ types |

### 4.2 Vectors and Matrices

| Type | Description |
|------|-------------|
| `real2` | 2D vector: `.x`, `.y` |
| `real3` | 3D vector: `.x`, `.y`, `.z` |
| `real4` | 4D vector: `.x`, `.y`, `.z`, `.w` |
| `int3` | 3D integer vector |
| `bool3` | 3D boolean vector |
| `Matrix` | 3×3 symmetric matrix: `.data[i][j]`, `.row(i)` |

Construct vectors with `{}` or constructors:

```c
real3 v = {1.0, 2.0, 3.0}
real3 w = real3(1.0, 2.0, 3.0)
```

### 4.3 Fields

| Type | Description |
|------|-------------|
| `Field` | 3D scalar field |
| `Field2` | 2-component vector field |
| `Field3` | 3-component vector field |
| `Field4` | 4-component vector field |
| `FieldSymmetricTensor` | Symmetric tensor field |
| `Profile<Z>` | 1D field varying only along Z |

### 4.4 Arrays

```c
int arr0 = [1, 2, 3]                // compile-time array
real arr1 = [1.0, 2.0, 3.0]         // type inferred
gmem real global_arr[AC_nx]         // global GPU memory array
```

### 4.5 Structs and Enums

```c
struct TimeParams {
    real dt
    real current_time
}

enum SUBSTEP {
    FIRST,
    SECOND,
    THIRD
}
```

---

## 5. Variables and Constants

### 5.1 Global Variables

All global variables need an explicit type:

```c
Field VTXBUF_LNRHO, VTXBUF_UUX, VTXBUF_UUY, VTXBUF_UUZ
real AC_cs_sound, AC_cp_sound
real AC_gamma, AC_nu_visc
```

### 5.2 Type Qualifiers

| Qualifier | Meaning |
|-----------|---------|
| `const` | Compile-time constant |
| `dconst` (default) | Device constant memory — fast GPU reads |
| `gmem` | Global GPU memory — for arrays accessed with different indices at different vertices |
| `run_const` | Constant during simulation execution |
| `input` | Host-side input parameter, not allocated on GPU |
| `output` | Reduction result output |
| `auxiliary` | Same input and output buffer |
| `single_precision` / `half_precision` | Override field precision |

### 5.3 Local Variables

Inside kernels, local variables can use implicit typing:

```c
Kernel my_kernel() {
    real a, b
    Field local_field
    a = 1.0
    b = 2.0
    int dsx = 0       // explicit type
    inferred = 1      // type inferred as int
}
```

### 5.4 Host Defines and Preprocessor

```c
hostdefine AC_INTEGRATION_ENABLED
hostdefine LSHOCK (1)
hostdefine LMAGNETIC (1)
```

These are visible in both host and device code and are used with `#if` for conditional compilation.

```c
#include "../../stdlib/operators.h"
#define COEFF (1. / 7.)
#define UU Field3(ux, uy, uz)

#if LSHOCK
#include "shock_capture.h"
#endif
```

Includes are searched relative to `DSL_MODULE_DIR`.

---

## 6. Math and Functions

### 6.1 Arithmetic

Standard C operators work: `+`, `-`, `*`, `/`, `%`, `+=`, `-=`.

```c
a = 1 + 2 * (3 - 4) / 5.0
```

### 6.2 Vector Operations

```c
real3 a = {1.0, 2.0, 3.0}
real3 b = {4.0, 5.0, 6.0}

c = dot(a, b)           // dot product
c = cross(a, b)         // cross product
c = norm(v)             // magnitude
```

### 6.3 Math Functions

Available via `#include` from stdlib:

```c
sqrt(x), pow(x, n), exp(x), log(x)
sin(x), cos(x), tan(x), atan2(y, x)
sinh(x), cosh(x), tanh(x)
fabs(x), abs(x)
min(a, b), max(a, b)
ceil(x)
```

### 6.4 Casting

```c
real_val = real(int_val)
int_val = int(real_val)
real3_vec = real3(1, 2, 3)
complex_val = (complex){1.0, 2.0}
```

### 6.5 Functions

Functions return values and are not executed directly — they are called from kernels:

```c
gradient(s) {
    return real3(derx(s), dery(s), derz(s))
}

divergence(v) {
    return derx(v.x) + dery(v.y) + derz(v.z)
}

curl(v) {
    return real3(dery(v.z) - derz(v.y),
                 derz(v.x) - derx(v.z),
                 derx(v.y) - dery(v.x))
}
```

Function parameters are passed by constant reference — they cannot be modified inside the function. Use temporary variables for intermediate results.

Function overloading (resolved by signature):

```c
fn()  { return 1 }
fn(x) { return x }
fn(x, y) { return x + y }
```

### 6.6 Elemental Functions

Declaring a function `elemental` lets it operate on vectors and arrays automatically:

```c
elemental abs(real x)
{
    return fabs(x)
}

// Works with real3 too:
real3 v = real3(-1.0, 2.0, -3.0)
result = abs(v)    // produces real3(1.0, 2.0, 3.0)
```

---

## 7. Control Flow

### 7.1 Conditionals

```c
if step > 0 {
    return s1 + beta[step] * roc * dt
} else {
    return s1
}

if time > switch_point {
    forcing_step = forcing(time)
}
```

Comparison operators: `==`, `!=`, `>`, `<`, `>=`, `<=`. Boolean expressions with `&&`, `||`.

### 7.2 Loops

**Range-based for loop** (Python-style, exclusive upper bound):

```c
for i in 0:10 {
    print("%d", i)
}

for i in 1:NGHOST+1 {
    field[vertexIdx.x][vertexIdx.y][i] = 0.0
}

for field in 0:NUM_FIELDS {
    write(Field(field), 0.0)
}
```

**C-style for loop**:

```c
for (int i = 0; i < N; i++) {
    // ...
}
```

**For over arrays**:

```c
int arr = [1, 2, 3]
for val in arr {
    print("%d\n", val)
}
```

---

## 8. Built-in Variables

The runtime provides these variables automatically inside kernels:

### 8.1 Grid and Vertex

| Variable | Type | Description |
|----------|------|-------------|
| `vertexIdx` | `int3` | Local vertex index within subdomain |
| `globalVertexIdx` | `int3` | Global vertex index |
| `AC_ds` | `real3` | Grid spacing (dx, dy, dz) |
| `AC_dsmin` | `real` | Minimum grid spacing |
| `AC_inv_ds` | `real3` | Inverse grid spacing |
| `AC_ngrid` | `int3` | Total grid dimensions |
| `AC_nxgrid, AC_nygrid, AC_nzgrid` | `int` | Total grid dimensions |
| `AC_nx, AC_ny, AC_nz` | `int` | Local grid dimensions |
| `AC_nmin, AC_nmax` | `int3` | Local min/max indices |
| `AC_n1, AC_n2` | `int` | Min/max indices |
| `AC_mx, AC_my, AC_mz` | `int` | Max indices |
| `NGHOST` | `int` | Number of halo cells |

### 8.2 Coordinate Functions

```c
real3 pos = grid_position()       // physical position of current vertex
real3 center = grid_center()      // grid center
```

### 8.3 Constants

| Constant | Description |
|----------|-------------|
| `AC_REAL_PI` | π |
| `AC_REAL_MAX` | Maximum representable real |
| `AC_REAL_MIN` | Minimum positive real |
| `AC_REAL_EPSILON` | Machine epsilon |

### 8.4 Printing

```c
print("density = %.14e\n", value(VTXBUF_LNRHO))
print("Thread (%d,%d,%d)\n", vertexIdx.x, vertexIdx.y, vertexIdx.z)
```

---

## 9. Boundary Conditions and ComputeSteps

### 9.1 Boundary Conditions

`BoundConds` declares how halo cells are filled at domain boundaries:

```c
BoundConds bcs {
    periodic(BOUNDARY_XYZ)           // all directions periodic
    // or:
    ac_bc_sym(BOUNDARY_Y)            // symmetry BC on Y
    periodic(BOUNDARY_Z)             // periodic on Z
    ac_const_bc(BOUNDARY_X, FIELD, 0.0)  // constant value BC
}
```

Custom boundary condition functions:

```c
bc_sym_z(AcBoundary boundary, Field field) {
    if (boundary == BOUNDARY_Z_BOT) {
        for i in 0:NGHOST {
            field[vertexIdx.x][vertexIdx.y][NGHOST-i] = field[vertexIdx.x][vertexIdx.y][NGHOST+i]
        }
    } else {
        for i in 0:NGHOST {
            field[vertexIdx.x][vertexIdx.y][AC_nz_max+i] = field[vertexIdx.x][vertexIdx.y][AC_nz_max-i]
        }
    }
}
```

### 9.2 ComputeSteps

`ComputeSteps` defines a sequence of kernels to run, with automatic halo exchange and boundary condition application between them:

```c
ComputeSteps AC_rhs(bcs) {
    twopass_solve_intermediate(AC_FIRST_SUBSTEP, AC_dt)
    twopass_solve_final(AC_FIRST_SUBSTEP, AC_current_time, AC_dt)

    twopass_solve_intermediate(AC_SECOND_SUBSTEP, AC_dt)
    twopass_solve_final(AC_SECOND_SUBSTEP, AC_current_time, AC_dt)

    twopass_solve_intermediate(AC_THIRD_SUBSTEP, AC_dt)
    twopass_solve_final(AC_THIRD_SUBSTEP, AC_current_time, AC_dt)
}
```

The compiler analyzes input/output fields to determine the correct ordering of halo exchanges, boundary condition application, and kernel launches. Dependencies between kernels are inferred automatically — if kernel A writes a field that kernel B reads, A runs first.

---

## 10. Reductions

Reduce operations aggregate values across the subdomain or the entire grid:

```c
output real max_vel
output global real global_max_vel
output real sum_vel

Kernel compute_reductions() {
    reduce_max(norm(UU), max_vel)            // subdomain max
    reduce_max(norm(UU), global_max_vel)     // global max across all ranks
    reduce_sum(ux, sum_vel)                   // subdomain sum
}
```

Supported operations: `reduce_min`, `reduce_max`, `reduce_sum`, `reduce_sum_add`.

Reduction results can be read in subsequent kernels via `write()`.

---

## 11. Profiles

`Profile<Z>` (and `Profile<X>`, `Profile<XY>`, etc.) are 1D or 2D fields that vary only along specified axes:

```c
Profile<Z> density_profile

Kernel reduce_to_profile() {
    reduce_sum(VTXBUF_LNRHO, density_profile)  // sum over x and y, left with z-dependent result
}
```

Profiles enable efficient z-averaging and similar operations on 3D grids.

---

## 12. Complete Example: A Simple Physics Kernel

This example shows the full structure of a computational physics kernel:

```c
// === Constants and Parameters ===
input real AC_dt
const real INV_DS = 1. / 0.04908738521
#define DER1_1 (INV_DS * 3.0 / 4.0)
#define DER1_2 (INV_DS * -3.0 / 20.0)
#define DER1_3 (INV_DS * 1.0 / 60.0)

// === Fields ===
Field VTXBUF_LNRHO, VTXBUF_ENTROPY
Field VTXBUF_UUX, VTXBUF_UUY, VTXBUF_UUZ
Field VTXBUF_AX, VTXBUF_AY, VTXBUF_AZ

#define UU Field3(VTXBUF_UUX, VTXBUF_UUY, VTXBUF_UUZ)
#define AA Field3(VTXBUF_AX, VTXBUF_AY, VTXBUF_AZ)

// === Stencils (derivative operators) ===
Stencil derx {
    [-3][0][0] = -DER1_3, [-2][0][0] = -DER1_2, [-1][0][0] = -DER1_1,
    [1][0][0]  =  DER1_1, [2][0][0]  =  DER1_2, [3][0][0]  =  DER1_3
}

Stencil dery {
    [0][-3][0] = -DER1_3, [0][-2][0] = -DER1_2, [0][-1][0] = -DER1_1,
    [0][1][0]  =  DER1_1, [0][2][0]  =  DER1_2, [0][3][0]  =  DER1_3
}

Stencil derz {
    [0][0][-3] = -DER1_3, [0][0][-2] = -DER1_2, [0][0][-1] = -DER1_1,
    [0][0][1]  =  DER1_1, [0][0][2]  =  DER1_2, [0][0][3]  =  DER1_3
}

// === Helper Functions ===
gradient(s) {
    return real3(derx(s), dery(s), derz(s))
}

divergence(v) {
    return derx(v.x) + dery(v.y) + derz(v.z)
}

curl(v) {
    return real3(dery(v.z) - derz(v.y),
                 derz(v.x) - derx(v.z),
                 derx(v.y) - dery(v.x))
}

// Continuity equation
continuity() {
    return -dot(vecvalue(UU), gradient(VTXBUF_LNRHO)) - divergence(UU)
}

// Entropy equation
entropy() {
    return -dot(vecvalue(UU), gradient(VTXBUF_ENTROPY))
}

// === RK3 Time Integrator ===
rk3(s0, s1, roc) {
    real alpha = 0., -5./9., -153. / 128.
    real beta  = 1. / 3., 15./ 16., 8. / 15.

    if step > 0 {
        return s1 + beta[step] * ((alpha[step] / beta[step - 1]) * (s1 - s0) + roc * dt)
    } else {
        return s1 + beta[step] * roc * dt
    }
}

// === Kernel ===
Kernel solve() {
    write(VTXBUF_LNRHO, rk3(previous(VTXBUF_LNRHO), value(VTXBUF_LNRHO), continuity()))
    write(VTXBUF_ENTROPY, rk3(previous(VTXBUF_ENTROPY), value(VTXBUF_ENTROPY), entropy()))

    write(VTXBUF_UUX, rk3(previous(VTXBUF_UUX), value(VTXBUF_UUX), continuity()))
    write(VTXBUF_UUY, rk3(previous(VTXBUF_UUY), value(VTXBUF_UUY), continuity()))
    write(VTXBUF_UUZ, rk3(previous(VTXBUF_UUZ), value(VTXBUF_UUZ), continuity()))

    write(VTXBUF_AX, rk3(previous(VTXBUF_AX), value(VTXBUF_AX), curl(AA).x))
    write(VTXBUF_AY, rk3(previous(VTXBUF_AY), value(VTXBUF_AY), curl(AA).y))
    write(VTXBUF_AZ, rk3(previous(VTXBUF_AZ), value(VTXBUF_AZ), curl(AA).z))
}

// === Boundary Conditions & Execution Graph ===
BoundConds bcs {
    periodic(BOUNDARY_XYZ)
}

ComputeSteps AC_rhs(bcs) {
    solve()
}
```

---

## 13. Tips and Best Practices

1. **Prefer `bool` over `int` for flags** — the compiler can optimize better when told a variable is truly boolean.

2. **Prefer `enum` over named integers** — similarly, enums give the compiler more information for optimization.

3. **Use `gmem` for arrays accessed with varying indices** — `dconst` arrays are stored in GPU constant memory and perform poorly when different threads access different elements.

4. **Use `elemental` functions for vectorizable operations** — `elemental abs(real x)` works automatically on `real3` and arrays.

5. **Use `hostdefine` with `#if` for conditional physics** — compile different physics modules (MHD, self-gravity, shock capture) by toggling host defines.

6. **Use `utility Kernel` for field resets** — prevents the compiler from eliminating fields during liveness analysis.

7. **Check intermediate files on errors** — if the compiler fails, inspect `user_kernels.ac.pp_stage*`, `user_kernels.inc.raw`, and `user_kernels_backup.inc` in the build directory.

8. **Stencils are functions, not loops** — a stencil like `derx(ux)` computes the weighted sum over neighbors in a single expression; you do not write explicit neighbor loops.

9. **Function parameters are constant** — you cannot modify them inside a function. Use local temporaries for intermediate values.

10. **For field iteration**, use `for field in 0:NUM_FIELDS { write(Field(field), ...); }` to apply the same operation to all fields.

---

## 14. Further Reading

- **Reference guide**: `ANALYSIS_DSL.md` — complete syntax and type reference
- **Official docs**: `acc-runtime/README.md` — build instructions and API details
- **Samples**: `acc-runtime/samples/` — MHD solver, blur, planes, random walker
- **Tests**: `acc-runtime/tests/` — small focused test cases
- **Standard library**: `acc-runtime/stdlib/` — reusable operators, derivatives, solvers, and boundary conditions
