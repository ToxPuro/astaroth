# Astaroth DSL Reference and Analysis

## Overview

Astaroth DSL (file extension `.ac`) is a **domain-specific language** for writing GPU-accelerated stencil kernels targeting structured grids in computational physics. It is a source-to-source DSL: `.ac` files are compiled by the `acc/` compiler into CUDA or HIP source code, which is then compiled and linked against the runtime library (`acc-runtime/api/`).

The language sits between a numerical library and a programming language — it lets physicists express PDE discretizations and time-stepping schemes as GPU kernels, while the compiler and runtime handle memory management, halo exchange, boundary conditions, and multi-GPU communication.

---

## Syntax and Data Types

### Primitive Types

| Type | Description |
|------|-------------|
| `real` | Floating-point scalar (double or single depending on `DOUBLE_PRECISION` CMake flag) |
| `real3` | 3D vector `(x, y, z)` of reals |
| `real2` | 2D vector `(x, y)` of reals |
| `real4` | 4D vector `(x, y, z, w)` of reals |
| `real5` | 5D vector of reals |
| `int` | Integer scalar |
| `int3` | 3D vector of ints |
| `bool` | Boolean scalar |
| `complex` | Complex number |
| `complex_float` | Single-precision complex |
| `Matrix` | 3x3 symmetric tensor / matrix type with `.row(i)`, `.data[i][j]` access |
| `Tensor` | Higher-order tensor type |

### Field Types

| Type | Description |
|------|-------------|
| `Field` | 3D scalar field mapped to GPU memory |
| `Field2` | 2-component vector field |
| `Field3` | 3-component vector field |
| `Field4` | 4-component vector field |
| `FieldSymmetricTensor` | Symmetric tensor field |
| `Profile<Z>` | 1D field varying only along Z axis |
| `VecZProfile` | Profile of vector along Z |

### Array Types

| Type | Description |
|------|-------------|
| `real[]`, `int[]`, `bool[]` | Runtime-sized arrays |
| `gmem real[]` | Global memory arrays |
| `gmem dynamic real[]` | Dynamically allocated global arrays |
| `ScalarArray` | Profile/1D array type |

### Multi-dimensional Arrays

```c
real arr2d[AC_nx][AC_ny]      // 2D static array
int arr_dynamic[NCHEM]         // Array with DSL constant size
gmem dynamic real global_radius_start[1]  // Dynamic global array
```

---

## Variable Declarations and Qualifiers

### Compile-time Constants

```c
const real AC_const_real = 2.0
const int AC_first = 2, AC_second = 2
const int3 AC_const_int3 = {1, 2, 3}
const bool AC_const_bool = false
const real3 AC_const_real3 = {1.0, 2.0, 3.0}

// Multi-dimensional const arrays
const int AC_2d_ints = [[1,2,3], [2,3,4]]
const int AC_const_ints = [1, 2, 3]
const real AC_const_reals = [1.0, 2, 0, 3.0]
const bool AC_const_bool_arr = [true, false]
const real3 AC_const_real3_arr = [{1.0, 2.0, 3.0}, {1.0, 2.0, 3.0}]

// Derived constants
const real AC_cv_sound = AC_cp_sound / AC_gamma
const Field3 UU = {VTXBUF_UUX, VTXBUF_UUY, VTXBUF_UUZ}
```

### Runtime Constants (set from host, constant on device)

```c
run_const real AC_nu_visc
run_const real AC_cs_sound
run_const int  AC_runtime_int
run_const real3 AC_runtime_real3
run_const bool AC_lspherical_coords
```

### Device Constants

```c
dconst real4 AC_real_4
dconst complex AC_complex_dconst
```

### Global Memory Variables

```c
gmem real AC_real_gmem_arr[3]
gmem real AC_gmem_const_dims[AC_nx_const][AC_ny_const]
gmem real3 AC_real3_gmem_arr[3]
```

### Input Parameters (from host)

```c
input int AC_step_number
input AC_SUBSTEP_NUMBER AC_SUBSTEP
input real AC_current_time
input real AC_dt
```

### Auxiliary Fields

```c
auxiliary Field BFIELDX
auxiliary Field BFIELDY
auxiliary Field BFIELDZ
const Field3 BFIELD = {BFIELDX, BFIELDY, BFIELDZ}
```

### Host-visible Defines

```c
hostdefine AC_INTEGRATION_ENABLED
hostdefine LDENSITY (1)
hostdefine LHYDRO (1)
hostdefine LMAGNETIC (1)
hostdefine LENTROPY (0)
hostdefine LSHOCK (0)
hostdefine LBFIELD (1 && LMAGNETIC)  // Boolean expression allowed
```

### Preprocessor Defines

```c
#define DER1_3 (INV_DS * 1. / 60.)
#define INV_DS (1. / 0.04908738521)
#define NO (0)
#define uu Field3(ux, uy, uz)
#define aa Field3(ax, ay, az)
#define STENCIL_ORDER (2)
```

### Structs and Enums

```c
struct TimeParams {
    real dt;
    real current_time;
}

enum AC_SUBSTEP_NUMBER {
    AC_FIRST_SUBSTEP,
    AC_SECOND_SUBSTEP,
    AC_THIRD_SUBSTEP
}

enum TOP_BOT {
    AC_top,
    AC_bot
}
```

### Local Variables (inside Kernels)

```c
Kernel kernel() {
    Field lnrho
    Field ux, uy, uz
    real a, b, c, d
    real3 vec = real3(1.0, 2.0, 3.0)
    int dsx = 0
    real dsy
    real arr1 = 1.0, 2.0, 3.0

    // Implicit typing (no keyword needed)
    var0 = 1         // inferred as int
    var1 = 1.0       // inferred as real
    var2 = real(1)   // explicit cast
    vec0 = real3(1, 2, 3)  // explicit cast
}
```

### Numeric Literals

```c
dt = 1.       // real literal (trailing zero optional)
dtt = 1e6     // scientific notation
dttt = 1e-128 // negative exponent
dtttt = 1.f   // explicit single precision
dttttt = 0.2d // explicit double precision
step = 0      // integer literal
```

---

## Stencils

A **stencil** defines a named weighted sum over neighboring grid vertices. It is the fundamental computational primitive.

### Basic Stencil

```c
Stencil blur {
  [-1][0][0] = COEFF,
  [1][0][0]  = COEFF,
  [0][-1][0] = COEFF,
  [0][1][0]  = COEFF,
  [0][0][-1] = COEFF,
  [0][0][1]  = COEFF,
  [0][0][0]  = COEFF
}
```

### First Derivative (Central Difference, 4th Order)

```c
Stencil derx {
    [-3][0][0] = -DER1_3,
    [-2][0][0] = -DER1_2,
    [-1][0][0] = -DER1_1,
    [1][0][0]  =  DER1_1,
    [2][0][0]  =  DER1_2,
    [3][0][0]  =  DER1_3
}

// Stencil coefficients defined via C preprocessor macros
#define DER1_3 (INV_DS * 1. / 60.)
#define DER1_2 (INV_DS * -3. / 20.)
#define DER1_1 (INV_DS * 3. / 4.)
```

### Derivative Stencils (per-axis)

```c
// d/dx
Stencil derx {
    [-3][0][0] = -DER1_3,
    [-2][0][0] = -DER1_2,
    [-1][0][0] = -DER1_1,
    [1][0][0]  =  DER1_1,
    [2][0][0]  =  DER1_2,
    [3][0][0]  =  DER1_3
}

// d/dy
Stencil dery {
    [0][-3][0] = -DER1_3,
    [0][-2][0] = -DER1_2,
    [0][-1][0] = -DER1_1,
    [0][1][0]  =  DER1_1,
    [0][2][0]  =  DER1_2,
    [0][3][0]  =  DER1_3
}

// d/dz
Stencil derz {
    [0][0][-3] = -DER1_3,
    [0][0][-2] = -DER1_2,
    [0][0][-1] = -DER1_1,
    [0][0][1]  =  DER1_1,
    [0][0][2]  =  DER1_2,
    [0][0][3]  =  DER1_3
}
```

### Second Derivative Stencils

```c
Stencil derxx {
    [-3][0][0] = DER2_3,
    [-2][0][0] = DER2_2,
    [-1][0][0] = DER2_1,
    [0][0][0]  = DER2_0,
    [1][0][0]  = DER2_1,
    [2][0][0]  = DER2_2,
    [3][0][0]  = DER2_3
}
```

### Cross-derivative Stencils

```c
Stencil derxy {
    [-3][-3][0] =  DERX_3,
    [-2][-2][0] =  DERX_2,
    [-1][-1][0] =  DERX_1,
    [0][0][0]   =  DERX_0,
    [1][1][0]   =  DERX_1,
    [2][2][0]   =  DERX_2,
    [3][3][0]   =  DERX_3,
    // Anti-symmetric parts
    [-3][3][0]  = -DERX_3,
    [-2][2][0]  = -DERX_2,
    [-1][1][0]  = -DERX_1,
    [1][-1][0]  = -DERX_1,
    [2][-2][0]  = -DERX_2,
    [3][-3][0]  = -DERX_3
}
```

### Max Stencil

```c
Max Stencil max5 {
    [-3][0][0] = 1,
    [-2][-2][-1] = 1,
    [-2][-2][0]  = 1,
    // ... many more non-zero offsets ...
    [3][0][0] = 1
}
```

### Large 3D Stencils (7x7x7 smoothing kernel)

The `smooth_kernel` in `smooth_kernel.ach` is a 343-point stencil (7x7x7) generated by the Python script `stencilgen.py`. Each coefficient is a product of binomial weights `[1, 9, 45, 70, 45, 9, 1]` normalized by 5832000.

### Stencil Usage

Stencils are **functions** that operate on fields:

```c
write(IMAGE0, blur(IMAGE0))                // Apply stencil to field
write(lnrho, derx(uy))                     // Derivative of one field, written to another
grad = real3(derx(s), dery(s), derz(s))    // Gradient as function call
```

---

## Functions

### Pure Functions (return a value, not a kernel)

```c
// Scalar function with implicit return type
fn(a, b) {
    return 1 + 2 * (1 - 1)
}

// Vector function
vecvalue(Field3 v) {
    return real3(value(v.x), value(v.y), value(v.z))
}

// Gradient computation
gradient(s) {
    return real3(derx(s), dery(s), derz(s))
}

// Divergence
divergence(v) {
    return derx(v.x) + dery(v.y) + derz(v.z)
}

// Curl
curl(v) {
    return real3(dery(v.z) - derz(v.y),
                 derz(v.x) - derx(v.z),
                 derx(v.y) - dery(v.x))
}

// Laplacian
laplace(s) {
    return derxx(s) + deryy(s) + derzz(s)
}

// Traceless rate of strain tensor
traceless_rateof_strain(v) {
    Matrix S
    S.data[0][0] = (2.0 / 3.0) * derx(v.x) - (1.0 / 3.0) * (dery(v.y) + derz(v.z))
    S.data[0][1] = (1.0 / 2.0) * (dery(v.x) + derx(v.y))
    S.data[0][2] = (1.0 / 2.0) * (derz(v.x) + derx(v.z))
    S.data[1][0] = S.data[0][1]
    S.data[1][1] = (2.0 / 3.0) * dery(v.y) - (1.0 / 3.0) * (derx(v.x) + derz(v.z))
    S.data[1][2] = (1.0 / 2.0) * (derz(v.y) + dery(v.z))
    S.data[2][0] = S.data[0][2]
    S.data[2][1] = S.data[1][2]
    S.data[2][2] = (2.0 / 3.0) * derz(v.z) - (1.0 / 3.0) * (derx(v.x) + dery(v.y))
    return S
}

// RK3 time integration
rk3(s0, s1, roc) {
    real alpha = 0., -5./9., -153. / 128.
    real beta  = 1. / 3., 15./ 16., 8. / 15.

    if step > 0 {
        return s1 + beta[step] * ((alpha[step] / beta[step - 1]) * (s1 - s0) + roc * dt)
    } else {
        return s1 + beta[step] * roc * dt
    }
}

// Momentum RHS (complex physics expression)
momentum() {
    S = traceless_rateof_strain(UU)
    cs2 = cs2_sound * exp(AC_gamma * value(VTXBUF_ENTROPY) / AC_cp_sound + (AC_gamma - 1.) * (value(VTXBUF_LNRHO) - AC_lnrho0))
    j = (1. / AC_mu0) * (gradient_of_divergence(AA) - veclaplace(AA))
    B = curl(AA)
    inv_rho = 1. / exp(value(VTXBUF_LNRHO))

    mom = -gradients(UU) * vecvalue(UU)
          - cs2 * ((1. / AC_cp_sound) * gradient(VTXBUF_ENTROPY) + gradient(VTXBUF_LNRHO))
          + inv_rho * cross(j, B)
          + AC_nu_visc * (veclaplace(UU) + (1. / 3.) * gradient_of_divergence(UU) + 2. * S * gradient(VTXBUF_LNRHO))
          + AC_zeta * gradient_of_divergence(UU)
    return mom
}

// Induction equation
induction() {
    return cross(UU, curl(AA)) + AC_eta * laplace(AA)
}

// Heat conduction
heat_conduction() {
    chi = AC_K_heatcond / (exp(VTXBUF_LNRHO) * AC_cp_sound)
    first_term = AC_gamma / AC_cp_sound * laplace(VTXBUF_ENTROPY) + (AC_gamma - 1.) * laplace(VTXBUF_LNRHO)
    second_term = AC_gamma / AC_cp_sound * gradient(VTXBUF_ENTROPY) + (AC_gamma - 1.) * gradient(VTXBUF_LNRHO)
    third_term = AC_gamma * (1. / AC_cp_sound * gradient(VTXBUF_ENTROPY) + gradient(VTXBUF_LNRHO)) + grad_ln_chi
    return AC_cp_sound * chi * (first_term + dot(second_term, third_term))
}

// Entropy RHS
entropy() {
    RHS = AC_eta * AC_mu0 * dot(j, j) + 2. * exp(VTXBUF_LNRHO) * AC_nu_visc * contract(S)
          + AC_zeta * exp(VTXBUF_LNRHO) * divergence(UU) * divergence(UU)
    return -dot(UU, gradient(VTXBUF_ENTROPY)) + inv_pT * RHS + heat_conduction()
}

// Helical forcing function
helical_forcing(k_force, xx, ff_re, ff_im, phi) {
    const real3 yy = 2.0 * R_PI * (xx / (AC_ds * AC_ngrid))
    cos_phi     = cos(phi)
    sin_phi     = sin(phi)
    cos_k_dot_x = cos(dot(k_force, yy))
    sin_k_dot_x = sin(dot(k_force, yy))
    real_comp_phase = cos_k_dot_x * cos_phi - sin_k_dot_x * sin_phi
    imag_comp_phase = cos_k_dot_x * sin_phi + sin_k_dot_x * cos_phi
    force = real3(ff_re.x * real_comp_phase - ff_im.x * imag_comp_phase,
                  ff_re.y * real_comp_phase - ff_im.y * imag_comp_phase,
                  ff_re.z * real_comp_phase - ff_im.z * imag_comp_phase)
    return force
}
```

### Function Overloading

Functions can be overloaded by signature:

```c
inline get_val()  { return 0.0 }
inline get_val(x) { return 0.0 }
inline get_val(x, y) { return 0.0 }

fn(int x)  { return x }
fn(int x, int y)  { return x }
fn(int x, int y, int z)  { return x }
fn() { return 1 }
```

### Utility Functions (not kernels)

```c
bc_sym_z(sgn, topbot, VtxBuffer j, rel) {
    if (topbot == AC_bot) {
        if (rel) {
            for i in 1:NGHOST+1 {
                j[vertexIdx.x][vertexIdx.y][AC_n1-i-1] = 2*j[vertexIdx.x][vertexIdx.y][AC_n1-1] + sgn*j[vertexIdx.x][vertexIdx.y][AC_n1+i-1]
            }
        } else {
            for i in 1:NGHOST+1 {
                j[vertexIdx.x][vertexIdx.y][AC_n1-i-1] = sgn*j[vertexIdx.x][vertexIdx.y][AC_n1+i-1]
            }
        }
    }
    // ... more branches
}
```

---

## Kernels

### Kernel Structure

```c
Kernel kernel_name([parameters]) {
    // GPU-executed code
    // Body can include: field access, stencil operations, math, control flow
    write(TARGET_FIELD, expression)
}
```

### Basic Kernels

```c
// Simple blur kernel
Kernel blur_kernel() {
    write(IMAGE0, blur(IMAGE0))
}

// Solve kernel (full MHD system)
Kernel solve() {
    write(lnrho, rk3(previous(lnrho), value(lnrho), continuity()))
    write(ss, rk3(previous(ss), current(ss), entropy()))

    mom = momentum()
    ind = induction()

    write(ux, rk3(previous(ux), current(ux), mom.x))
    write(uy, rk3(previous(uy), current(uy), mom.y))
    write(uz, rk3(previous(uz), current(uz), mom.z))

    write(ax, rk3(previous(ax), current(ax), ind.x))
    write(ay, rk3(previous(ay), current(ay), ind.y))
    write(az, rk3(previous(az), current(az), ind.z))
}
```

### Multi-pass Kernels

```c
// Intermediate step: computes time derivative
Kernel twopass_solve_intermediate(AC_SUBSTEP_NUMBER step_num, real dt) {
    write(VTXBUF_LNRHO, rk3_intermediate(VTXBUF_LNRHO, continuity(), int(step_num), dt))
    #if LENTROPY
    write(VTXBUF_ENTROPY, rk3_intermediate(VTXBUF_ENTROPY, entropy(), int(step_num), dt))
    #endif
    write(UU, rk3_intermediate(UU, momentum(), int(step_num), dt))
    #if LMAGNETIC
    write(AA, rk3_intermediate(AA, induction(), int(step_num), dt))
    #endif
    #if LBFIELD
    if step_num == 2 {
        write(BFIELD, curl(AA))
    }
    #endif
}

// Final step: applies boundary conditions and final update
fixed_boundary Kernel twopass_solve_final(int step_num, real current_time, real dt) {
    write(VTXBUF_LNRHO, rk3_final(VTXBUF_LNRHO, step_num))
    write(VTXBUF_ENTROPY, rk3_final(VTXBUF_ENTROPY, step_num))
    write(UU, rk3_final(UU, step_num) + forcing_step)
    #if LMAGNETIC
    write(AA, rk3_final(AA, step_num))
    #endif
}
```

### Utility Kernels

```c
// Scale all fields
utility Kernel scale() {
    for field in 0:NUM_FIELDS {
        write(Field(field), Field(field))
    }
    #if LMAGNETIC
    write(AA, AC_scaling_factor * AA)
    #endif
}

// Reset all fields to zero
utility Kernel reset() {
    for field in 0:NUM_FIELDS {
        write(Field(field), 0.0)
    }
}
```

### Kernels with Parameters

```c
Kernel singlepass_solve(int step_num, TimeParams time_params) {
    write(VTXBUF_LNRHO, rk3(VTXBUF_LNRHO, continuity(), step_num, time_params.dt))
    #if LENTROPY
    write(VTXBUF_ENTROPY, rk3(VTXBUF_ENTROPY, entropy(), step_num, time_params.dt))
    #endif
    if (AC_lforcing) {
        if step_num == 2 {
            if time_params.current_time > AC_switch_forcing {
                forcing_step = forcing(time_params.dt)
            }
        }
        write(UU, rk3(UU, momentum(), step_num, time_params.dt) + forcing_step)
    }
    #if LMAGNETIC
    write(AA, rk3(AA, induction(), step_num, time_params.dt))
    #endif
}
```

### Extended Grid Kernels

```c
dims(AC_extended_mlocal) Field OMEGA
dims(AC_extended_mlocal) Field EXTENDED_LN_DENSITY
dims(AC_extended_mlocal) Field EXTENDED_GRAVITY_POTENTIAL

fixed_boundary Kernel sor_red(real omega) {
    poisson_sor_red_black_extended(SOR_RED, 4*AC_REAL_PI*G_NEWTON*exp(EXTENDED_LN_DENSITY),
                                    EXTENDED_GRAVITY_POTENTIAL, value(OMEGA))
}
```

### Global Output Kernels

```c
global output real AC_residual2
global output real AC_rhs_norm2

Kernel get_residual_kernel() {
    residual = 4*AC_REAL_PI*G_NEWTON*exp(EXTENDED_LN_DENSITY) - laplace_extended(EXTENDED_GRAVITY_POTENTIAL)
    reduce_sum(residual*residual, AC_residual2)
}
```

---

## Field Access Functions

These are the core mechanisms for reading and writing field data from within kernels:

| Function | Description |
|----------|-------------|
| `write(field, value)` | Write `value` to `field` at the current vertex |
| `value(field)` | Read the **current** (in-place) value of `field` |
| `current(field)` | Alias for `value(field)` — read current value |
| `previous(field)` | Read the **previous time-step** value of `field` (double-buffered) |

```c
// RK3 time integration with field access
write(lnrho, rk3(previous(lnrho), value(lnrho), continuity()))

// In-place read
cs2 = cs2_sound * exp(AC_gamma * current(ss) / cp_sound + (AC_gamma - 1.) * (current(lnrho) - lnrho0))
inv_rho = 1. / exp(current(lnrho))
```

---

## Reduction Operations

Global or subdomain-wide reduction operations:

```c
reduce_min(visc_dt, advec_dt, AC_dt_min)
reduce_max(norm(UU), UU_MAX_ADVEC)
reduce_max(alfven_speed, ALFVEN_SPEED_MAX)
reduce_max(VTXBUF_SHOCK, AC_MAX_SHOCK)
reduce_sum(t*s, BICGSTAB_tTs)
reduce_sum(t*t, BICGSTAB_tTt)
```

---

## ComputeSteps (Kernel Sequences)

`ComputeSteps` groups kernels into a **scheduled sequence** with automatic halo exchange and boundary condition application:

```c
BoundConds bcs {
    periodic(BOUNDARY_XYZ)
}

// Single time step: 3 substeps
ComputeSteps AC_rhs(boundconds) {
    twopass_solve_intermediate(AC_FIRST_SUBSTEP, AC_dt)
    twopass_solve_final(AC_FIRST_SUBSTEP, AC_current_time, AC_dt)

    twopass_solve_intermediate(AC_SECOND_SUBSTEP, AC_dt)
    twopass_solve_final(AC_SECOND_SUBSTEP, AC_current_time, AC_dt)

    twopass_solve_intermediate(AC_THIRD_SUBSTEP, AC_dt)
    twopass_solve_final(AC_THIRD_SUBSTEP, AC_current_time, AC_dt)
}

// Single substep
ComputeSteps AC_rhs_substep(boundconds) {
    twopass_solve_intermediate(AC_SUBSTEP, AC_dt)
    twopass_solve_final(AC_SUBSTEP, AC_current_time, AC_dt)
}

// Initialization
ComputeSteps AC_initialize(bcs) {
    initial_condition()
}

// Solve
ComputeSteps AC_solve(bcs) {
    singlepass_solve()
}

// Timestep calculation
ComputeSteps AC_calc_timestep(bcs) {
    calc_timestep_kernel()
}
```

---

## Boundary Conditions

```c
BoundConds boundconds {
    periodic(BOUNDARY_XYZ)

    // Or conditional BCs:
    #if LSELFGRAVITY
        copy_extended_to_grid(BOUNDARY_X_BOT, GRAVITY_POTENTIAL, EXTENDED_GRAVITY_POTENTIAL)
        multipole_expansion_bc_inner_extended(BOUNDARY_X_BOT, EXTENDED_GRAVITY_POTENTIAL, G_NEWTON)
        multipole_expansion_bc_outer_extended(BOUNDARY_X_TOP, EXTENDED_GRAVITY_POTENTIAL, G_NEWTON)
    #endif

    ac_bc_a2(BOUNDARY_X, UU)
    ac_bc_sym(BOUNDARY_Y)
    periodic(BOUNDARY_Z)

    ac_const_bc(BOUNDARY_X, EXTENDED_BICGSTAB_P, 0.0)
}

// Symmetry BC function
bc_sym_z(sgn, topbot, VtxBuffer j, rel) {
    if (topbot == AC_bot) {
        for i in 1:NGHOST+1 {
            j[vertexIdx.x][vertexIdx.y][AC_n1-i-1] = sgn * j[vertexIdx.x][vertexIdx.y][AC_n1+i-1]
        }
    } else if (topbot == AC_top) {
        for i in 1:NGHOST+1 {
            j[vertexIdx.x][vertexIdx.y][AC_n2+i-1] = sgn * j[vertexIdx.x][vertexIdx.y][AC_n2-i-1]
        }
    }
}
```

---

## Control Flow

### If Statements

```c
if step > 0 {
    return s1 + beta[step] * ((alpha[step] / beta[step - 1]) * (s1 - s0) + roc * dt)
} else {
    return s1 + beta[step] * roc * dt
}

if step_num == 2 {
    if time_params.current_time > AC_switch_forcing {
        forcing_step = forcing(time_params.dt)
    }
}

if (rr > 2.0 * AC_ds.x) {
    // Gaussian profile computation
}

if 0 == 0 && 1 == 1 {
    return 0
}

// Comparison operators: ==, !=, >, <, >=, <=
if divu < 0.0 {
    return -divu
} else {
    return 0.0
}

// Boolean expressions
if LSPHERICAL_SINK_PARTICLE {
    // Preprocessor conditionals for code inclusion
}
```

### For Loops

```c
// C-style loops
for (int i = 0; i < N; i++) {
    // ...
}

// Range-based loops
for j in 0:n {
    c = j
}

for i in 1:NGHOST+1 {
    j[vertexIdx.x][vertexIdx.y][AC_n1-i-1] = sgn * j[vertexIdx.x][vertexIdx.y][AC_n1+i-1]
}

// Field iteration
for field in 0:NUM_FIELDS {
    write(Field(field), Field(field))
}

// Nested loops
for step in 0:10 {
    i = xorshift(i)
}
for step in 0:(i % 40) {
    i = xorshift(i)
}
```

---

## Built-in Variables and Functions

### Grid/Vertex Variables (provided by runtime)

| Variable | Type | Description |
|----------|------|-------------|
| `vertexIdx` | `int3` | Local vertex index within subdomain |
| `globalVertexIdx` | `int3` | Global vertex index across all ranks |
| `AC_ds` | `real3` | Grid spacing (dx, dy, dz) |
| `AC_dsmin` | `real` | Minimum grid spacing |
| `AC_dsmin_2` | `real` | Minimum grid spacing squared |
| `AC_inv_ds` | `real3` | Inverse grid spacing |
| `AC_ngrid` | `int3` | Total grid dimensions |
| `AC_nxgrid, AC_nygrid, AC_nzgrid` | `int` | Total grid dimensions |
| `AC_n1, AC_n2` | `int` | Min/max indices in a dimension |
| `AC_mx, AC_my, AC_mz` | `int` | Max indices |
| `AC_nmin` | `int3` | Local min indices |
| `AC_nmax` | `int3` | Local max indices |
| `AC_nx, AC_ny, AC_nz` | `int` | Local grid dimensions |
| `AC_REAL_PI` | `real` | PI constant |
| `AC_REAL_MAX` | `real` | Maximum real value |
| `NGHOST` | `int` | Number of ghost/halo cells |

### Coordinate Functions

```c
real3 xx = grid_position()        // Physical position of current vertex
real3 center = grid_center()      // Grid center
real3 xx = grid_position() - grid_center()  // Offset from center
```

### Mathematical Functions

```c
sqrt(x)       // Square root
pow(x, n)     // Power
exp(x)        // Exponential
log(x)        // Natural logarithm
sin(x), cos(x) // Trig functions
atan2(y, x)   // Two-argument arctangent
abs(x)        // Absolute value
real(x)       // Cast to real
int(x)        // Cast to int
uint64_t(x)   // Cast to uint64_t
```

### Vector Operations

```c
real3 a = {ax, ay, az}          // Vector construction
real3 b = real3(x, y, z)        // Vector construction via constructor
c = a.x, a.y, a.z               // Component access
a.x = 1.0                       // Component assignment
c = dot(a, b)                   // Dot product
c = cross(a, b)                 // Cross product
c = norm(v)                     // Vector magnitude

// Matrix access
Matrix S
S.data[0][0] = 1.0
S.data[1][1] = 1 * S[1][1]
S[0][0] = 1
real3 row0 = S.row(0)
real c = contract(S)  // Contraction (sum of squared components)
```

### Random Number Generation

```c
// From stdlib/utils
rand_uniform()      // Uniform random in [0, 1)
rand_normal()       // Normal random
xorshift(state)     // xorshift RNG (inline function)
```

### Print/Output

```c
print("test: %d\n", value)
print("Residual at (%d,%d,%d): %.14e,%.14e\n", vertexIdx.x, vertexIdx.y, vertexIdx.z, density, laplacian)
print("Length of an array: %lu\n", len(arr1))
```

### Array Operations

```c
real arr1 = 1.0, 2.0, 3.0
print("Length: %lu\n", len(arr1))
arr1[0] = 1
hello = [1, 2, 3]
hello[0] = 1
```

### CUDA Intrinsics (embedded)

```c
__shared__ real tpb_radius_sum
if (threadIdx.x == 0 && threadIdx.y == 0) {
    tpb_radius_sum = 0.0
}
__syncthreads()
atomicAdd(&tpb_radius_sum, radius_sum)
```

---

## Includes and Modular Structure

```c
#include "../../stdlib/integrators.h"
#include "../../stdlib/math"          // Without .h — directory include
#include "../../stdlib/grid"          // Without .h — directory include
#include "../../stdlib/derivs.h"
#include "../../stdlib/operators.h"
#include "../../stdlib/units.h"
#include "../../stdlib/map.h"
#include "../../stdlib/utils/kernels.h"
#include "../../stdlib/bc.h"
#include "../../standalone_params.h"

#if LSHOCK
#include "smooth_kernel.ach"          // .ach = stencil header
#endif

#if LSELFGRAVITY
#include "../../stdlib/bicgstab.h"
#include "../../stdlib/poisson.h"
#include "../../stdlib/spherical_harmonics.h"
#include "../../stdlib/grid_extension.h"
#endif
```

---

## Preprocessor/Directives

```c
// C preprocessor (runs before the DSL compiler)
#include "file.h"
#define MACRO (value)
hostdefine MACRO (value)       // Define passed to host compilation

#if CONDITION
// Conditional compilation
#include "file.h"
#endif

#if CONDITION
#error "Something"
#endif

// Comments
// Single line
/* Block comment */
```

---

## Complete Example: MHD Solver

```c
// 1. Constants and parameters
input int step_number
input real AC_dt
const real INV_DS = 1. / 0.04908738521

real AC_cs_sound, AC_cp_sound
real AC_gamma, AC_nu_visc, AC_zeta, AC_eta
real AC_lnrho0, AC_mu0, AC_lnT0

// 2. Fields
Field VTXBUF_LNRHO, VTXBUF_UUX, VTXBUF_UUY, VTXBUF_UUZ, VTXBUF_AX, VTXBUF_AY, VTXBUF_AZ, VTXBUF_ENTROPY

#define UU Field3(VTXBUF_UUX, VTXBUF_UUY, VTXBUF_UUZ)
#define AA Field3(VTXBUF_AX, VTXBUF_AY, VTXBUF_AZ)

// 3. Stencils (derivative operators)
Stencil derx {
    [-3][0][0] = -DER1_3,
    [-2][0][0] = -DER1_2,
    [-1][0][0] = -DER1_1,
    [1][0][0]  =  DER1_1,
    [2][0][0]  =  DER1_2,
    [3][0][0]  =  DER1_3
}
// ... more stencils for dery, derz, derxx, deryy, derzz, derxy, derxz, deryz ...

// 4. Physics functions
continuity() {
    return -dot(vecvalue(UU), gradient(VTXBUF_LNRHO)) - divergence(UU)
}

momentum() {
    S = traceless_rateof_strain(UU)
    cs2 = cs2_sound * exp(AC_gamma * value(VTXBUF_ENTROPY) / AC_cp_sound + (AC_gamma - 1.) * (value(VTXBUF_LNRHO) - AC_lnrho0))
    j = (1. / AC_mu0) * (gradient_of_divergence(AA) - veclaplace(AA))
    B = curl(AA)
    inv_rho = 1. / exp(value(VTXBUF_LNRHO))
    mom = -gradients(UU) * vecvalue(UU)
          - cs2 * ((1. / AC_cp_sound) * gradient(VTXBUF_ENTROPY) + gradient(VTXBUF_LNRHO))
          + inv_rho * cross(j, B)
          + AC_nu_visc * (veclaplace(UU) + (1. / 3.) * gradient_of_divergence(UU) + 2. * S * gradient(VTXBUF_LNRHO))
          + AC_zeta * gradient_of_divergence(UU)
    return mom
}

induction() {
    return cross(vecvalue(UU), curl(AA)) + AC_eta * veclaplace(AA)
}

// 5. Time integration
rk3(s0, s1, roc) {
    real alpha = 0., -5./9., -153. / 128.
    real beta  = 1. / 3., 15./ 16., 8. / 15.
    if step_number > 0 {
        return s1 + beta[step_number] * ((alpha[step_number] / beta[step_number - 1]) * (s1 - s0) + roc * AC_dt)
    } else {
        return s1 + beta[step_number] * roc * AC_dt
    }
}

// 6. Kernel
Kernel solve() {
    write(VTXBUF_LNRHO, rk3(previous(VTXBUF_LNRHO), value(VTXBUF_LNRHO), continuity()))
    write(VTXBUF_ENTROPY, rk3(previous(VTXBUF_ENTROPY), value(VTXBUF_ENTROPY), entropy()))
    mom = momentum()
    ind = induction()
    write(VTXBUF_UUX, rk3(previous(VTXBUF_UUX), value(VTXBUF_UUX), mom.x))
    write(VTXBUF_UUY, rk3(previous(VTXBUF_UUY), value(VTXBUF_UUY), mom.y))
    write(VTXBUF_UUZ, rk3(previous(VTXBUF_UUZ), value(VTXBUF_UUZ), mom.z))
    write(VTXBUF_AX, rk3(previous(VTXBUF_AX), value(VTXBUF_AX), ind.x))
    write(VTXBUF_AY, rk3(previous(VTXBUF_AY), value(VTXBUF_AY), ind.y))
    write(VTXBUF_AZ, rk3(previous(VTXBUF_AZ), value(VTXBUF_AZ), ind.z))
}

// 7. Boundary conditions and execution graph
BoundConds boundconds {
    periodic(BOUNDARY_XYZ)
}

ComputeSteps AC_rhs(boundconds) {
    twopass_solve_intermediate(AC_FIRST_SUBSTEP, AC_dt)
    twopass_solve_final(AC_FIRST_SUBSTEP, AC_current_time, AC_dt)
    twopass_solve_intermediate(AC_SECOND_SUBSTEP, AC_dt)
    twopass_solve_final(AC_SECOND_SUBSTEP, AC_current_time, AC_dt)
    twopass_solve_intermediate(AC_THIRD_SUBSTEP, AC_dt)
    twopass_solve_final(AC_THIRD_SUBSTEP, AC_current_time, AC_dt)
}
```

---

## Complete Example: Image Blur

```c
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

This demonstrates the entire DSL in 7 lines of kernel logic — just a field, a 7-point stencil, and a kernel that writes the convolution result.

---

## Compilation Flow

```
.ac file (Astaroth DSL source)
    |
    v
Preprocessing (C preprocessor: #include, #define, #if)
    |
    v
acc/ compiler (Flex lexer + Bison parser)
    |
    |-- ac.l: Tokenizer
    |-- ac.y: Grammar parser → AST
    |-- ast.h: AST node definitions
    |-- codegen.c: Code generation
    |-- implementation.c: Kernel implementation
    |-- stencilgen.c: Stencil expansion
    |
    v
Generated output files:
    |-- user_kernels.ac.pp_stage*: Preprocessed DSL
    |-- user_kernels.inc.raw: Unformatted CUDA/HIP code
    |-- user_kernels_backup.inc: Formatted generated code
    |-- user_defines.inc: Project defines
    |-- user_kernels.inc: Final compiled kernels
    |
    v
CUDA/HIP compiler (nvcc / hipcc)
    |
    v
libkernels.so (shared library with GPU kernels)
    |
    v
Runtime links against: acc-runtime/api/
```

---

## Key Design Principles

1. **Stencil-as-first-class**: Stencils are named, parameterized weighted sums over neighbor offsets, serving as the primary communication pattern abstraction.

2. **Double-buffered fields**: `previous(field)` vs `value(field)` provides automatic time-step history for explicit time integrators (like RK3).

3. **ComputeSteps orchestration**: Rather than manual kernel launching, the DSL declaratively specifies sequences of kernels with boundary conditions, and the runtime handles dependency ordering, halo exchange, and scheduling.

4. **Conditional compilation**: `hostdefine` and `#if` enable modular simulation configurations (e.g., MHD vs hydro, with/without self-gravity, with/without shock capture).

5. **Mixed precision support**: Numeric suffixes (`.f`, `.d`) and CMake flags control precision throughout the codebase.

6. **Mathematical expressions as functions**: Physics RHS (continuity, momentum, induction, entropy) are expressed as pure functions that return values, not kernels. The kernel layer simply wraps these in `write()` calls with time integrators.

7. **Modular includes**: The DSL heavily uses `#include` to compose physics modules from the `stdlib/` — users mix and match operators, derivatives, and solvers.

8. **GPU pragmas embedded**: CUDA intrinsics (`__shared__`, `__syncthreads()`, `atomicAdd`, `threadIdx`) can be embedded directly in kernel bodies for custom synchronization and reduction patterns.

9. **Function overloading**: Resolved by signature (number and type of parameters), enabling generic-looking APIs like `get_val()` vs `get_val(x)` vs `get_val(x, y)`.

10. **Profile types**: `Profile<Z>` fields vary only along one dimension, enabling efficient 1D operations (like z-averaging) on 3D grids.
