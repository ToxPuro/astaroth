/*
    Copyright (C) 2014-2021, Johannes Pekkila, Miikka Vaisala.

    This file is part of Astaroth.

    Astaroth is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Astaroth is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Astaroth.  If not, see <http://www.gnu.org/licenses/>.
*/

/**
 * @file
 * \brief Brief info.
 *
 * Detailed info.
 *
 */
#include "astaroth_utils.h"

#include <math.h>
#include <stdbool.h>

#include "errchk.h"
#include "memory.h" // acHostMeshCreate, acHostMeshDestroy, acHostMeshApplyPeriodicBounds

#ifdef AC_INTEGRATION_ENABLED
/*
// Standalone flags (currently defined in the DSL)
#define LDENSITY (1)
#define R_PI ((Scalar)M_PI)
*/

/*
typedef AcReal Scalar;
// typedef AcReal3 Vector;
// typedef AcMatrix Matrix;

#if AC_DOUBLE_PRECISION == 1
typedef double Vector __attribute__((vector_size(4 * sizeof(double))));
#else
typedef float Vector __attribute__((vector_size(4 * sizeof(float))));

#define fabs fabsf
#define exp expf
#define sqrt sqrtf
#define cos cosf
#define sin sinf
#endif
*/
typedef long double Scalar;
typedef struct {
    Scalar x, y, z;
} Vector;

typedef struct {
    Vector row[3];
} Matrix;

#define fabs fabsl
#define exp expl
#define sqrt sqrtl
#define cos cosl
#define sin sinl

#define SCALAR_PI (M_PIl) // Long double variant

static Vector
operator-(const Vector& a)
{
    return (Vector){-a.x, -a.y, -a.z};
}

static Vector
operator+(const Vector& a, const Vector& b)
{
    return (Vector){a.x + b.x, a.y + b.y, a.z + b.z};
}

static Vector
operator-(const Vector& a, const Vector& b)
{
    return (Vector){a.x - b.x, a.y - b.y, a.z - b.z};
}

static Vector
operator*(const Scalar& a, const Vector& b)
{
    return (Vector){a * b.x, a * b.y, a * b.z};
}

static AcMeshInfo* mesh_info = NULL;

static inline int
getInt(const AcIntParam param)
{
    return mesh_info->int_params[param];
}

static inline Scalar
getReal(const AcRealParam param)
{
    return (Scalar)mesh_info->real_params[param];
}

static inline int
IDX(const int i, const int j, const int k)
{
    return acVertexBufferIdx(i, j, k, (*mesh_info));
}

typedef struct {
    Scalar value;
    Vector gradient;
    Matrix hessian;
#if LUPWD
    Vector upwind;
#endif
} ScalarData;

typedef struct {
    ScalarData xdata;
    ScalarData ydata;
    ScalarData zdata;
} VectorData;

static inline Scalar
first_derivative(const Scalar* pencil, const Scalar inv_ds)
{
#if STENCIL_ORDER == 2
    const Scalar coefficients[] = {0, (Scalar)(1. / 2.)};
#elif STENCIL_ORDER == 4
    const Scalar coefficients[] = {0, (Scalar)(2.0 / 3.0), (Scalar)(-1.0 / 12.0)};
#elif STENCIL_ORDER == 6
    const Scalar coefficients[] = {
        0,
        (Scalar)3.0 / (Scalar)4.0,
        (Scalar)-3.0 / (Scalar)20.0,
        (Scalar)1.0 / (Scalar)60.0,
    };
#elif STENCIL_ORDER == 8
    const Scalar coefficients[] = {
        0, (Scalar)(4.0 / 5.0), (Scalar)(-1.0 / 5.0), (Scalar)(4.0 / 105.0), (Scalar)(-1.0 / 280.0),
    };
#endif

#define MID (STENCIL_ORDER / 2)
    Scalar res = 0;

    // #pragma unroll
    for (int i = 1; i <= MID; ++i)
        // for (int i = MID; i >= 1; --i)
        res += coefficients[i] * (pencil[MID + i] - pencil[MID - i]);

    return res * inv_ds;
}

static inline Scalar
second_derivative(const Scalar* pencil, const Scalar inv_ds)
{
#if STENCIL_ORDER == 2
    const Scalar coefficients[] = {-2, 1};
#elif STENCIL_ORDER == 4
    const Scalar coefficients[] = {
        (Scalar)(-5.0 / 2.0),
        (Scalar)(4.0 / 3.0),
        (Scalar)(-1.0 / 12.0),
    };
#elif STENCIL_ORDER == 6
    const Scalar coefficients[] = {
        (Scalar)-49.0 / (Scalar)18.0,
        (Scalar)3.0 / (Scalar)2.0,
        (Scalar)-3.0 / (Scalar)20.0,
        (Scalar)1.0 / (Scalar)90.0,
    };
#elif STENCIL_ORDER == 8
    const Scalar coefficients[] = {
        (Scalar)(-205.0 / 72.0), (Scalar)(8.0 / 5.0),    (Scalar)(-1.0 / 5.0),
        (Scalar)(8.0 / 315.0),   (Scalar)(-1.0 / 560.0),
    };
#endif

#define MID (STENCIL_ORDER / 2)
    Scalar res = coefficients[0] * pencil[MID];

    // #pragma unroll
    for (int i = 1; i <= MID; ++i)
        // for (int i = MID; i >= 1; --i)
        res += coefficients[i] * (pencil[MID + i] + pencil[MID - i]);

    return res * inv_ds * inv_ds;
}

/** inv_ds: inverted mesh spacing f.ex. 1. / mesh.int_params[AC_dsx] */
static inline Scalar
cross_derivative(const Scalar* pencil_a, const Scalar* pencil_b, const Scalar inv_ds_a,
                 const Scalar inv_ds_b)
{
#if STENCIL_ORDER == 2
    const Scalar coefficients[] = {0, (Scalar)(1.0 / 4.0)};
#elif STENCIL_ORDER == 4
    const Scalar coefficients[] = {
        (Scalar)0.,
        0,
        0,
    }; // TODO correct coefficients, these are just placeholders
#elif STENCIL_ORDER == 6
    const Scalar fac            = (Scalar)1. / (Scalar)720.;
    const Scalar coefficients[] = {
        0 * fac,
        (Scalar)(270.0) * fac,
        (Scalar)(-27.0) * fac,
        (Scalar)(2.0) * fac,
    };
#elif STENCIL_ORDER == 8
    const Scalar fac            = ((Scalar)(1. / 20160.));
    const Scalar coefficients[] = {
        0 * fac,
        (Scalar)(8064.) * fac,
        (Scalar)(-1008.) * fac,
        (Scalar)(128.) * fac,
        (Scalar)(-9.) * fac,
    };
#endif

#define MID (STENCIL_ORDER / 2)
    Scalar res = (Scalar)(0.);

    // #pragma unroll
    for (int i = 1; i <= MID; ++i) {
        // for (int i = MID; i >= 1; --i) {
        res += coefficients[i] *
               (pencil_a[MID + i] + pencil_a[MID - i] - pencil_b[MID + i] - pencil_b[MID - i]);
    }
    return res * inv_ds_a * inv_ds_b;
}

static inline Scalar
derx(const int i, const int j, const int k, const AcReal* arr)
{
    Scalar pencil[STENCIL_ORDER + 1];
    // #pragma unroll
    for (int offset = 0; offset < STENCIL_ORDER + 1; ++offset)
        pencil[offset] = (Scalar)arr[IDX(i + offset - STENCIL_ORDER / 2, j, k)];

    return first_derivative(pencil, ((Scalar)1. / getReal(AC_dsx)));
}

static inline Scalar
derxx(const int i, const int j, const int k, const AcReal* arr)
{
    Scalar pencil[STENCIL_ORDER + 1];
    // #pragma unroll
    for (int offset = 0; offset < STENCIL_ORDER + 1; ++offset)
        pencil[offset] = (Scalar)arr[IDX(i + offset - STENCIL_ORDER / 2, j, k)];

    return second_derivative(pencil, ((Scalar)1. / getReal(AC_dsx)));
}

static inline Scalar
derxy(const int i, const int j, const int k, const AcReal* arr)
{
    Scalar pencil_a[STENCIL_ORDER + 1];
    // #pragma unroll
    for (int offset = 0; offset < STENCIL_ORDER + 1; ++offset)
        pencil_a[offset] = (Scalar)arr[IDX(i + offset - STENCIL_ORDER / 2, //
                                           j + offset - STENCIL_ORDER / 2, k)];

    Scalar pencil_b[STENCIL_ORDER + 1];
    // #pragma unroll
    for (int offset = 0; offset < STENCIL_ORDER + 1; ++offset)
        pencil_b[offset] = (Scalar)arr[IDX(i + offset - STENCIL_ORDER / 2, //
                                           j + STENCIL_ORDER / 2 - offset, k)];

    return cross_derivative(pencil_a, pencil_b, ((Scalar)1. / getReal(AC_dsx)),
                            ((Scalar)1. / getReal(AC_dsy)));
}


static inline Scalar
dery(const int i, const int j, const int k, const AcReal* arr)
{
    Scalar pencil[STENCIL_ORDER + 1];
    // #pragma unroll
    for (int offset = 0; offset < STENCIL_ORDER + 1; ++offset)
        pencil[offset] = (Scalar)arr[IDX(i, j + offset - STENCIL_ORDER / 2, k)];

    return first_derivative(pencil, ((Scalar)1. / getReal(AC_dsy)));
}

static inline Scalar
deryy(const int i, const int j, const int k, const AcReal* arr)
{
    Scalar pencil[STENCIL_ORDER + 1];
    // #pragma unroll
    for (int offset = 0; offset < STENCIL_ORDER + 1; ++offset)
        pencil[offset] = (Scalar)arr[IDX(i, j + offset - STENCIL_ORDER / 2, k)];

    return second_derivative(pencil, ((Scalar)1. / getReal(AC_dsy)));
}

/*
 * =============================================================================
 * Level 0.3 (Built-in functions available during the Stencil Processing Stage)
 * =============================================================================
 */
/*
static inline Vector
operator-(const Vector a, const Vector b)
{
    return (Vector){a.x - b.x, a.y - b.y, a.z - b.z};
}

static inline Vector
operator+(const Vector a, const Vector b)
{
    return (Vector){a.x + b.x, a.y + b.y, a.z + b.z};
}

static inline Vector
operator-(const Vector a)
{
    return (Vector){-a.x, -a.y, -a.z};
}

static inline Vector operator*(const Scalar a, const Vector b)
{
    return (Vector){a * b.x, a * b.y, a * b.z};
}
*/

static inline Scalar
dot(const Vector a, const Vector b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

static inline Vector
mul(const Matrix aa, const Vector x)
{
    return (Vector){dot(aa.row[0], x), dot(aa.row[1], x), dot(aa.row[2], x)};
}

static inline Vector
cross(const Vector a, const Vector b)
{
    Vector c;

    c.x = a.y * b.z - a.z * b.y;
    c.y = a.z * b.x - a.x * b.z;
    c.z = a.x * b.y - a.y * b.x;

    return c;
}
/*
static inline bool
is_valid(const Scalar a)
{
    return !isnan(a) && !isinf(a);
}

static inline bool
is_valid(const Vector a)
{
    return is_valid(a.x) && is_valid(a.y) && is_valid(a.z);
}
*/

/*
 * =============================================================================
 * Stencil Processing Stage (equations)
 * =============================================================================
 */



__attribute__((unused)) static inline Scalar
length(const Vector vec)
{
    return sqrt(vec.x * vec.x + vec.y * vec.y + vec.z * vec.z);
}

static inline Scalar
reciprocal_len(const Vector vec)
{
    return (Scalar)(1.0) / sqrt(vec.x * vec.x + vec.y * vec.y + vec.z * vec.z);
}

__attribute__((unused)) static inline Vector
normalized(const Vector vec)
{
    const Scalar inv_len = reciprocal_len(vec);
    return (Vector){inv_len * vec.x, inv_len * vec.y, inv_len * vec.z};
}

#define H_CONST ((Scalar)(0.0))
#define C_CONST ((Scalar)(0.0))


__attribute__((unused)) static inline bool
is_valid(const Scalar a)
{
    return !isnan(a) && !isinf(a);
}

__attribute__((unused)) static inline bool
is_valid_vec(const Vector a)
{
    return is_valid(a.x) && is_valid(a.y) && is_valid(a.z);
}

#if LFORCING
Vector
simple_vortex_forcing(Vector a, Vector b, Scalar magnitude)
{
    return magnitude * cross(normalized(b - a), (Vector){0, 0, 1}); // Vortex
}

Vector
simple_outward_flow_forcing(Vector a, Vector b, Scalar magnitude)
{
    return magnitude * (1 / length(b - a)) * normalized(b - a); // Outward flow
}

// The Pencil Code forcing_hel_noshear(), manual Eq. 222, inspired forcing function with adjustable
// helicity
Vector
helical_forcing(Scalar magnitude, Vector k_force, Vector xx, Vector ff_re, Vector ff_im, Scalar phi)
{
    (void)magnitude; // WARNING: unused
    xx.x = xx.x * ((Scalar)2.0 * SCALAR_PI / (getReal(AC_dsx) * getInt(AC_nx)));
    xx.y = xx.y * ((Scalar)2.0 * SCALAR_PI / (getReal(AC_dsy) * getInt(AC_ny)));
    xx.z = xx.z * ((Scalar)2.0 * SCALAR_PI / (getReal(AC_dsz) * getInt(AC_nz)));

    Scalar cos_phi     = cos(phi);
    Scalar sin_phi     = sin(phi);
    Scalar cos_k_dot_x = cos(dot(k_force, xx));
    Scalar sin_k_dot_x = sin(dot(k_force, xx));
    // Phase affect only the x-component
    // Scalar real_comp       = cos_k_dot_x;
    // Scalar imag_comp       = sin_k_dot_x;
    Scalar real_comp_phase = cos_k_dot_x * cos_phi - sin_k_dot_x * sin_phi;
    Scalar imag_comp_phase = cos_k_dot_x * sin_phi + sin_k_dot_x * cos_phi;

    Vector force = (Vector){ff_re.x * real_comp_phase - ff_im.x * imag_comp_phase,
                            ff_re.y * real_comp_phase - ff_im.y * imag_comp_phase,
                            ff_re.z * real_comp_phase - ff_im.z * imag_comp_phase};

    return force;
}

Vector
forcing(int3 globalVertexIdx, Scalar dt)
{
    Vector a = (Scalar)(.5) * (Vector){getInt(AC_nx) * getReal(AC_dsx),
                                       getInt(AC_ny) * getReal(AC_dsy),
                                       getInt(AC_nz) * getReal(AC_dsz)}; // source (origin)
    (void)a;                                                             // WARNING: not used
    Vector xx = (Vector){
        (globalVertexIdx.x - getInt(AC_nx_min)) * getReal(AC_dsx),
        (globalVertexIdx.y - getInt(AC_ny_min)) * getReal(AC_dsy),
        (globalVertexIdx.z - getInt(AC_nz_min)) * getReal(AC_dsz),
    }; // sink (current index)
    const Scalar cs2 = getReal(AC_cs2_sound);
    const Scalar cs  = sqrt(cs2);

    // Placeholders until determined properly
    Scalar magnitude = getReal(AC_forcing_magnitude);
    Scalar phase     = getReal(AC_forcing_phase);
    Vector k_force   = (Vector){getReal(AC_k_forcex), getReal(AC_k_forcey), getReal(AC_k_forcez)};
    Vector ff_re = (Vector){getReal(AC_ff_hel_rex), getReal(AC_ff_hel_rey), getReal(AC_ff_hel_rez)};
    Vector ff_im = (Vector){getReal(AC_ff_hel_imx), getReal(AC_ff_hel_imy), getReal(AC_ff_hel_imz)};

    (void)phase;   // WARNING: unused with simple forcing. Should be defined in helical_forcing
    (void)k_force; // WARNING: unused with simple forcing. Should be defined in helical_forcing
    (void)ff_re;   // WARNING: unused with simple forcing. Should be defined in helical_forcing
    (void)ff_im;   // WARNING: unused with simple forcing. Should be defined in helical_forcing

    // Determine that forcing funtion type at this point.
    // Vector force = simple_vortex_forcing(a, xx, magnitude);
    // Vector force = simple_outward_flow_forcing(a, xx, magnitude);
    Vector force = helical_forcing(magnitude, k_force, xx, ff_re, ff_im, phase);

    // Scaling N = magnitude*cs*sqrt(k*cs/dt)  * dt
    const Scalar NN = cs * sqrt(getReal(AC_kaver) * cs);
    // MV: Like in the Pencil Code. I don't understandf the logic here.
    force.x = sqrt(dt) * NN * force.x;
    force.y = sqrt(dt) * NN * force.y;
    force.z = sqrt(dt) * NN * force.z;

    if (is_valid_vec(force)) {
        return force;
    }
    else {
        return (Vector){0, 0, 0};
    }
}
#endif

static void
solve_alpha_step(AcMesh in, const int step_number, const AcReal dt, const int i, const int j,
                 const int k, AcMesh* out)
{
    const int idx = acVertexBufferIdx(i, j, k, in.info);


    Scalar rate_of_change[NUM_VTXBUF_HANDLES] = {0};
    rate_of_change[VTXBUF_LNRHO]              = derxx(i,j,k,in.vertex_buffer[VTXBUF_LNRHO]) + deryy(i,j,k,in.vertex_buffer[VTXBUF_LNRHO]);

    // Williamson (1980) NOTE: older version of astaroth used inhomogenous
    const Scalar alpha[] = {(Scalar)(.0), (Scalar)(-5. / 9.), (Scalar)(-153. / 128.)};
    for (int w = VTXBUF_LNRHO; w <= VTXBUF_LNRHO; ++w) {
        if (step_number == 0) {
            out->vertex_buffer[w][idx] =(AcReal)(rate_of_change[w] * (Scalar)dt);
        }
        else {
            out->vertex_buffer[w][idx] = (AcReal) (alpha[step_number] * (Scalar)out->vertex_buffer[w][idx] +
                                         rate_of_change[w] * (Scalar)dt);
        }
    }
}

static void
solve_beta_step(const AcMesh in, const int step_number, const AcReal dt, const int i, const int j,
                const int k, AcMesh* out)
{
    const int idx = acVertexBufferIdx(i, j, k, in.info);

    // Williamson (1980) NOTE: older version of astaroth used inhomogenous
    const Scalar beta[] = {(Scalar)(1. / 3.), (Scalar)(15. / 16.), (Scalar)(8. / 15.)};

    for (int w = VTXBUF_LNRHO; w <= VTXBUF_LNRHO; ++w)
        out->vertex_buffer[w][idx] += (AcReal) (beta[step_number] * (Scalar)in.vertex_buffer[w][idx]);

    (void)dt; // Suppress unused variable warning if forcing not used
}

// Checks whether the parameters passed in an AcMeshInfo are valid
static void
checkConfiguration(const AcMeshInfo info)
{
#if AC_VERBOSE
    for (int i = 0; i < NUM_REAL_PARAMS; ++i) {
        if (!is_valid(info.real_params[i])) {
            fprintf(stderr, "WARNING: Passed an invalid value %g to model solver (%s). Skipping.\n",
                    (double)info.real_params[i], realparam_names[i]);
        }
    }

    for (int i = 0; i < NUM_REAL3_PARAMS; ++i) {
        if (!is_valid(info.real3_params[i].x)) {
            fprintf(stderr,
                    "WARNING: Passed an invalid value %g to model solver (%s.x). Skipping.\n",
                    (double)info.real3_params[i].x, realparam_names[i]);
        }
        if (!is_valid(info.real3_params[i].y)) {
            fprintf(stderr,
                    "WARNING: Passed an invalid value %g to model solver (%s.y). Skipping.\n",
                    (double)info.real3_params[i].y, realparam_names[i]);
        }
        if (!is_valid(info.real3_params[i].z)) {
            fprintf(stderr,
                    "WARNING: Passed an invalid value %g to model solver (%s.z). Skipping.\n",
                    (double)info.real3_params[i].z, realparam_names[i]);
        }
    }
#endif

    ERRCHK_ALWAYS(is_valid((Scalar)1. / (Scalar)info.real_params[AC_dsx]));
    ERRCHK_ALWAYS(is_valid((Scalar)1. / (Scalar)info.real_params[AC_dsy]));
    ERRCHK_ALWAYS(is_valid((Scalar)1. / (Scalar)info.real_params[AC_dsz]));
    // ERRCHK_ALWAYS(is_valid(info.real_params[AC_cs2_sound]));
}

AcResult
acHostIntegrateStep(AcMesh mesh, const AcReal dt)
{
    mesh_info = &(mesh.info);

    // Setup built-in parameters
    // mesh_info->real_params[AC_inv_dsx] = (Scalar)(1.0) / mesh_info->real_params[AC_dsx];
    // mesh_info->real_params[AC_inv_dsy] = (Scalar)(1.0) / mesh_info->real_params[AC_dsy];
    // mesh_info->real_params[AC_inv_dsz] = (Scalar)(1.0) / mesh_info->real_params[AC_dsz];
    // mesh_info->real_params[AC_cs2_sound] = mesh_info->real_params[AC_cs_sound] *
    //                                       mesh_info->real_params[AC_cs_sound];
    checkConfiguration(*mesh_info);

    AcMesh intermediate_mesh;
    acHostMeshCreate(mesh.info, &intermediate_mesh);

    const int nx_min = getInt(AC_nx_min);
    const int nx_max = getInt(AC_nx_max);

    const int ny_min = getInt(AC_ny_min);
    const int ny_max = getInt(AC_ny_max);

    const int nz_min = getInt(AC_nz_min);
    const int nz_max = getInt(AC_nz_max);

    for (int step_number = 0; step_number < 3; ++step_number) {

        // Boundconds
        acHostMeshApplyPeriodicBounds(&mesh);

        // Alpha step
        // #pragma omp parallel for
        for (int k = nz_min; k < nz_max; ++k) {
            for (int j = ny_min; j < ny_max; ++j) {
                for (int i = nx_min; i < nx_max; ++i) {
                    solve_alpha_step(mesh, step_number, dt, i, j, k, &intermediate_mesh);
                }
            }
        }

        // Beta step
        // #pragma omp parallel for
        for (int k = nz_min; k < nz_max; ++k) {
            for (int j = ny_min; j < ny_max; ++j) {
                for (int i = nx_min; i < nx_max; ++i) {
                    solve_beta_step(intermediate_mesh, step_number, dt, i, j, k, &mesh);
                }
            }
        }
    }

    acHostMeshDestroy(&intermediate_mesh);
    mesh_info = NULL;
    return AC_SUCCESS;
}

#else  // AC_INTEGRATION_ENABLED == 0
AcResult
acHostIntegrateStep(AcMesh mesh, const AcReal dt)
{
    (void)mesh; // Unused
    (void)dt;   // Unused
    ERROR("Parameters required by acHostIntegrateStep not defined.");
    return AC_FAILURE;
}
#endif // AC_INTEGRATION_ENABLED
