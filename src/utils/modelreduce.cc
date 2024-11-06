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
#include "astaroth.h"
#include "astaroth_utils.h"

#include <math.h>

#include "errchk.h"

#if AC_DOUBLE_PRECISION == 0 // HACK TODO fix, make cleaner (purkkaratkaisu)
#define fabs fabsf
#define exp expf
#define sqrt sqrtf
#endif

// Function pointer definitions
typedef long double (*ReduceFunc)(const long double, const long double);
typedef long double (*ReduceInitialScalFunc)(const long double);
typedef long double (*ReduceInitialVecFunc)(const long double, const long double,
                                            const long double);
typedef long double (*ReduceInitialVecScalFunc)(const long double, const long double,
                                                const long double, const long double);

// clang-format off
/* Comparison funcs */
static inline long double
max(const long double a, const long double b) { return a > b ? a : b; }

static inline long double
min(const long double a, const long double b) { return a < b ? a : b; }

static inline long double
sum(const long double a, const long double b) { return a + b; }

/* Function used to determine the values used during reduction */
static inline long double
length_scal(const long double a) { return (long double)(a); }

static inline long double
length_vec(const long double a, const long double b, const long double c) { return sqrtl(a*a + b*b + c*c); }

static inline long double
squared_scal(const long double a) { return (long double)(a*a); }

static inline long double
squared_vec(const long double a, const long double b, const long double c) { return squared_scal(a) + squared_scal(b) + squared_scal(c); }

static inline long double
exp_squared_scal(const long double a) { return expl(a)*expl(a); }

static inline long double
exp_squared_vec(const long double a, const long double b, const long double c) { return exp_squared_scal(a) + exp_squared_scal(b) + exp_squared_scal(c); }

static inline long double
length_alf(const long double a, const long double b, const long double c, const long double d) { return sqrtl(a*a + b*b + c*c)/sqrtl(expl(d)); }

static inline long double
squared_alf(const long double a, const long double b, const long double c, const long double d) { return (squared_scal(a) + squared_scal(b) + squared_scal(c))/(expl(d)); }
// clang-format on
//
int
get_initial_idx(AcMeshInfo info)
{
#if TWO_D == 0
    const int initial_idx = acGridVertexBufferIdx(NGHOST_X,
                                              NGHOST_Y,
                                              NGHOST_Z, info);
#else
    const int initial_idx = acGridVertexBufferIdx(NGHOST_X,
                                              NGHOST_Y,
                                              0,info);
#endif
    return initial_idx;

}
long double
get_inv_n(AcMeshInfo info)
{
#if TWO_D == 0
	const int n_grid_points = info[AC_nxyzgrid];
#else
	const int n_grid_points = info[AC_nxygrid];
#endif
        return (long double)1.0l / n_grid_points;
}

AcReal
acHostReduceScal(const AcMesh mesh, const AcReduction reduction, const VertexBufferHandle a)
{
    ReduceInitialScalFunc reduce_initial;
    ReduceFunc reduce;

    switch (reduction.map_vtxbuf_single)
    {
      case AC_MAP_VTXBUF: {
         reduce_initial = length_scal;
         break;
       }
      case AC_MAP_VTXBUF_SQUARE: {
         reduce_initial = squared_scal;
         break;
       }
      case AC_MAP_VTXBUF_EXP_SQUARE: {
         reduce_initial = exp_squared_scal;
         break;
       }
       default:
         ERROR("Unrecognized RTYPE");
    } 

    switch (reduction.reduce_op)
    {
      case REDUCE_SUM: {
         reduce = sum;
         break;
       }
      case REDUCE_MAX: {
         reduce = max;
         break;
       }
      case REDUCE_MIN: {
         reduce = min;
         break;
       }
       default:
         ERROR("Unrecognized RTYPE");
    } 

    const int initial_idx = get_initial_idx(mesh.info);

    long double res;
    if (reduction.reduce_op == REDUCE_MAX || reduction.reduce_op == REDUCE_MIN)
        res = reduce_initial((long double)mesh.vertex_buffer[a][initial_idx]);
    else
        res = 0;

    const int3 mins = acGetMinNN(mesh.info);
    const int3 maxs = acGetGridMaxNN(mesh.info);
    for (int k = mins.z; k < maxs.z; ++k) {
        for (int j = mins.y; j < maxs.y; ++j) {
            for (int i = mins.x; i < maxs.x;
                 ++i) {
                const int idx              = acGridVertexBufferIdx(i, j, k, mesh.info);
                const long double curr_val = reduce_initial(
                    (long double)mesh.vertex_buffer[a][idx]);
                res = reduce(res, curr_val);
            }
        }
    }
    // fprintf(stderr, "%s host result %g\n", rtype_names[rtype], res);
    if (reduction.post_processing_op) {
	const long double inv_n = get_inv_n(mesh.info);
        return (AcReal) sqrtl(inv_n * res);
    }
    else {
        return (AcReal) res;
    }
}

AcReal
acHostReduceVec(const AcMesh mesh, const AcReduction reduction, const VertexBufferHandle a,
                const VertexBufferHandle b, const VertexBufferHandle c)
{
    // AcReal (*reduce_initial)(AcReal, AcReal, AcReal);
    ReduceInitialVecFunc reduce_initial;
    ReduceFunc reduce;

    switch (reduction.map_vtxbuf_vec)
    {
      case AC_MAP_VTXBUF3_NORM: {
         reduce_initial = length_vec;
         break;
       }
      case AC_MAP_VTXBUF3_SQUARE: {
         reduce_initial = squared_vec;
         break;
       }
      case AC_MAP_VTXBUF3_EXP_SQUARE: {
         reduce_initial = exp_squared_vec;
         break;
       }
       default:
         ERROR("Unrecognized RTYPE");
    } 

    switch (reduction.reduce_op)
    {
      case REDUCE_SUM: {
         reduce = sum;
         break;
       }
      case REDUCE_MAX: {
         reduce = max;
         break;
       }
      case REDUCE_MIN: {
         reduce = min;
         break;
       }
       default:
         ERROR("Unrecognized RTYPE");
    }

    const int initial_idx = get_initial_idx(mesh.info);

    long double res;
    if (reduction.reduce_op == REDUCE_MAX || reduction.reduce_op == REDUCE_MIN)
        res = reduce_initial((long double)mesh.vertex_buffer[a][initial_idx],
                             (long double)mesh.vertex_buffer[b][initial_idx],
                             (long double)mesh.vertex_buffer[c][initial_idx]);
    else
        res = 0;
    const int3 mins = acGetMinNN(mesh.info);
    const int3 maxs = acGetGridMaxNN(mesh.info);

    for (int k = mins.z; k < maxs.z; ++k) {
        for (int j = mins.y; j < maxs.y; j++) {
            for (int i = mins.x; i < maxs.x;
                 i++) {
                const int idx              = acGridVertexBufferIdx(i, j, k, mesh.info);
                const long double curr_val = reduce_initial((long double)mesh.vertex_buffer[a][idx],
                                                            (long double)mesh.vertex_buffer[b][idx],
                                                            (long double)
                                                                mesh.vertex_buffer[c][idx]);
                res                        = reduce(res, curr_val);
            }
        }
    }

    if (reduction.post_processing_op) {
	const long double inv_n = get_inv_n(mesh.info);
        return (AcReal) sqrtl(inv_n * res);
    }
    else {
        return (AcReal) res;
    }
}

AcReal
acHostReduceVecScal(const AcMesh mesh, const AcReduction reduction, const VertexBufferHandle a,
                    const VertexBufferHandle b, const VertexBufferHandle c,
                    const VertexBufferHandle d)
{
    // AcReal (*reduce_initial)(AcReal, AcReal, AcReal);
    ReduceInitialVecScalFunc reduce_initial;
    ReduceFunc reduce;



    switch (reduction.map_vtxbuf_vec_scal)
    {
	    case AC_MAP_VTXBUF4_ALFVEN_NORM:
	    {
		    reduce_initial = length_alf;
		    break;
	    }
	    case AC_MAP_VTXBUF4_ALFVEN_SQUARE:
	    {
		    reduce_initial = squared_alf;
		    break;
	    }
            default:
         	ERROR("Unrecognized RTYPE");
    }

    switch (reduction.reduce_op)
    {
      case REDUCE_SUM: {
         reduce = sum;
         break;
       }
      case REDUCE_MAX: {
         reduce = max;
         break;
       }
      case REDUCE_MIN: {
         reduce = min;
         break;
       }
       default:
         ERROR("Unrecognized RTYPE");
    }

    const int initial_idx = get_initial_idx(mesh.info);

    long double res;
    if (reduction.reduce_op == REDUCE_MAX || reduction.reduce_op == REDUCE_MIN)
        res = reduce_initial((long double)mesh.vertex_buffer[a][initial_idx],
                             (long double)mesh.vertex_buffer[b][initial_idx],
                             (long double)mesh.vertex_buffer[c][initial_idx],
                             (long double)mesh.vertex_buffer[d][initial_idx]);
    else
        res = 0;

    const int3 mins = acGetMinNN(mesh.info);
    const int3 maxs = acGetGridMaxNN(mesh.info);
    for (int k = mins.z; k < maxs.z; ++k) {
        for (int j = mins.y; j < maxs.y; j++) {
            for (int i = mins.x; i < maxs.x;
                 i++) {
                const int idx              = acGridVertexBufferIdx(i, j, k, mesh.info);
                const long double curr_val = reduce_initial((long double)mesh.vertex_buffer[a][idx],
                                                            (long double)mesh.vertex_buffer[b][idx],
                                                            (long double)mesh.vertex_buffer[c][idx],
                                                            (long double)
                                                                mesh.vertex_buffer[d][idx]);
                res                        = reduce(res, curr_val);
            }
        }
    }

    if (reduction.post_processing_op) {
	const long double inv_n = get_inv_n(mesh.info);
        return (AcReal) sqrtl(inv_n * res);
    }
    else {
        return (AcReal) res;
    }
}
