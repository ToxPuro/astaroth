/*
    Copyright (C) 2014-2021, Johannes Pekkila, Miikka Vaisala, Oskar Lappi.

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
#include "astaroth_utils.h"

#include <math.h>
#include <stdbool.h>
#include <string.h>

#include "errchk.h"

// Defines for colored output
static inline bool
is_valid(const AcReal a)
{
    return !isnan(a) && !isinf(a);
}

Error
acGetError(const AcReal model, const AcReal candidate)
{
    Error error;
    error.abs_error = 0;

    error.model     = model;
    error.candidate = candidate;

    if (error.model == error.candidate ||
        fabsl((long double)model - (long double)candidate) == 0) { // If exact
        error.abs_error = 0;
        error.rel_error = 0;
        error.ulp_error = 0;
    }
    else if (!is_valid(error.model) || !is_valid(error.candidate)) {
        error.abs_error = (long double)INFINITY;
        error.rel_error = (long double)INFINITY;
        error.ulp_error = (long double)INFINITY;
    }
    else {
        const int base = 2;
        const int p    = sizeof(AcReal) == 4 ? 24 : 53; // Bits in the significant

        const long double e = floorl(logl(fabsl((long double)error.model)) / logl(2));

        const long double ulp             = powl(base, e - (p - 1));
        const long double machine_epsilon = 0.5l * powl(base, -(p - 1));
        error.abs_error                   = fabsl((long double)model - (long double)candidate);
        error.ulp_error                   = error.abs_error / ulp;
        error.rel_error = fabsl(1.0l - (long double)candidate / (long double)model) /
                          machine_epsilon;
    }

    error.maximum_magnitude = error.minimum_magnitude = 0;

    return error;
}

static inline void
print_error_to_file(const char* label, const Error error, const char* path)
{
    FILE* file = fopen(path, "a");
    ERRCHK_ALWAYS(file);
    fprintf(file, "%s, %Lg, %Lg, %Lg, %g, %g\n", label, error.ulp_error, error.abs_error,
            error.rel_error, (double)error.maximum_magnitude, (double)error.minimum_magnitude);
    fclose(file);
}

/** Returns true if the error is acceptable, false otherwise. */
bool
acEvalError(const char* label, const Error error)
{
    // Accept the error if the relative error is < max_ulp_error ulps.
    // Also consider the error zero if it is less than the minimum value in the mesh scaled to
    // machine epsilon
    const long double max_ulp_error = 5;

    bool acceptable;
    if (error.ulp_error < max_ulp_error)
        acceptable = true;
    else if (error.abs_error < (long double)error.minimum_magnitude * (long double)AC_REAL_EPSILON)
        acceptable = true;
    else
        acceptable = false;

    printf("%-15s... %s ", label, acceptable ? AC_GRN "OK!" AC_COL_RESET : AC_RED "FAIL! " AC_COL_RESET);

    printf("| %.3Lg (abs), %.3Lg (ulps), %.3Lg (rel). Range: [%.3g, %.3g]\tpoint: %d,%d,%d\n", //
           error.abs_error, error.ulp_error, error.rel_error,                 //
           (double)error.minimum_magnitude, (double)error.maximum_magnitude,error.x,error.y,error.z);
    print_error_to_file(label, error, "verification.out");

    return acceptable;
}

static AcReal
get_maximum_magnitude(const AcReal* field, const AcMeshInfo info, const bool communicated_field)
{
    AcReal maximum = (AcReal)-INFINITY;

    const int3 nn_min = acGetMinNN(info);
    const int x_start = communicated_field ? 0 : nn_min.x;
    const int y_start = communicated_field ? 0 : nn_min.y;
    const int z_start = communicated_field ? 0 : nn_min.z;

    const int3 nn = acGetGridNN(info);
    const int3 mm = acGetGridMM(info);

    const int x_end = communicated_field ? mm.x : nn.x;
    const int y_end = communicated_field ? mm.y : nn.y;
    const int z_end = communicated_field ? mm.z : nn.z;

    for (int x = x_start; x < x_end; ++x) 
    {
    	for (int y = y_start; y < y_end; ++y) 
	{
    		for (int z = z_start; z < z_end; ++z) 
		{
			const size_t i = acVertexBufferIdx(x,y,z,info);
        		maximum = std::max(maximum, std::abs(field[i]));
		}
	}
    }
    return maximum;
}

static AcReal
get_minimum_magnitude(const AcReal* field, const AcMeshInfo info, const bool communicated_field)
{
    AcReal minimum = (AcReal)INFINITY;

    const int3 nn_min = acGetMinNN(info);
    const int x_start = communicated_field ? 0 : nn_min.x;
    const int y_start = communicated_field ? 0 : nn_min.y;
    const int z_start = communicated_field ? 0 : nn_min.z;

    const int3 nn = acGetGridNN(info);
    const int3 mm = acGetGridMM(info);

    const int x_end = communicated_field ? mm.x : nn.x;
    const int y_end = communicated_field ? mm.y : nn.y;
    const int z_end = communicated_field ? mm.z : nn.z;


    for (int x = x_start; x < x_end; ++x) 
    {
    	for (int y = y_start; y < y_end; ++y) 
	{
    		for (int z = z_start; z < z_end; ++z) 
		{
			const size_t i = acVertexBufferIdx(x,y,z,info);
        		minimum = std::min(minimum, std::abs(field[i]));
		}
	}
    }
    return minimum;
}

// Get the maximum absolute error. Works well if all the values in the mesh are approximately
// in the same range.
// Finding the maximum ulp error is not useful, as it picks up on the noise beyond the
// floating-point precision range and gives huge errors with values that should be considered
// zero (f.ex. 1e-19 and 1e-22 give error of around 1e4 ulps)
static Error
get_max_abs_error(const AcReal* model, const AcReal* candidate, const AcMeshInfo info, const bool communicated_field)
{
    Error error {};
    error.abs_error = -1;


    const int3 nn_min = acGetMinNN(info);

    const int3 nn = acGetGridNN(info);
    const int3 mm = acGetGridMM(info);

    const int x_start = communicated_field ? 0 : nn_min.x;
    const int y_start = communicated_field ? 0 : nn_min.y;
    const int z_start = communicated_field ? 0 : nn_min.z;

    const int x_end = communicated_field ? mm.x : nn.x;
    const int y_end = communicated_field ? mm.y : nn.y;
    const int z_end = communicated_field ? mm.z : nn.z;

    for (int x = x_start; x < x_end; ++x) 
    {
    	for (int y = y_start; y < y_end; ++y) 
	{
    		for (int z = z_start; z < z_end; ++z) 
		{
			const size_t i = acVertexBufferIdx(x,y,z,info);
        		Error curr_error = acGetError(model[i], candidate[i]);
        		if (curr_error.abs_error > error.abs_error)
			{
            			error = curr_error;
				error.x = x;
				error.y = y;
				error.z = z;
			}
		}
	}
    }
    error.maximum_magnitude = get_maximum_magnitude(model, info, communicated_field);
    error.minimum_magnitude = get_minimum_magnitude(model, info, communicated_field);

    return error;
}

/** Returns true when successful, false if errors were found. */
AcResult
acVerifyMesh(const char* label, const AcMesh model, const AcMesh candidate)
{
    printf("---Test: %s---\n", label);
    fflush(stdout);
    printf("Errors at the point of the maximum absolute error:\n");

    int errors_found = 0;
    for (int i = 0; i < NUM_VTXBUF_HANDLES; ++i) {
        const Error error = get_max_abs_error(model.vertex_buffer[i], candidate.vertex_buffer[i],
                                              model.info, vtxbuf_is_communicated[i]);
        const bool acceptable = acEvalError(vtxbuf_names[i], error);
        if (!acceptable)
            ++errors_found;
    }

    if (errors_found > 0)
        printf("Failure. Found %d errors\n", errors_found);

    return errors_found ? AC_FAILURE : AC_SUCCESS;
}

/** Writes an error slice in the z direction */
#if TWO_D == 0
AcResult
acMeshDiffWriteSliceZ(const char* path, const AcMesh model, const AcMesh candidate, const size_t z)
{
    ERRCHK_ALWAYS(NUM_VTXBUF_HANDLES > 0);

    const AcMeshInfo info = model.info;
    const int3 mm = acGetGridMM(info);
    ERRCHK_ALWAYS((int)z < mm.z);

    FILE* fp = fopen(path, "w");
    ERRCHK_ALWAYS(fp);

    const size_t mx = mm.x;
    const size_t my = mm.y;
    for (size_t y = 0; y < my; ++y) {
        for (size_t x = 0; x < mx; ++x) {
            const size_t idx                = acGridVertexBufferIdx(x, y, z, info);
            const VertexBufferHandle vtxbuf = (VertexBufferHandle)0;
            const AcReal m                  = model.vertex_buffer[vtxbuf][idx];
            const AcReal c                  = candidate.vertex_buffer[vtxbuf][idx];
            const Error error               = acGetError(m, c);
            fprintf(fp, "%Lg ", error.ulp_error); // error.abs_error);
        }
        fprintf(fp, "\n");
    }

    fclose(fp);
    return AC_SUCCESS;
}
#endif

/** Writes out the entire diff of two meshes */
AcResult
acMeshDiffWrite(const char* path, const AcMesh model, const AcMesh candidate)
{
    ERRCHK_ALWAYS(NUM_VTXBUF_HANDLES > 0);

    const AcMeshInfo info = model.info;

    FILE* fp = fopen(path, "w");
    ERRCHK_ALWAYS(fp);

    const int3 mm = acGetGridMaxNN(info);
    const size_t mx = mm.x;
    const size_t my = mm.y;
    const size_t mz = mm.z;
    for (size_t z = 0; z < mz; ++z) {
        for (size_t y = 0; y < my; ++y) {
            for (size_t x = 0; x < mx; ++x) {
                const size_t idx                = acGridVertexBufferIdx(x, y, z, info);
                const VertexBufferHandle vtxbuf = (VertexBufferHandle)0;
                const AcReal m                  = model.vertex_buffer[vtxbuf][idx];
                const AcReal c                  = candidate.vertex_buffer[vtxbuf][idx];
                const Error error               = acGetError(m, c);
                fprintf(fp, "%Lg ", error.ulp_error); // error.abs_error);
            }
            fprintf(fp, "\n");
        }
        fprintf(fp, "\n\n");
    }
    fprintf(fp, "\nSTEP_BOUNDARY\n");

    fclose(fp);
    return AC_SUCCESS;
}
