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
#include "astaroth_utils.h"
#include "user_builtin_non_scalar_constants.h"

#include "errchk.h"

static const char dataformat_path[] = "data-format.csv";

AcResult
acHostVertexBufferSet(const VertexBufferHandle handle, const AcReal value, AcMesh* mesh)
{
    const size_t n = acVertexBufferSize(mesh->info);
    for (size_t i = 0; i < n; ++i)
        mesh->vertex_buffer[handle][i] = value;

    return AC_SUCCESS;
}
AcResult
acHostMeshSet(const AcReal value, AcMesh* mesh)
{
    for (size_t w = 0; w < NUM_VTXBUF_HANDLES; ++w)
        acHostVertexBufferSet((VertexBufferHandle)w, value, mesh);

    return AC_SUCCESS;
}
static inline __device__ AcReal UNUSED
lagrangian_correction(const Field j, const Field2 coords, const Volume indexes, const AcReal3 lengths, const Volume nn_min, const Volume nn_max)
{
	const AcReal x_coeff = (j == coords.x)*lengths.x;
	const AcReal y_coeff = (j == coords.y)*lengths.y;
        return  x_coeff*((indexes.x >= nn_max.x) - (indexes.x < nn_min.x))
              + y_coeff*((indexes.y >= nn_max.y) - (indexes.y < nn_min.y));
}

static inline __device__ AcReal UNUSED
lagrangian_correction(const Field j, const Field3 coords, const Volume indexes, const AcReal3 lengths, const Volume nn_min, const Volume nn_max)
{
	const AcReal x_coeff = (j == coords.x)*lengths.x;
	const AcReal y_coeff = (j == coords.y)*lengths.y;
	const AcReal z_coeff = (j == coords.z)*lengths.z;
        return  x_coeff*((indexes.x >= nn_max.x) - (indexes.x < nn_min.x))
              + y_coeff*((indexes.y >= nn_max.y) - (indexes.y < nn_min.y))
              + z_coeff*((indexes.z >= nn_max.z) - (indexes.z < nn_min.z));
}


AcResult
acHostMeshApplyPeriodicBounds(AcMesh* mesh)
{
    const AcMeshInfo info = mesh->info;
    for (int w = 0; w < NUM_VTXBUF_HANDLES; ++w) {
	if (!vtxbuf_is_communicated[w]) continue;
        const Volume start = {0, 0, 0};
	const Volume end = acGetGridMM(info);

	const Volume nn = acGetGridNN(info);
	const size_t nx = nn.x;
	const size_t ny = nn.y;
	const size_t nz = nn.z;

	const Volume nn_min = acGetMinNN(info);

	const size_t nx_min = nn_min.x;
	const size_t ny_min = nn_min.y;
	const size_t nz_min = nn_min.z;


        // The old kxt was inclusive, but our mx_max is exclusive
	const Volume nn_max  = acGetGridMaxNN(info);
        const size_t nx_max = nn_max.x;
        const size_t ny_max = nn_max.y;
        const size_t nz_max = nn_max.z;

        // #pragma omp parallel for
        for (size_t k_dst = start.z; k_dst < end.z; ++k_dst) {
            for (size_t j_dst = start.y; j_dst < end.y; ++j_dst) {
                for (size_t i_dst = start.x; i_dst < end.x; ++i_dst) {

                    // If destination index is inside the computational domain, return since
                    // the boundary conditions are only applied to the ghost zones
                    if (i_dst >= nx_min && i_dst < nx_max && j_dst >= ny_min && j_dst < ny_max &&
                        k_dst >= nz_min && k_dst < nz_max)
                        continue;
                    // Find the source index
                    // Map to nx, ny, nz coordinates
                    int i_src = i_dst - nx_min;
                    int j_src = j_dst - ny_min;
                    int k_src = k_dst - nz_min;

                    // Translate (s.t. the index is always positive)
                    i_src += nx;
                    j_src += ny;
                    k_src += nz;

                    // Wrap
                    i_src %= nx;
                    j_src %= ny;
                    k_src %= nz;

                    // Map to mx, my, mz coordinates
                    i_src += nx_min;
                    j_src += ny_min;
                    k_src += nz_min;

                    const size_t src_idx = acGridVertexBufferIdx(i_src, j_src, k_src, info);
                    const size_t dst_idx = acGridVertexBufferIdx(i_dst, j_dst, k_dst, info);
                    ERRCHK(src_idx < acGridVertexBufferSize(info));
                    ERRCHK(dst_idx < acGridVertexBufferSize(info));
                    mesh->vertex_buffer[w][dst_idx] = mesh->vertex_buffer[w][src_idx];
#if AC_LAGRANGIAN_GRID
		    {
			const AcReal3 lengths = acGetLengths(info);
			mesh->vertex_buffer[w][dst_idx] += lagrangian_correction((Field)w, AC_COORDS, {i_dst,j_dst,k_dst}, lengths, nn_min, nn_max);
		    }
#endif
                }
            }
        }
    }
    return AC_SUCCESS;
}

AcResult
acHostMeshApplyConstantBounds(const AcReal value, AcMesh* mesh)
{
    const AcMeshInfo info = mesh->info;
    const AcMeshDims dims = acGetGridMeshDims(info);

    for (size_t w = 0; w < NUM_VTXBUF_HANDLES; ++w) {
        for (size_t k = 0; k < as_size_t(dims.m1.z); ++k) {
            for (size_t j = 0; j < as_size_t(dims.m1.y); ++j) {
                for (size_t i = 0; i < as_size_t(dims.m1.x); ++i) {
                    if (k >= as_size_t(dims.n0.z) && k < as_size_t(dims.n1.z))
                        if (j >= as_size_t(dims.n0.y) && j < as_size_t(dims.n1.y))
                            if (i >= as_size_t(dims.n0.x) && i < as_size_t(dims.n1.x))
                                continue;

                    const size_t idx            = acGridVertexBufferIdx(i, j, k, info);
                    mesh->vertex_buffer[w][idx] = value;
                }
            }
        }
    }
    return AC_SUCCESS;
}

AcResult
acHostMeshClear(AcMesh* mesh)
{
    return acHostMeshSet(0, mesh);
}

AcResult
acHostMeshWriteToFile(const AcMesh mesh, const size_t id)
{
    const Volume mm = acGetGridMM(mesh.info);
    FILE* header = fopen(dataformat_path, "w");
    ERRCHK_ALWAYS(header);
    fprintf(header, "use_double, mx, my, mz\n");
    fprintf(header, "%d, %ld, %ld, %ld\n", sizeof(AcReal) == 8, mm.x, mm.y, mm.z);
    fclose(header);

    for (size_t i = 0; i < NUM_FIELDS; ++i) {
        const size_t len = 4096;
        char buf[len];
        const int retval = snprintf(buf, len, "%s-%.5lu.dat", field_names[i], id);
        ERRCHK_ALWAYS(retval >= 0);
        ERRCHK_ALWAYS((size_t)retval <= len);

        FILE* fp = fopen(buf, "w");
        ERRCHK_ALWAYS(fp);

        const size_t bytes = sizeof(mesh.vertex_buffer[i][0]);
        const size_t count = acVertexBufferSize(mesh.info);
        const size_t res   = fwrite(mesh.vertex_buffer[i], bytes, count, fp);
        ERRCHK_ALWAYS(res == count);

        fclose(fp);
    }
    return AC_SUCCESS;
}

AcResult
acHostMeshReadFromFile(const size_t id, AcMesh* mesh)
{
    const size_t len = 4096;
    char buf[len];
    int use_double, mx, my, mz;

    FILE* header = fopen(dataformat_path, "r");
    ERRCHK_ALWAYS(header);
    fgets(buf, len, header);
    fscanf(header, "%d, %d, %d, %d\n", &use_double, &mx, &my, &mz);
    fclose(header);
    const Volume mm = acGetGridMM(mesh->info);

    ERRCHK_ALWAYS(use_double == (sizeof(AcReal) == 8));
    ERRCHK_ALWAYS(mx == (int)mm.x);
    ERRCHK_ALWAYS(my == (int)mm.y);
    ERRCHK_ALWAYS(mz == (int)mm.z);

    for (size_t i = 0; i < NUM_FIELDS; ++i) {

        const int retval = snprintf(buf, len, "%s-%.5lu.dat", field_names[i], id);
        ERRCHK_ALWAYS(retval >= 0);
        ERRCHK_ALWAYS((size_t)retval <= len);

        FILE* fp = fopen(buf, "r");
        ERRCHK_ALWAYS(fp);

        const size_t bytes = sizeof(mesh->vertex_buffer[i][0]);
        const size_t count = acVertexBufferSize(mesh->info);
        const size_t res   = fread(mesh->vertex_buffer[i], bytes, count, fp);
        ERRCHK_ALWAYS(res == count);

        fclose(fp);
    }
    return AC_SUCCESS;
}
