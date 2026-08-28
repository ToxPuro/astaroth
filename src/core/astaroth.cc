/*
    Copyright (C) 2014-2026, Johannes Pekkila, Miikka Vaisala, Touko Puro.

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
#include "astaroth.h"

#include <stdlib.h>
#include <string.h>  // strcmp

#include "astaroth_cuda_wrappers.h"
#include "astaroth_legacy.h"
#include "astaroth_node.h"
#include "config_helpers.h"
#include "datatypes.h"
#include "math_utils_base.h"

static
AcReal*
acCallocHostReal(const size_t n_cells)
{
    AcReal* res;
    const size_t bytes = sizeof(AcReal)*n_cells;
    ERRCHK_CUDA_ALWAYS(acMallocHost((void**)&res, bytes));
    ERRCHK_ALWAYS(res);
    memset(res,0,bytes);
    return res;
}

AcReal*
acHostCreateVertexBufferVariable(const AcMeshInfo info, const VertexBufferHandle vtxbuf)
{
    const size_t n_cells = acVertexBufferSize(info,vtxbuf);
    return acCallocHostReal(n_cells);
}

AcReal*
acHostCreateVertexBuffer(const AcMeshInfo info)
{
    const size_t n_cells = acVertexBufferSize(info);
    return acCallocHostReal(n_cells);
}

AcResult
acHostMeshCreateProfiles(AcMesh* mesh)
{
    const auto mm = acGetLocalMM(mesh->info);
    const size3_t counts = (size3_t){as_size_t(mm.x),as_size_t(mm.y),as_size_t(mm.z)};
    for(int p = 0; p < NUM_PROFILES; ++p)
    {
	    mesh->profile[p] = acCallocHostReal(prof_size(Profile(p),counts));
            ERRCHK_ALWAYS(mesh->profile[p]);
    }
    return AC_SUCCESS;
}

AcResult
acHostMeshCreate(const AcMeshInfo info, AcMesh* mesh)
{
    mesh->info = info;
    acHostUpdateParams(&mesh->info);
    for (size_t w = 0; w < NUM_VTXBUF_HANDLES; ++w) 
	mesh->vertex_buffer[w] = acHostCreateVertexBuffer(mesh->info,VertexBufferHandle(w));
    return acHostMeshCreateProfiles(mesh);
}
AcResult
acHostMeshCopyVertexBuffers(const AcMesh src, AcMesh dst)
{
    for (size_t w = 0; w < NUM_VTXBUF_HANDLES; ++w) {
        if(src.vertex_buffer[w] == NULL) continue;
	if(dst.vertex_buffer[w] == NULL) continue;
	memcpy(dst.vertex_buffer[w], src.vertex_buffer[w], acVertexBufferSizeBytes(src.info,VertexBufferHandle(w)));
    }
    return AC_SUCCESS;
}

AcResult
acHostMeshCopy(const AcMesh src, AcMesh* dst)
{
    ERRCHK_ALWAYS(acHostMeshCreate(src.info,dst) == AC_SUCCESS);
    ERRCHK_ALWAYS(acHostMeshCopyVertexBuffers(src,*dst) == AC_SUCCESS);
    return AC_SUCCESS;
}

AcResult
acHostGridMeshCreate(const AcMeshInfo info, AcMesh* mesh)
{
    mesh->info = info;
    const size_t n_cells = acGridVertexBufferSize(mesh->info);
    for (size_t w = 0; w < NUM_VTXBUF_HANDLES; ++w) {
        mesh->vertex_buffer[w] = acCallocHostReal(n_cells);
        ERRCHK_ALWAYS(mesh->vertex_buffer[w]);
    }

    return AC_SUCCESS;
}
AcResult
acVerifyCompatibility(const size_t mesh_size, const size_t mesh_info_size, const size_t comp_info, const int num_reals, 
		      const int num_ints, const int num_bools, const int num_real_arrays,
		      const int num_int_arrays, const int num_bool_arrays)
{
	AcResult res = AC_SUCCESS;
	if(mesh_size != sizeof(AcMesh))
	{
		fprintf(stderr,"Astaroth warning: mismatch in AcMesh size: %zu|%zu\n",mesh_size,sizeof(AcMesh));
		res = AC_FAILURE;
	}
	if(mesh_info_size != sizeof(AcMeshInfo))
	{
		fprintf(stderr,"Astaroth warning: mismatch in AcMeshInfo size: %zu|%zu\n",mesh_info_size,sizeof(AcMeshInfo));
		res = AC_FAILURE;
	}
	if(comp_info != sizeof(AcCompInfo))
	{
		fprintf(stderr,"Astaroth warning: mismatch in AcCompInfo size: %zu|%zu\n",comp_info,sizeof(AcCompInfo));
		res = AC_FAILURE;
	}
	if(num_ints != NUM_INT_PARAMS)
	{
		fprintf(stderr,"Astaroth warning: mismatch in NUM_INT_PARAMS : %d|%d\n",num_ints,NUM_INT_PARAMS);
	}
	if(num_reals != NUM_REAL_PARAMS)
	{
		fprintf(stderr,"Astaroth warning: mismatch in NUM_INT_PARAMS : %d|%d\n",num_reals,NUM_REAL_PARAMS);
	}
	if(num_bools != NUM_BOOL_PARAMS)
	{
		fprintf(stderr,"Astaroth warning: mismatch in NUM_BOOL_PARAMS: %d|%d\n",num_bools,NUM_BOOL_PARAMS);
	}
	if(num_int_arrays != NUM_INT_ARRAYS)
	{
		fprintf(stderr,"Astaroth warning: mismatch in NUM_INT_ARRAYS: %d|%d\n",num_int_arrays,NUM_INT_ARRAYS);
	}
	if(num_bool_arrays != NUM_BOOL_ARRAYS)
	{
		fprintf(stderr,"Astaroth warning: mismatch in NUM_BOOL_ARRAYS: %d|%d\n",num_bool_arrays,NUM_BOOL_ARRAYS);
	}
	if(num_real_arrays != NUM_REAL_ARRAYS)
	{
		fprintf(stderr,"Astaroth warning: mismatch in NUM_REAL_ARRAYS: %d|%d\n",num_real_arrays,NUM_REAL_ARRAYS);
	}
	return res;
}

static AcReal
randf(void)
{
    // TODO: rand() considered harmful, replace
    return (AcReal)rand() / (AcReal)RAND_MAX;
}

AcResult
acHostMeshRandomize(AcMesh* mesh)
{
    for (size_t w = 0; w < NUM_VTXBUF_HANDLES; ++w) {
	if(mesh->vertex_buffer[w] == NULL) continue;
        const size_t n = acVertexBufferSize(mesh->info,VertexBufferHandle(w));
        for (size_t i = 0; i < n; ++i) {
            mesh->vertex_buffer[w][i] = randf();
        }
    }

    return AC_SUCCESS;
}
AcResult
acHostGridMeshRandomize(AcMesh* mesh)
{
    const size_t n = acGridVertexBufferSize(mesh->info);
    for (size_t w = 0; w < NUM_VTXBUF_HANDLES; ++w) {
        for (size_t i = 0; i < n; ++i) {
            mesh->vertex_buffer[w][i] = randf();
        }
    }

    return AC_SUCCESS;
}
AcResult
acHostMeshDestroyVertexBuffer(AcReal** vtxbuf)
{
	if(*vtxbuf == NULL) return AC_SUCCESS;
	acFreeHost(*vtxbuf);
	(*vtxbuf) = NULL;
	return AC_SUCCESS;
}

AcResult
acHostMeshDestroy(AcMesh* mesh)
{
    for (size_t w = 0; w < NUM_VTXBUF_HANDLES; ++w)
	acHostMeshDestroyVertexBuffer(&mesh->vertex_buffer[w]);

    return AC_SUCCESS;
}

/**
    Astaroth helper functions
*/

size_t
acGetKernelId(const AcKernel kernel)
{
	return (size_t) kernel;
}

size_t
acGetKernelIdByName(const char* name)
{
    for (size_t id = 0; id < NUM_KERNELS; ++id) {
        if (!strcmp(kernel_names[id], name))
            return id;
    }
    fprintf(stderr, "acGetKernelIdByName failed: did not find kernel %s from the list of kernels\n",
            name);
    return (size_t)-1;
}

Volume
acGetLocalNN(const AcMeshInfo info)
{
    return to_volume(info[AC_nlocal]);
}

Volume
acGetLocalMM(const AcMeshInfo info)
{
    return to_volume(info[AC_mlocal]);
}

Volume
acGetGridNN(const AcMeshInfo info)
{
    return to_volume(info[AC_ngrid]);
}

Volume
acGetGridMM(const AcMeshInfo info)
{
    return to_volume(info[AC_mgrid]);
}

Volume
acGetMaxNN(const AcMeshInfo info)
{
    return to_volume(info[AC_nlocal_max]);
}

Volume
acGetMinNN(const AcMeshInfo info)
{
    return to_volume(info[AC_nmin]);
}

Volume
acGetGridMaxNN(const AcMeshInfo info)
{
    return to_volume(info[AC_ngrid_max]);
}

AcReal3
acGetLengths(const AcMeshInfo info)
{
	return info[AC_len];
}


#include "get_vtxbufs_funcs.h"
#include "stencil_accesses.h"
void
acStoreConfig(const AcMeshInfo info, const char* filename)
{
	FILE* fp =  filename == NULL ? stdout : fopen(filename,"w");
	AcScalarTypes::run<load_scalars>(info, fp, "", false);
	AcArrayTypes::run<load_arrays>(info,fp, "", false);

	AcScalarCompTypes::run<load_comp_scalars>(info.run_consts, fp, "", false);
	AcArrayCompTypes::run<load_comp_arrays>(info,    fp, "", false);
	if(filename != NULL) fclose(fp);
}

