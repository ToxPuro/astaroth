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
#include "datatypes.h"
#include "math_utils_base.h"

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

const char*
acGetFieldName(const Field field)
{
	return field_names[field];
}

size_t
acGetNumFields(void)
{
    return NUM_VTXBUF_HANDLES;
}


#include "get_vtxbufs_funcs.h"
#include "stencil_accesses.h"
