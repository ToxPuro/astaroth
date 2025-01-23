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
#ifndef AC_INSIDE_AC_LIBRARY 
#define AC_INSIDE_AC_LIBRARY 
#endif
#include "astaroth.h"
#include "buffer.cc"
#include "user_builtin_non_scalar_constants.h"

#include <string.h> // strcmp

#include "math_utils.h"

static const int max_num_nodes   = 1;
static Node nodes[max_num_nodes] = {0};
static int num_nodes             = 0;

AcResult
acCheckDeviceAvailability(void)
{
    int device_count; // Separate from num_devices to avoid side effects
    ERRCHK_CUDA_ALWAYS(cudaGetDeviceCount(&device_count));
    if (device_count > 0)
        return AC_SUCCESS;
    else
        return AC_FAILURE;
}

int
acGetNumDevicesPerNode(void)
{
    int num_devices;
    ERRCHK_CUDA_ALWAYS(cudaGetDeviceCount(&num_devices));
    return num_devices;
}

size_t
acGetNumFields(void)
{
    return NUM_VTXBUF_HANDLES;
}

AcResult
acGetFieldHandle(const char* field, size_t* handle)
{
    for (size_t i = 0; i < NUM_VTXBUF_HANDLES; ++i) {
        if (!strcmp(field, field_names[i])) {
            *handle = i;
            return AC_SUCCESS;
        }
    }

    *handle = SIZE_MAX;
    return AC_FAILURE;
}

Node
acGetNode(void)
{
    ERRCHK_ALWAYS(num_nodes > 0);
    return nodes[0];
}

AcReal*
acHostCreateVertexBuffer(const AcMeshInfo info)
{
    const size_t n_cells = acVertexBufferSize(info);
    AcReal* res = (AcReal*)calloc(n_cells, sizeof(AcReal));
    ERRCHK_ALWAYS(res);
    return res;
}

AcResult
acHostMeshCreateProfiles(AcMesh* mesh)
{
    const auto mm = acGetLocalMM(mesh->info);
    const size3_t counts = (size3_t){as_size_t(mm.x),as_size_t(mm.y),as_size_t(mm.z)};
    for(size_t p = 0; p < NUM_PROFILES; ++p)
    {
	    mesh->profile[p] = (AcReal*)calloc(prof_size(Profile(p),counts), sizeof(AcReal));
            ERRCHK_ALWAYS(mesh->profile[p]);
    }
    return AC_SUCCESS;
}

AcResult
acHostMeshCreate(const AcMeshInfo info, AcMesh* mesh)
{
    mesh->info = info;
    acHostUpdateBuiltinParams(&mesh->info);
    for (size_t w = 0; w < NUM_VTXBUF_HANDLES; ++w) 
	mesh->vertex_buffer[w] = acHostCreateVertexBuffer(mesh->info);
    return acHostMeshCreateProfiles(mesh);
}
AcResult
acHostMeshCopyVertexBuffers(const AcMesh src, AcMesh dst)
{
    for (size_t w = 0; w < NUM_VTXBUF_HANDLES; ++w) {
	ERRCHK_ALWAYS(src.vertex_buffer[w]);
	memcpy(dst.vertex_buffer[w], src.vertex_buffer[w], acVertexBufferSizeBytes(src.info));
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
        mesh->vertex_buffer[w] = (AcReal*)calloc(n_cells, sizeof(AcReal));
        ERRCHK_ALWAYS(mesh->vertex_buffer[w]);
    }

    return AC_SUCCESS;
}
AcResult
acVerifyCompatibility(const size_t mesh_size, const size_t mesh_info_size, const size_t params_size, const size_t comp_info, const int num_reals, 
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
	if(params_size != sizeof(AcMeshInfoParams))
	{
		fprintf(stderr,"Astaroth warning: mismatch in AcMeshInfoParams size: %zu|%zu\n",mesh_info_size,sizeof(AcMeshInfoParams));
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
    const size_t n = acVertexBufferSize(mesh->info);
    for (size_t w = 0; w < NUM_VTXBUF_HANDLES; ++w) {
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
	free(*vtxbuf);
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

AcResult
acReduceProfileX(const Profile prof, const AcMeshDims dims, const AcReal* src, AcReal** tmp, size_t* tmp_size, AcReal* dst, const cudaStream_t stream)
{
    if constexpr (NUM_PROFILES == 0) return AC_FAILURE;
    const AcProfileType type = prof_types[prof];
    const AcMeshOrder order    = acGetMeshOrderForProfile(type);
    dst += NGHOST;

    const AcBuffer buffer_in =
    {
	    (AcReal*)src,
	    as_size_t(dims.nn.x*dims.nn.y*dims.nn.z),
	    true,
	    (AcShape){
		    as_size_t(dims.nn.x),
		    as_size_t(dims.nn.y),
		    as_size_t(dims.nn.z),
		    1
	    },
    };

    AcBuffer buffer = acTransposeBuffer(buffer_in, order, stream);
    cudaDeviceSynchronize();

    const size_t num_segments = (type & ONE_DIMENSIONAL_PROFILE) ? buffer.shape.z*buffer.shape.w
	                                                         : buffer.shape.y*buffer.shape.z*buffer.shape.w;
    fprintf(stderr,"PROFILE X n segments: %zu\n",num_segments);
    fprintf(stderr,"PROFILE X elems: %zu\n",buffer.count);
    acSegmentedReduce(stream, buffer.data, buffer.count, num_segments, dst,tmp,tmp_size);

    acBufferDestroy(&buffer);
    return AC_SUCCESS;

}


AcResult
acReduceProfile(const Profile prof, AcReduceBuffer buffer, AcReal* dst, const cudaStream_t stream)
{
    if constexpr (NUM_PROFILES == 0) return AC_FAILURE;
    if(!reduced_profiles[prof])      return AC_NOT_ALLOCATED;
    const AcProfileType type = prof_types[prof];
    if(type & ONE_DIMENSIONAL_PROFILE) dst += NGHOST;
    const AcMeshOrder order    = acGetMeshOrderForProfile(type);
    acTranspose(order,buffer.src.data,buffer.transposed.data,get_volume_from_shape(buffer.src.shape),stream);

    const size_t num_segments = (type & ONE_DIMENSIONAL_PROFILE) ? buffer.transposed.shape.z*buffer.transposed.shape.w
	                                                         : buffer.transposed.shape.y*buffer.transposed.shape.z*buffer.transposed.shape.w;
    acSegmentedReduce(stream, buffer.transposed.data, buffer.transposed.count, num_segments, dst,buffer.cub_tmp,buffer.cub_tmp_size);
    return AC_SUCCESS;
}


#include "../config_helpers.h"
void
acStoreConfig(const AcMeshInfo info, const char* filename)
{
	FILE* fp = fopen(filename,"w");
	AcScalarTypes::run<load_scalars>(info.params.scalars, fp, "", false);
	AcArrayTypes::run<load_arrays>(info.params.arrays,info.params.scalars,    fp, "", false);

	AcScalarCompTypes::run<load_comp_scalars>(info.run_consts, fp, "", false);
	AcArrayCompTypes::run<load_comp_arrays>(info.run_consts,    fp, "", false);
	fclose(fp);
}

