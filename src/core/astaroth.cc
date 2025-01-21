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
acVerifyCompatibility(const size_t mesh_size, const size_t mesh_info_size, const int num_reals, 
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

int3
acGetLocalNN(const AcMeshInfo info)
{
    return info[AC_nlocal];
}

int3
acGetLocalMM(const AcMeshInfo info)
{
    return info[AC_mlocal];
}

int3
acGetGridNN(const AcMeshInfo info)
{
    return info[AC_ngrid];
}

int3
acGetGridMM(const AcMeshInfo info)
{
    return info[AC_mgrid];
}

int3
acGetMaxNN(const AcMeshInfo info)
{
    return info[AC_nlocal_max];
}

int3
acGetMinNN(const AcMeshInfo info)
{
    return info[AC_nmin];
}

int3
acGetGridMaxNN(const AcMeshInfo info)
{
    return info[AC_ngrid_max];
}

AcReal3
acGetLengths(const AcMeshInfo info)
{
	return info[AC_len];
}


#include "get_vtxbufs_funcs.h"
AcBuffer
acBufferCreate(const AcShape shape, const bool on_device)
{
    AcBuffer buffer    = {.data = NULL, .count = acShapeSize(shape), .on_device = on_device, .shape = shape};
    const size_t bytes = sizeof(buffer.data[0]) * buffer.count;
    if (buffer.on_device) {
        ERRCHK_CUDA_ALWAYS(cudaMalloc((void**)&buffer.data, bytes));
    }
    else {
        buffer.data = (AcReal*)malloc(bytes);
    }
    ERRCHK_ALWAYS(buffer.data);
    return buffer;
}

void
acBufferDestroy(AcBuffer* buffer)
{
    if (buffer->on_device)
        cudaFree(buffer->data);
    else
        free(buffer->data);
    buffer->data  = NULL;
    buffer->count = 0;
}

AcResult
acBufferMigrate(const AcBuffer in, AcBuffer* out)
{
    cudaMemcpyKind kind;
    if (in.on_device) {
        if (out->on_device)
            kind = cudaMemcpyDeviceToDevice;
        else
            kind = cudaMemcpyDeviceToHost;
    }
    else {
        if (out->on_device)
            kind = cudaMemcpyHostToDevice;
        else
            kind = cudaMemcpyHostToHost;
    }

    ERRCHK_ALWAYS(in.count == out->count);
    ERRCHK_CUDA_ALWAYS(cudaMemcpy(out->data, in.data, sizeof(in.data[0]) * in.count, kind));
    return AC_SUCCESS;
}

AcBuffer
acBufferCopy(const AcBuffer in, const bool on_device)
{
    AcBuffer cpu_buffer = acBufferCreate(in.shape, on_device);
    acBufferMigrate(in,&cpu_buffer);
    return cpu_buffer;
}


static size_t
get_size_from_dim(const int dim, const int3 dims)
{
    	const auto size   = dim == X_ORDER_INT ? dims.x :
        		    dim == Y_ORDER_INT ? dims.y :
        		    dims.z;
        return as_size_t(size);
}
AcShape
acGetTransposeBufferShape(const AcMeshOrder order, const int3 dims)
{
	const int first_dim  = order % N_DIMS;
	const int second_dim = (order/N_DIMS) % N_DIMS;
	const int third_dim  = ((order/N_DIMS)/N_DIMS)  % N_DIMS;
	return (AcShape){
		get_size_from_dim(first_dim,dims),
		get_size_from_dim(second_dim,dims),
		get_size_from_dim(third_dim,dims),
		1
	};
}
AcShape
acGetReductionShape(const AcProfileType type, const AcMeshDims dims)
{
	const AcShape order_size = acGetTransposeBufferShape(
			acGetMeshOrderForProfile(type),
			dims.m1
			);
	if(type & ONE_DIMENSIONAL_PROFILE)
	{
		return
		{
			order_size.x - 2*NGHOST,
			order_size.y - 2*NGHOST,
			order_size.z - (type != PROFILE_Z)*2*NGHOST,
			order_size.w
		};
	}
	else if(type & TWO_DIMENSIONAL_PROFILE)
		return
		{
			order_size.x - 2*NGHOST,
			order_size.y,
			order_size.z,
			order_size.w
		};
	return order_size;
}

int3
get_int3_from_shape(const AcShape shape)
{
	return (int3){(int)shape.x,(int)shape.y,(int)shape.z};
}

AcBuffer
acBufferRemoveHalos(const AcBuffer buffer_in, const int3 halo_sizes, const cudaStream_t stream)
{
	const AcShape res_shape = 
	{
		buffer_in.shape.x - 2*halo_sizes.x,
		buffer_in.shape.y - 2*halo_sizes.y,
		buffer_in.shape.z - 2*halo_sizes.z,
		1
	};	

	const AcShape in_offset = 
	{
		as_size_t(halo_sizes.x),
		as_size_t(halo_sizes.y),
		as_size_t(halo_sizes.z),
		0,
	};	

	const AcShape out_offset = 
	{
		0,
		0,
		0,
		0,
	};	

	const AcShape in_shape    = buffer_in.shape;
	const AcShape block_shape = buffer_in.shape;

	AcBuffer dst = acBufferCreate(res_shape,true);
    	acReindex(stream,buffer_in.data, in_offset, in_shape, dst.data, out_offset , dst.shape, block_shape);
	return dst;
}

AcBuffer
acTransposeBuffer(const AcBuffer src, const AcMeshOrder order, const cudaStream_t stream)
{
	const int3 dims = get_int3_from_shape(src.shape);
	AcBuffer dst = acBufferCreate(acGetTransposeBufferShape(order,dims),true);
	acTranspose(order,src.data,dst.data,dims,stream);
	return dst;
}
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
acReduceProfile(const Profile prof, const AcMeshDims dims, const AcReal* src, AcReal** tmp, size_t* tmp_size, AcReal* dst, const cudaStream_t stream)
{
    if constexpr (NUM_PROFILES == 0) return AC_FAILURE;
    const AcProfileType type = prof_types[prof];
    if(type & ONE_DIMENSIONAL_PROFILE) dst += NGHOST;
    const AcMeshOrder order    = acGetMeshOrderForProfile(type);
    auto active_dims = acGetProfileReduceScratchPadDims(prof,as_size_t(dims.m1),as_size_t(dims.nn));
    const AcBuffer buffer_in =
    {
	    (AcReal*)src,
	    active_dims.x*active_dims.y*active_dims.z,
	    true,
	    (AcShape){
		    active_dims.x,
		    active_dims.y,
		    active_dims.z,
		    1
	    },
    };
    AcBuffer buffer = acTransposeBuffer(buffer_in, order, stream);

    const size_t num_segments = (type & ONE_DIMENSIONAL_PROFILE) ? buffer.shape.z*buffer.shape.w
	                                                         : buffer.shape.y*buffer.shape.z*buffer.shape.w;
    acSegmentedReduce(stream, buffer.data, buffer.count, num_segments, dst,tmp,tmp_size);

    acBufferDestroy(&buffer);
    return AC_SUCCESS;
}


#include "../config_helpers.h"
void
acStoreConfig(const AcMeshInfo info, const char* filename)
{
	FILE* fp = fopen(filename,"w");
	AcScalarTypes::run<load_scalars>(info.params.scalars, fp, "", false);
	AcArrayTypes::run<load_arrays>(info.params.arrays,    fp, "", false);

	AcScalarCompTypes::run<load_comp_scalars>(info.run_consts, fp, "", false);
	AcArrayCompTypes::run<load_comp_arrays>(info.run_consts,    fp, "", false);
	fclose(fp);
}
