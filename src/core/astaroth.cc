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
acInit(const AcMeshInfo mesh_info)
{
#if TWO_D == 1
     (void)mesh_info;
     fprintf(stderr,"acInit not supported for 2D simulations\n");
     exit(EXIT_FAILURE);
    return AC_FAILURE;
#else
    num_nodes = 1;
    return acNodeCreate(0, mesh_info, &nodes[0]);
#endif
}

AcResult
acQuit(void)
{
#if TWO_D == 1
     fprintf(stderr,"acQuit not supported for 2D simulations\n");
     exit(EXIT_FAILURE);
    return AC_FAILURE;
#else
    ERRCHK_ALWAYS(num_nodes);
    num_nodes = 0;
    return acNodeDestroy(nodes[0]);
#endif
}

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

AcResult
acSynchronize(void)
{
#if TWO_D == 1
     fprintf(stderr,"acSynchronize not supported for 2D simulations\n");
     exit(EXIT_FAILURE);
    return AC_FAILURE;
#else
    ERRCHK_ALWAYS(num_nodes);
    return acNodeSynchronizeStream(nodes[0], STREAM_ALL);
#endif
}

AcResult
acSynchronizeStream(const Stream stream)
{
#if TWO_D == 1
    (void)stream;
    fprintf(stderr,"acSynchronizeStream not supported for 2D simulations\n");
    exit(EXIT_FAILURE);
    return AC_FAILURE;
#else
    ERRCHK_ALWAYS(num_nodes);
    return acNodeSynchronizeStream(nodes[0], stream);
#endif
}

AcResult
acLoadDeviceConstant(const AcRealParam param, const AcReal value)
{
#if TWO_D == 1
    (void)param;
    (void)value;
    fprintf(stderr,"acLoadDeviceConstant not supported for 2D simulations\n");
    exit(EXIT_FAILURE);
    return AC_FAILURE;
#else
    ERRCHK_ALWAYS(num_nodes);
    return acNodeLoadConstant(nodes[0], STREAM_DEFAULT, param, value);
#endif
}

AcResult
acLoad(const AcMesh host_mesh)
{
#if TWO_D == 1
    (void)host_mesh;
    fprintf(stderr,"acLoad not supported for 2D simulations\n");
    exit(EXIT_FAILURE);
    return AC_FAILURE;
#else
    ERRCHK_ALWAYS(num_nodes);
    return acNodeLoadMesh(nodes[0], STREAM_DEFAULT, host_mesh);
#endif
}

AcResult
acSetVertexBuffer(const VertexBufferHandle handle, const AcReal value)
{
#if TWO_D == 1
    (void)value;
    (void)handle;
    fprintf(stderr,"acSetVertexBuffer not supported for 2D simulations\n");
    exit(EXIT_FAILURE);
    return AC_FAILURE;
#else
    ERRCHK_ALWAYS(num_nodes);
    return acNodeSetVertexBuffer(nodes[0], STREAM_DEFAULT, handle, value);
#endif
}

AcResult
acStore(AcMesh* host_mesh)
{
#if TWO_D == 1
    (void)host_mesh;
    fprintf(stderr,"acStore not supported for 2D simulations\n");
    exit(EXIT_FAILURE);
    return AC_FAILURE;
#else
    ERRCHK_ALWAYS(num_nodes);
    return acNodeStoreMesh(nodes[0], STREAM_DEFAULT, host_mesh);
#endif
}

AcResult
acIntegrate(const AcReal dt)
{
#if TWO_D == 1
   (void)dt;
   fprintf(stderr,"acIntegrate not supported for 2D simulations\n");
   exit(EXIT_FAILURE);
   return AC_FAILURE;
#else
    ERRCHK_ALWAYS(num_nodes);
    return acNodeIntegrate(nodes[0], dt);
#endif
}

AcResult
acIntegrateGBC(const AcMeshInfo config, const AcReal dt)
{
#if TWO_D == 1
    (void)config;
    (void)dt;
    fprintf(stderr,"acIntegrateGDBC not supported for 2D simulations\n");
    exit(EXIT_FAILURE);
    return AC_FAILURE;
#else
    ERRCHK_ALWAYS(num_nodes);
    return acNodeIntegrateGBC(nodes[0], config, dt);
#endif
}

AcResult
acIntegrateStep(const int isubstep, const AcReal dt)
{
#if TWO_D == 1
    (void)isubstep;
    (void)dt;
    fprintf(stderr,"acIntegrateStep not supported for 2D simulations\n");
    exit(EXIT_FAILURE);
    return AC_FAILURE;
#else
    ERRCHK_ALWAYS(num_nodes);
    DeviceConfiguration config;
    acNodeQueryDeviceConfiguration(nodes[0], &config);

    const int3 start = (int3){NGHOST, NGHOST, NGHOST};
    const int3 end   = start + config.grid.n;
    return acNodeIntegrateSubstep(nodes[0], STREAM_DEFAULT, isubstep, start, end, dt);
#endif
}

AcResult
acIntegrateStepWithOffset(const int isubstep, const AcReal dt, const int3 start, const int3 end)
{
#if TWO_D == 1
    (void)isubstep;
    (void)dt;
    (void)start; 
    (void)end; 
    fprintf(stderr,"acIntegrateStepWithOffset not supported for 2D simulations\n");
    exit(EXIT_FAILURE);
    return AC_FAILURE;
#else
    ERRCHK_ALWAYS(num_nodes);
    return acNodeIntegrateSubstep(nodes[0], STREAM_DEFAULT, isubstep, start, end, dt);
#endif
}

AcResult
acBoundcondStep(void)
{
#if TWO_D == 1
    fprintf(stderr,"acIntegrateStepWithOffset not supported for 2D simulations\n");
    exit(EXIT_FAILURE);
    return AC_FAILURE;
#else
    ERRCHK_ALWAYS(num_nodes);
    return acNodePeriodicBoundconds(nodes[0], STREAM_DEFAULT);
#endif
}

AcResult
acBoundcondStepGBC(const AcMeshInfo config)
{
#if TWO_D == 1
    (void)config;
    fprintf(stderr,"acBoundcondStepGBC not supported for 2D simulations\n");
    exit(EXIT_FAILURE);
    return AC_FAILURE;
#else
    ERRCHK_ALWAYS(num_nodes);
    return acNodeGeneralBoundconds(nodes[0], STREAM_DEFAULT, config);
#endif
}

AcResult
acStoreWithOffset(const int3 dst, const size_t num_vertices, AcMesh* host_mesh)
{
#if TWO_D == 1
    (void)dst;
    (void)num_vertices;
    (void)host_mesh;
    fprintf(stderr,"acStoreWithOffset not supported for 2D simulations\n");
    exit(EXIT_FAILURE);
    return AC_FAILURE;
#else
    ERRCHK_ALWAYS(num_nodes);
    return acNodeStoreMeshWithOffset(nodes[0], STREAM_DEFAULT, dst, dst, num_vertices, host_mesh);
#endif
}

AcResult
acLoadWithOffset(const AcMesh host_mesh, const int3 src, const int num_vertices)
{
#if TWO_D == 1
     (void)host_mesh;
     (void)src;
     (void)num_vertices;
     fprintf(stderr,"acLoadWithOffset not supported for 2D simulations\n");
     exit(EXIT_FAILURE);
     return AC_FAILURE;
#else
    ERRCHK_ALWAYS(num_nodes);
    return acNodeLoadMeshWithOffset(nodes[0], STREAM_DEFAULT, host_mesh, src, src, num_vertices);
#endif
}

AcResult
acSynchronizeMesh(void)
{
#if TWO_D == 1
     fprintf(stderr,"acSynchronizeMesh not supported for 2D simulations\n");
     exit(EXIT_FAILURE);
     return AC_FAILURE;
#else
    ERRCHK_ALWAYS(num_nodes);
    return acNodeSynchronizeMesh(nodes[0], STREAM_DEFAULT);
#endif
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



AcResult
acHostMeshCreate(const AcMeshInfo info, AcMesh* mesh)
{
    mesh->info = info;
    acHostUpdateBuiltinParams(&mesh->info);
    const size_t n_cells = acVertexBufferSize(mesh->info);
    for (size_t w = 0; w < NUM_VTXBUF_HANDLES; ++w) {
        mesh->vertex_buffer[w] = (AcReal*)calloc(n_cells, sizeof(AcReal));
        ERRCHK_ALWAYS(mesh->vertex_buffer[w]);
    }
    const auto mm = acGetLocalMM(info);
    const size3_t counts = (size3_t){as_size_t(mm.x),as_size_t(mm.y),as_size_t(mm.z)};
    for(size_t p = 0; p < NUM_PROFILES; ++p)
    {
	    mesh->profile[p] = (AcReal*)calloc(prof_size(Profile(p),counts), sizeof(AcReal));
            ERRCHK_ALWAYS(mesh->profile[p]);
    }
    return AC_SUCCESS;
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
acHostMeshDestroy(AcMesh* mesh)
{
    for (size_t w = 0; w < NUM_VTXBUF_HANDLES; ++w)
        free(mesh->vertex_buffer[w]);

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
    return acConstructInt3Param(AC_nx, AC_ny, AC_nz, info);
}

int3
acGetLocalMM(const AcMeshInfo info)
{
    return acConstructInt3Param(AC_mx, AC_my, AC_mz, info);
}

int3
acGetGridNN(const AcMeshInfo info)
{
    return acConstructInt3Param(AC_nxgrid, AC_nygrid, AC_nzgrid, info);
}

int3
acGetGridMM(const AcMeshInfo info)
{
    return acConstructInt3Param(AC_mxgrid, AC_mygrid, AC_mzgrid, info);
}

int3
acGetMinNN(const AcMeshInfo info)
{
    return acConstructInt3Param(NGHOST_X, NGHOST_Y, NGHOST_Z, info);
}

int3
acGetMaxNN(const AcMeshInfo info)
{
    return acConstructInt3Param(AC_nx_max, AC_ny_max, AC_nz_max, info);
}

int3
acGetGridMaxNN(const AcMeshInfo info)
{
    return acConstructInt3Param(AC_nxgrid_max, AC_nygrid_max, AC_nzgrid_max, info);
}

AcReal3
acGetLengths(const AcMeshInfo info)
{
	return acConstructReal3Param(AC_xlen,AC_ylen,AC_zlen,info);
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

AcMeshOrder
acGetMeshOrderForProfile(const AcProfileType type)
{
    	switch(type)
    	{
    	        case(PROFILE_X):
    	    	    return ZYX;
    	        case(PROFILE_Y):
		    return XZY;
    	        case(PROFILE_Z):
			return XYZ;
    	        case(PROFILE_XY):
			return ZXY;
    	        case(PROFILE_XZ):
			return YXZ;
    	        case(PROFILE_YX):
			return ZYX;
    	        case(PROFILE_YZ):
			return XYZ;
    	        case(PROFILE_ZX):
			return YZX;
    	        case(PROFILE_ZY):
			return XZY;
    	}
	return XYZ;
};

static size_t
get_size_from_dim(const int dim, const AcMeshDims dims)
{
    	const auto size   = dim == X_ORDER_INT ? dims.m1.x :
        		    dim == Y_ORDER_INT ? dims.m1.y :
        		    dims.m1.z;
        return as_size_t(size);
}
AcShape
acGetTransposeBufferShape(const AcMeshOrder order, const AcMeshDims dims)
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
			dims
			);
	if(type & ONE_DIMENSIONAL_PROFILE)
	{
		return
		{
			order_size.x - 2*NGHOST,
			order_size.y - 2*NGHOST,
			order_size.z,
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
AcResult
acReduceProfile(const Profile prof, const AcMeshDims dims, const AcReal* src, AcReal* dst, const cudaStream_t stream)
{
    if constexpr (NUM_PROFILES == 0) return AC_FAILURE;
    const AcProfileType type = prof_types[prof];
    const AcShape buffer_shape = acGetReductionShape(type,dims);
    const AcMeshOrder order    = acGetMeshOrderForProfile(type);
    AcBuffer gpu_transpose_buffer = acBufferCreate(acGetTransposeBufferShape(order,dims),true);
    acTranspose(acGetMeshOrderForProfile(type),src,gpu_transpose_buffer.data, dims.m1, stream);
    AcBuffer buffer          = acBufferCreate(buffer_shape, true);
    // Indices and shapes
    const AcIndex in_offset = 
    {
	    .x = NGHOST,
	    .y = NGHOST,
	    .z = 0,
	    .w = 0,
    };
    const AcShape in_shape    = gpu_transpose_buffer.shape;
    const AcShape block_shape = gpu_transpose_buffer.shape;
    // Remove ghost zones
    const AcIndex buffer_offset = {
        .x = 0,
        .y = 0,
        .z = 0,
        .w = 0,
    };
    acReindex(stream,                        //
              gpu_transpose_buffer.data, in_offset, in_shape, //
              buffer.data, buffer_offset, buffer_shape, block_shape);

    const size_t num_segments = (type & ONE_DIMENSIONAL_PROFILE) ? buffer_shape.z*buffer_shape.w
	                                                         : buffer_shape.y*buffer_shape.z*buffer_shape.w;
    acSegmentedReduce(stream, buffer.data, buffer.count, num_segments, dst);

    acBufferDestroy(&buffer);
    acBufferDestroy(&gpu_transpose_buffer);
    return AC_SUCCESS;

}
