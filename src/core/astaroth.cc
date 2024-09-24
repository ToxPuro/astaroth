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

AcReal
acReduceScal(const ReductionType rtype, const VertexBufferHandle vtxbuf_handle)
{
#if TWO_D == 1
    (void)rtype;
    (void)vtxbuf_handle;
    fprintf(stderr,"acReduceScal not supported for 2D simulations\n");
    exit(EXIT_FAILURE);
#else
    ERRCHK_ALWAYS(num_nodes);

    AcReal result;
    acNodeReduceScal(nodes[0], STREAM_DEFAULT, rtype, vtxbuf_handle, &result);
    return result;
#endif
}

AcReal
acReduceVec(const ReductionType rtype, const VertexBufferHandle a, const VertexBufferHandle b,
            const VertexBufferHandle c)
{
#if TWO_D == 1
    (void)rtype;
    (void)a;
    (void)b;
    (void)c;
    fprintf(stderr,"acReduceVec not supported for 2D simulations\n");
    exit(EXIT_FAILURE);
    return AC_FAILURE;
#else
    ERRCHK_ALWAYS(num_nodes);

    AcReal result;
    acNodeReduceVec(nodes[0], STREAM_DEFAULT, rtype, a, b, c, &result);
    return result;
#endif
}

AcReal
acReduceVecScal(const ReductionType rtype, const VertexBufferHandle a, const VertexBufferHandle b,
                const VertexBufferHandle c, const VertexBufferHandle d)
{
#if TWO_D == 1
    (void)rtype;
    (void)a;
    (void)b;
    (void)c;
    (void)d;
    fprintf(stderr,"acReduceVecScal not supported for 2D simulations\n");
    exit(EXIT_FAILURE);
    return AC_FAILURE;
#else
    ERRCHK_ALWAYS(num_nodes);

    AcReal result;
    acNodeReduceVecScal(nodes[0], STREAM_DEFAULT, rtype, a, b, c, d, &result);
    return result;
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
    const size_t n_cells = acVertexBufferSize(mesh->info);
    for (size_t w = 0; w < NUM_VTXBUF_HANDLES; ++w) {
        mesh->vertex_buffer[w] = (AcReal*)calloc(n_cells, sizeof(AcReal));
        ERRCHK_ALWAYS(mesh->vertex_buffer[w]);
    }

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
#if TWO_D == 0
	auto z = AC_nz;
#else
	auto z = 1;
#endif
    return acConstructInt3Param(AC_nx, AC_ny, z, info);
}

int3
acGetLocalMM(const AcMeshInfo info)
{
#if TWO_D == 0
	auto z = AC_mz;
#else
	auto z = 1;
#endif
    return acConstructInt3Param(AC_mx, AC_my, z, info);
}

int3
acGetGridNN(const AcMeshInfo info)
{
#if TWO_D == 0
	auto z = AC_nzgrid;
#else
	auto z = 1;
#endif
    return acConstructInt3Param(AC_nxgrid, AC_nygrid, z, info);
}

int3
acGetGridMM(const AcMeshInfo info)
{
#if TWO_D == 0
	auto z = AC_mzgrid;
#else
	auto z = 1;
#endif
    return acConstructInt3Param(AC_mxgrid, AC_mygrid, z, info);
}

int3
acGetMinNN(const AcMeshInfo info)
{
    return acConstructInt3Param(NGHOST_X, NGHOST_Y, NGHOST_Z, info);
}

int3
acGetMaxNN(const AcMeshInfo info)
{
#if TWO_D == 0
	auto z = AC_nz_max;
#else
	auto z = 1;
#endif
    return acConstructInt3Param(AC_nx_max, AC_ny_max, z, info);
}

int3
acGetGridMaxNN(const AcMeshInfo info)
{
#if TWO_D == 0
	auto z = AC_nzgrid_max;
#else
	auto z = 1;
#endif
    return acConstructInt3Param(AC_nxgrid_max, AC_nygrid_max, z, info);
}

AcReal3
acGetLengths(const AcMeshInfo info)
{
#if TWO_D == 0
	auto z = AC_zlen;
#else
	auto z = -1.0;
#endif
	return acConstructReal3Param(AC_xlen,AC_ylen,z,info);
}


#include "get_vtxbufs_funcs.h"
