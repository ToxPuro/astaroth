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
acHostUpdateBuiltinParams(AcMeshInfo* config)
{
    ERRCHK_ALWAYS(config->int_params[AC_nx] > 0 || config->int_params[AC_nxgrid] > 0);
    ERRCHK_ALWAYS(config->int_params[AC_ny] > 0 || config->int_params[AC_nxgrid] > 0);
#if TWO_D == 0
    ERRCHK_ALWAYS(config->int_params[AC_nz] > 0 || config->int_params[AC_nxgrid] > 0);
#endif
    if(config->int_params[AC_nx] <= 0)
	config->int_params[AC_nx] = config->int_params[AC_nxgrid];
    if(config->int_params[AC_ny] <= 0)
	config->int_params[AC_ny] = config->int_params[AC_nygrid];
#if TWO_D == 0
    if(config->int_params[AC_nz] <= 0)
	config->int_params[AC_nz] = config->int_params[AC_nzgrid];
#endif

    config->int_params[AC_mx] = config->int_params[AC_nx] + STENCIL_ORDER;
    ///////////// PAD TEST
    // config->int_params[AC_mx] = config->int_params[AC_nx] + STENCIL_ORDER + PAD_SIZE;
    ///////////// PAD TEST
    config->int_params[AC_my] = config->int_params[AC_ny] + STENCIL_ORDER;
#if TWO_D == 0 
    config->int_params[AC_mz] = config->int_params[AC_nz] + STENCIL_ORDER;
#endif

    // Bounds for the computational domain, i.e. nx_min <= i < nx_max
    config->int_params[AC_nx_min] = STENCIL_ORDER / 2;
    config->int_params[AC_ny_min] = STENCIL_ORDER / 2;
#if TWO_D == 0
    config->int_params[AC_nz_min] = STENCIL_ORDER / 2;
#endif

    config->int_params[AC_nx_max] = config->int_params[AC_nx_min] + config->int_params[AC_nx];
    config->int_params[AC_ny_max] = config->int_params[AC_ny_min] + config->int_params[AC_ny];
#if TWO_D == 0
    config->int_params[AC_nz_max] = config->int_params[AC_nz_min] + config->int_params[AC_nz];
#endif

    /*
    #ifdef AC_dsx
        printf("HELLO!\n");
        ERRCHK_ALWAYS(config->real_params[AC_dsx] > 0);
        config->real_params[AC_inv_dsx] = (AcReal)(1.) / config->real_params[AC_dsx];
        ERRCHK_ALWAYS(is_valid(config->real_params[AC_inv_dsx]));
    #endif
    #ifdef AC_dsy
        ERRCHK_ALWAYS(config->real_params[AC_dsy] > 0);
        config->real_params[AC_inv_dsy] = (AcReal)(1.) / config->real_params[AC_dsy];
        ERRCHK_ALWAYS(is_valid(config->real_params[AC_inv_dsy]));
    #endif
    #ifdef AC_dsz
        ERRCHK_ALWAYS(config->real_params[AC_dsz] > 0);
        config->real_params[AC_inv_dsz] = (AcReal)(1.) / config->real_params[AC_dsz];
        ERRCHK_ALWAYS(is_valid(config->real_params[AC_inv_dsz]));
    #endif
    */

    /* Additional helper params */
    // Int helpers
    config->int_params[AC_mxy]  = config->int_params[AC_mx] * config->int_params[AC_my];
    config->int_params[AC_nxy]  = config->int_params[AC_nx] * config->int_params[AC_ny];

    config->real_params[AC_xlen] = config->int_params[AC_nxgrid]*config->real_params[AC_dsx];
    config->real_params[AC_ylen] = config->int_params[AC_nygrid]*config->real_params[AC_dsy];
#if TWO_D == 0
    config->int_params[AC_nxyz] = config->int_params[AC_nxy] * config->int_params[AC_nz];
    config->real_params[AC_zlen] = config->int_params[AC_nzgrid]*config->real_params[AC_dsz];
#endif

    return AC_SUCCESS;
}

AcResult
acSetMeshDims(const size_t nx, const size_t ny, const size_t nz, AcMeshInfo* info)
{
#if TWO_D == 1
	(void)nz;
#endif
    info->int_params[AC_nxgrid] = nx;
    info->int_params[AC_nygrid] = ny;
#if TWO_D == 0
    info->int_params[AC_nzgrid] = nz;
#endif
    
    //needed to keep since before acGridInit the user can call this arbitary number of times
    info->int_params[AC_nx] = nx;
    info->int_params[AC_ny] = ny;
#if TWO_D  == 0
    info->int_params[AC_nz] = nz;
#endif
    return acHostUpdateBuiltinParams(info);
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
		fprintf(stderr,"Astaroth warning: mismatch in NUM_INT_PARAMS : %d|%d\n",num_bools,NUM_BOOL_PARAMS);
	}
	if(num_int_arrays != NUM_INT_ARRAYS)
	{
		fprintf(stderr,"Astaroth warning: mismatch in NUM_INT_ARRAYS: %d|%d\n",num_int_arrays,NUM_INT_ARRAYS);
	}
	if(num_bool_arrays != NUM_BOOL_ARRAYS)
	{
		fprintf(stderr,"Astaroth warning: mismatch in NUM_INT_ARRAYS: %d|%d\n",num_bool_arrays,NUM_BOOL_ARRAYS);
	}
	if(num_real_arrays != NUM_REAL_ARRAYS)
	{
		fprintf(stderr,"Astaroth warning: mismatch in NUM_INT_ARRAYS: %d|%d\n",num_real_arrays,NUM_REAL_ARRAYS);
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
acGetKernelId(const Kernel kernel)
{
    for (size_t id = 0; id < NUM_KERNELS; ++id) {
        if (kernel == kernels[id])
            return id;
    }
    fprintf(stderr, "acGetKernelId failed: did not find kernel %p from the list of kernels\n",
            kernel);
    return (size_t)-1;
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
