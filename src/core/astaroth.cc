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
#include "buffer.cc"
#include "user_builtin_non_scalar_constants.h"

#include <string.h> // strcmp

#include "math_utils.h"

static const int max_num_nodes   = 1;
static Node nodes[max_num_nodes] = {0};
static int num_nodes             = 0;

AcResult
acInit(const AcMeshInfo mesh_info)
{
    num_nodes = 1;
    return acNodeCreate(0, mesh_info, &nodes[0]);
}

AcResult
acQuit(void)
{
    ERRCHK_ALWAYS(num_nodes);
    num_nodes = 0;
    return acNodeDestroy(nodes[0]);
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
    ERRCHK_ALWAYS(num_nodes);
    return acNodeSynchronizeStream(nodes[0], STREAM_ALL);
}

AcResult
acSynchronizeStream(const Stream stream)
{
    ERRCHK_ALWAYS(num_nodes);
    return acNodeSynchronizeStream(nodes[0], stream);
}

AcResult
acLoadDeviceConstant(const AcRealParam param, const AcReal value)
{
    ERRCHK_ALWAYS(num_nodes);
    return acNodeLoadConstant(nodes[0], STREAM_DEFAULT, param, value);
}

AcResult
acLoad(const AcMesh host_mesh)
{
    ERRCHK_ALWAYS(num_nodes);
    return acNodeLoadMesh(nodes[0], STREAM_DEFAULT, host_mesh);
}

AcResult
acSetVertexBuffer(const VertexBufferHandle handle, const AcReal value)
{
    ERRCHK_ALWAYS(num_nodes);
    return acNodeSetVertexBuffer(nodes[0], STREAM_DEFAULT, handle, value);
}

AcResult
acStore(AcMesh* host_mesh)
{
    ERRCHK_ALWAYS(num_nodes);
    return acNodeStoreMesh(nodes[0], STREAM_DEFAULT, host_mesh);
}

AcResult
acIntegrate(const AcReal dt)
{
    ERRCHK_ALWAYS(num_nodes);
    return acNodeIntegrate(nodes[0], dt);
}

AcResult
acIntegrateGBC(const AcMeshInfo config, const AcReal dt)
{
    ERRCHK_ALWAYS(num_nodes);
    return acNodeIntegrateGBC(nodes[0], config, dt);
}


AcResult
acIntegrateStep(const int isubstep, const AcReal dt)
{
    ERRCHK_ALWAYS(num_nodes);
    DeviceConfiguration config;
    acNodeQueryDeviceConfiguration(nodes[0], &config);

    const Volume start = (Volume){NGHOST, NGHOST, NGHOST};
    const Volume end   = start + config.grid.n;
    return acNodeIntegrateSubstep(nodes[0], STREAM_DEFAULT, isubstep, start, end, dt);
}

AcResult
acIntegrateStepWithOffset(const int isubstep, const AcReal dt, const Volume start, const Volume end)
{
    ERRCHK_ALWAYS(num_nodes);
    return acNodeIntegrateSubstep(nodes[0], STREAM_DEFAULT, isubstep, start, end, dt);
}

AcResult
acBoundcondStep(void)
{
    ERRCHK_ALWAYS(num_nodes);
    return acNodePeriodicBoundconds(nodes[0], STREAM_DEFAULT);
}

AcResult
acBoundcondStepGBC(const AcMeshInfo config)
{
    ERRCHK_ALWAYS(num_nodes);
    return acNodeGeneralBoundconds(nodes[0], STREAM_DEFAULT, config);
}

AcReal
acReduceScal(const AcReduction reduction, const VertexBufferHandle vtxbuf_handle)
{
    ERRCHK_ALWAYS(num_nodes);

    AcReal result;
    acNodeReduceScal(nodes[0], STREAM_DEFAULT, reduction, vtxbuf_handle, &result);
    return result;
}

AcReal
acReduceVec(const AcReduction reduction, const VertexBufferHandle a, const VertexBufferHandle b,
            const VertexBufferHandle c)
{
    ERRCHK_ALWAYS(num_nodes);

    AcReal result;
    acNodeReduceVec(nodes[0], STREAM_DEFAULT, reduction, a, b, c, &result);
    return result;
}

AcReal
acReduceVecScal(const AcReduction reduction, const VertexBufferHandle a, const VertexBufferHandle b,
                const VertexBufferHandle c, const VertexBufferHandle d)
{
    ERRCHK_ALWAYS(num_nodes);

    AcReal result;
    acNodeReduceVecScal(nodes[0], STREAM_DEFAULT, reduction, a, b, c, d, &result);
    return result;
}

AcResult
acStoreWithOffset(const int3 dst, const size_t num_vertices, AcMesh* host_mesh)
{
    ERRCHK_ALWAYS(num_nodes);
    return acNodeStoreMeshWithOffset(nodes[0], STREAM_DEFAULT, dst, dst, num_vertices, host_mesh);
}

AcResult
acLoadWithOffset(const AcMesh host_mesh, const int3 src, const int num_vertices)
{
    ERRCHK_ALWAYS(num_nodes);
    return acNodeLoadMeshWithOffset(nodes[0], STREAM_DEFAULT, host_mesh, src, src, num_vertices);
}

AcResult
acSynchronizeMesh(void)
{
    ERRCHK_ALWAYS(num_nodes);
    return acNodeSynchronizeMesh(nodes[0], STREAM_DEFAULT);
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
    for(int p = 0; p < NUM_PROFILES; ++p)
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
    acHostUpdateParams(&mesh->info);
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
acReduceProfileWithBounds(const Profile prof, AcReduceBuffer buffer, AcReal* dst, const cudaStream_t stream, const Volume start, const Volume end, const Volume start_after_transpose, const Volume end_after_transpose)
{
    if constexpr (NUM_PROFILES == 0) return AC_FAILURE;
    if(buffer.src.data == NULL)      return AC_NOT_ALLOCATED;
    const AcProfileType type = prof_types[prof];
    const AcMeshOrder order    = acGetMeshOrderForProfile(type);


    acTransposeWithBounds(order,buffer.src.data,buffer.transposed.data,get_volume_from_shape(buffer.src.shape),start,end,stream);

    const Volume dims = end_after_transpose-start_after_transpose;

    const size_t num_segments = (type & ONE_DIMENSIONAL_PROFILE) ? dims.z*buffer.transposed.shape.w
	                                                         : dims.y*dims.z*buffer.transposed.shape.w;

    const size_t count = buffer.transposed.shape.w*dims.x*dims.y*dims.z;

    const AcReal* reduce_src = buffer.transposed.data
	    		      + start_after_transpose.x + start_after_transpose.y*buffer.transposed.shape.x + start_after_transpose.z*buffer.transposed.shape.x*buffer.transposed.shape.y;

    acSegmentedReduce(stream, reduce_src, count, num_segments, dst,buffer.cub_tmp,buffer.cub_tmp_size);
    return AC_SUCCESS;
}

AcResult
acReduceProfile(const Profile prof, AcReduceBuffer buffer, AcReal* dst, const cudaStream_t stream)
{
	return acReduceProfileWithBounds(prof,buffer,dst,stream,(Volume){0,0,0},get_volume_from_shape(buffer.src.shape),(Volume){0,0,0},get_volume_from_shape(buffer.transposed.shape));
}

#include "../config_helpers.h"
void
acStoreConfig(const AcMeshInfo info, const char* filename)
{
	FILE* fp =  filename == NULL ? stdout : fopen(filename,"w");
	AcScalarTypes::run<load_scalars>(info, fp, "", false);
	AcArrayTypes::run<load_arrays>(info,fp, "", false);

	AcScalarCompTypes::run<load_comp_scalars>(info.run_consts, fp, "", false);
	AcArrayCompTypes::run<load_comp_arrays>(info.run_consts,    fp, "", false);
	if(filename != NULL) fclose(fp);
}



void
acQueryIntparams(void)
{
    for (int i = 0; i < NUM_INT_PARAMS; ++i)
        printf("%s (%d)\n", intparam_names[i], i);
}

void
acQueryInt3params(void)
{
    for (int i = 0; i < NUM_INT3_PARAMS; ++i)
        printf("%s (%d)\n", int3param_names[i], i);
}

void
acQueryRealparams(void)
{
    for (int i = 0; i < NUM_REAL_PARAMS; ++i)
        printf("%s (%d)\n", realparam_names[i], i);
}

void
acQueryReal3params(void)
{
    for (int i = 0; i < NUM_REAL3_PARAMS; ++i)
        printf("%s (%d)\n", real3param_names[i], i);
}


void
acQueryKernels(void)
{
    for (int i = 0; i < NUM_KERNELS; ++i)
        printf("%s (%d)\n", kernel_names[i], i);
}


void
acPrintIntParams(const AcIntParam a, const AcIntParam b, const AcIntParam c, const AcMeshInfo info)
{
    acPrintIntParam(a, info);
    acPrintIntParam(b, info);
    acPrintIntParam(c, info);
}

