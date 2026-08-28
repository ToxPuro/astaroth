/*
    Copyright (C) 2014-2026, Johannes Pekkila, Miikka Vaisala, Touko Puro, Ondřej Míchal.

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

#include "astaroth_legacy.h"

#include "astaroth_node.h"

static const int max_num_nodes   = 1;
static Node nodes[max_num_nodes] = {0};
static int num_nodes             = 0;

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

AcResult
acCheckDeviceAvailability(void)
{
    int runtime_version, max_runtime_version_supported_by_driver;
    ERRCHK_CUDA_ALWAYS(acDriverGetVersion(&max_runtime_version_supported_by_driver));
    ERRCHK_CUDA_ALWAYS(acRuntimeGetVersion(&runtime_version));
    if(runtime_version > max_runtime_version_supported_by_driver)
    {
            fprintf(stderr,"AC error!: Reported maximum supported runtime by the driver was %d but used runtime is %d!!\n",max_runtime_version_supported_by_driver,runtime_version);
            fprintf(stderr,"AC error!: Reported maximum supported runtime by the driver was %d but used runtime is %d!!\n",max_runtime_version_supported_by_driver,runtime_version);
            fprintf(stderr,"AC error!: Reported maximum supported runtime by the driver was %d but used runtime is %d!!\n",max_runtime_version_supported_by_driver,runtime_version);
            ERRCHK_ALWAYS(runtime_version <= max_runtime_version_supported_by_driver);
    }
    int device_count; // Separate from num_devices to avoid side effects
    ERRCHK_CUDA_ALWAYS(acGetDeviceCount(&device_count));
    if (device_count > 0)
        return AC_SUCCESS;
    else
        return AC_FAILURE;
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

const char*
acGetFieldName(const Field field)
{
	return field_names[field];
}

Node
acGetNode(void)
{
    ERRCHK_ALWAYS(num_nodes > 0);
    return nodes[0];
}

int
acGetNumDevicesPerNode(void)
{
    int num_devices;
    ERRCHK_CUDA_ALWAYS(acGetDeviceCount(&num_devices));
    return num_devices;
}

size_t
acGetNumFields(void)
{
    return NUM_VTXBUF_HANDLES;
}

AcResult
acInit(const AcMeshInfo mesh_info)
{
    num_nodes = 1;
    return acNodeCreate(0, mesh_info, &nodes[0]);
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
acLoad(const AcMesh host_mesh)
{
    ERRCHK_ALWAYS(num_nodes);
    return acNodeLoadMesh(nodes[0], STREAM_DEFAULT, host_mesh);
}

AcResult
acLoadDeviceConstant(const AcRealParam param, const AcReal value)
{
    ERRCHK_ALWAYS(num_nodes);
    return acNodeLoadConstant(nodes[0], STREAM_DEFAULT, param, value);
}

AcResult
acLoadWithOffset(const AcMesh host_mesh, const int3 src, const int num_vertices)
{
    ERRCHK_ALWAYS(num_nodes);
    return acNodeLoadMeshWithOffset(nodes[0], STREAM_DEFAULT, host_mesh, src, src, num_vertices);
}

AcResult
acQuit(void)
{
    ERRCHK_ALWAYS(num_nodes);
    num_nodes = 0;
    return acNodeDestroy(nodes[0]);
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
acStoreWithOffset(const int3 dst, const size_t num_vertices, AcMesh* host_mesh)
{
    ERRCHK_ALWAYS(num_nodes);
    return acNodeStoreMeshWithOffset(nodes[0], STREAM_DEFAULT, dst, dst, num_vertices, host_mesh);
}

AcResult
acSynchronize(void)
{
    ERRCHK_ALWAYS(num_nodes);
    return acNodeSynchronizeStream(nodes[0], STREAM_ALL);
}

AcResult
acSynchronizeMesh(void)
{
    ERRCHK_ALWAYS(num_nodes);
    return acNodeSynchronizeMesh(nodes[0], STREAM_DEFAULT);
}

AcResult
acSynchronizeStream(const Stream stream)
{
    ERRCHK_ALWAYS(num_nodes);
    return acNodeSynchronizeStream(nodes[0], stream);
}
