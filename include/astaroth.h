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
#pragma once

#include "acc_runtime.h"
#define NGHOST (STENCIL_ORDER / 2) // Astaroth 2.0 backwards compatibility

typedef struct {
    AcReal* vertex_buffer[NUM_VTXBUF_HANDLES];
    AcMeshInfo info;
} AcMesh;

#define STREAM_0 (0)
#define STREAM_1 (1)
#define STREAM_2 (2)
#define STREAM_3 (3)
#define STREAM_4 (4)
#define STREAM_5 (5)
#define STREAM_6 (6)
#define STREAM_7 (7)
#define STREAM_8 (8)
#define STREAM_9 (9)
#define STREAM_10 (10)
#define STREAM_11 (11)
#define STREAM_12 (12)
#define STREAM_13 (13)
#define STREAM_14 (14)
#define STREAM_15 (15)
#define STREAM_16 (16)
#define STREAM_17 (17)
#define STREAM_18 (18)
#define STREAM_19 (19)
#define STREAM_20 (20)
#define STREAM_21 (21)
#define STREAM_22 (22)
#define STREAM_23 (23)
#define STREAM_24 (24)
#define STREAM_25 (25)
#define STREAM_26 (26)
#define STREAM_27 (27)
#define STREAM_28 (28)
#define STREAM_29 (29)
#define STREAM_30 (30)
#define STREAM_31 (31)
#define NUM_STREAMS (32)
#define STREAM_DEFAULT (STREAM_0)
#define STREAM_ALL (NUM_STREAMS)
typedef int Stream;

#define AC_FOR_RTYPES(FUNC)                                                                        \
    FUNC(RTYPE_MAX)                                                                                \
    FUNC(RTYPE_MIN)                                                                                \
    FUNC(RTYPE_RMS)                                                                                \
    FUNC(RTYPE_RMS_EXP)                                                                            \
    FUNC(RTYPE_ALFVEN_MAX)                                                                         \
    FUNC(RTYPE_ALFVEN_MIN)                                                                         \
    FUNC(RTYPE_ALFVEN_RMS)                                                                         \
    FUNC(RTYPE_SUM)

#define AC_FOR_BCTYPES(FUNC)                                                                       \
    FUNC(AC_BOUNDCOND_PERIODIC)                                                                    \
    FUNC(AC_BOUNDCOND_SYMMETRIC)                                                                   \
    FUNC(AC_BOUNDCOND_ENTROPY_1)                                                                   \
    FUNC(AC_BOUNDCOND_ANTISYMMETRIC)                                                               \
    FUNC(AC_BOUNDCOND_ADD_ONE)                                                                     \
    FUNC(AC_BOUNDCOND_ADD_TWO)                                                                     \
    FUNC(AC_BOUNDCOND_ADD_FOUR)                                                                     

#define AC_FOR_INIT_TYPES(FUNC)                                                                    \
    FUNC(INIT_TYPE_RANDOM)                                                                         \
    FUNC(INIT_TYPE_XWAVE)                                                                          \
    FUNC(INIT_TYPE_GAUSSIAN_RADIAL_EXPL)                                                           \
    FUNC(INIT_TYPE_ABC_FLOW)                                                                       \
    FUNC(INIT_TYPE_SIMPLE_CORE)                                                                    \
    FUNC(INIT_TYPE_KICKBALL)                                                                       \
    FUNC(INIT_TYPE_VEDGE)                                                                          \
    FUNC(INIT_TYPE_VEDGEX)                                                                         \
    FUNC(INIT_TYPE_RAYLEIGH_TAYLOR)                                                                \
    FUNC(INIT_TYPE_RAYLEIGH_BENARD)

#define AC_GEN_ID(X) X,
// Naming the associated number of the boundary condition types
typedef enum {
    AC_FOR_BCTYPES(AC_GEN_ID) //
    NUM_BCTYPES,
} AcBoundcond;

typedef enum {
    AC_FOR_RTYPES(AC_GEN_ID) //
    NUM_RTYPES
} ReductionType;

typedef enum {
    AC_FOR_INIT_TYPES(AC_GEN_ID) //
    NUM_INIT_TYPES
} InitType;

#undef AC_GEN_ID

#define _UNUSED __attribute__((unused)) // Does not give a warning if unused
#define AC_GEN_STR(X) #X,
static const char* bctype_names[] _UNUSED       = {AC_FOR_BCTYPES(AC_GEN_STR) "-end-"};
static const char* rtype_names[] _UNUSED        = {AC_FOR_RTYPES(AC_GEN_STR) "-end-"};
static const char* initcondtype_names[] _UNUSED = {AC_FOR_INIT_TYPES(AC_GEN_STR) "-end-"};
#undef AC_GEN_STR
#undef _UNUSED

typedef struct node_s* Node;
typedef struct device_s* Device;

typedef struct {
    int3 m;
    int3 n;
} GridDims;

typedef struct {
    int num_devices;
    Device* devices;

    GridDims grid;
    GridDims subgrid;
} DeviceConfiguration;

#ifdef __cplusplus
extern "C" {
#endif

/*
 * =============================================================================
 * Helper functions
 * =============================================================================
 */
static inline size_t
acVertexBufferSize(const AcMeshInfo info)
{
    return info.int_params[AC_mx] * info.int_params[AC_my] * info.int_params[AC_mz];
}

static inline size_t
acVertexBufferSizeBytes(const AcMeshInfo info)
{
    return sizeof(AcReal) * acVertexBufferSize(info);
}

static inline size_t
acVertexBufferCompdomainSize(const AcMeshInfo info)
{
    return info.int_params[AC_nx] * info.int_params[AC_ny] * info.int_params[AC_nz];
}

static inline size_t
acVertexBufferCompdomainSizeBytes(const AcMeshInfo info)
{
    return sizeof(AcReal) * acVertexBufferCompdomainSize(info);
}

static int3
acConstructInt3Param(const AcIntParam a, const AcIntParam b, const AcIntParam c,
                     const AcMeshInfo info)
{
    return (int3){
        info.int_params[a],
        info.int_params[b],
        info.int_params[c],
    };
}

static inline size_t
acVertexBufferIdx(const int i, const int j, const int k, const AcMeshInfo info)
{
    return i +                          //
           j * info.int_params[AC_mx] + //
           k * info.int_params[AC_mx] * info.int_params[AC_my];
}

static inline int3
acVertexBufferSpatialIdx(const size_t i, const AcMeshInfo info)
{
    const int3 mm = acConstructInt3Param(AC_mx, AC_my, AC_mz, info);

    return (int3){
        (int)i % mm.x,
        ((int)i % (mm.x * mm.y)) / mm.x,
        (int)i / (mm.x * mm.y),
    };
}

/** Prints all parameters inside AcMeshInfo */
static inline void
acPrintMeshInfo(const AcMeshInfo config)
{
    for (int i = 0; i < NUM_INT_PARAMS; ++i)
        printf("[%s]: %d\n", intparam_names[i], config.int_params[i]);
    for (int i = 0; i < NUM_INT3_PARAMS; ++i)
        printf("[%s]: (%d, %d, %d)\n", int3param_names[i], config.int3_params[i].x,
               config.int3_params[i].y, config.int3_params[i].z);
    for (int i = 0; i < NUM_REAL_PARAMS; ++i)
        printf("[%s]: %g\n", realparam_names[i], (double)(config.real_params[i]));
    for (int i = 0; i < NUM_REAL3_PARAMS; ++i)
        printf("[%s]: (%g, %g, %g)\n", real3param_names[i], (double)(config.real3_params[i].x),
               (double)(config.real3_params[i].y), (double)(config.real3_params[i].z));
}

/** Prints a list of boundary condition types */
static inline void
acQueryBCtypes(void)
{
    for (int i = 0; i < NUM_BCTYPES; ++i)
        printf("%s (%d)\n", bctype_names[i], i);
}

/** Prints a list of initial condition condition types */
static inline void
acQueryInitcondtypes(void)
{
    for (int i = 0; i < NUM_INIT_TYPES; ++i)
        printf("%s (%d)\n", initcondtype_names[i], i);
}

/** Prints a list of reduction types */
static inline void
acQueryRtypes(void)
{
    for (int i = 0; i < NUM_RTYPES; ++i)
        printf("%s (%d)\n", rtype_names[i], i);
}

/** Prints a list of int parameters */
static inline void
acQueryIntparams(void)
{
    for (int i = 0; i < NUM_INT_PARAMS; ++i)
        printf("%s (%d)\n", intparam_names[i], i);
}

/** Prints a list of int3 parameters */
static inline void
acQueryInt3params(void)
{
    for (int i = 0; i < NUM_INT3_PARAMS; ++i)
        printf("%s (%d)\n", int3param_names[i], i);
}

/** Prints a list of real parameters */
static inline void
acQueryRealparams(void)
{
    for (int i = 0; i < NUM_REAL_PARAMS; ++i)
        printf("%s (%d)\n", realparam_names[i], i);
}

/** Prints a list of real3 parameters */
static inline void
acQueryReal3params(void)
{
    for (int i = 0; i < NUM_REAL3_PARAMS; ++i)
        printf("%s (%d)\n", real3param_names[i], i);
}

/** Prints a list of Scalar array handles */
/*
static inline void
acQueryScalarrays(void)
{
    for (int i = 0; i < NUM_SCALARARRAY_HANDLES; ++i)
        printf("%s (%d)\n", scalarray_names[i], i);
}
*/

/** Prints a list of vertex buffer handles */
static inline void
acQueryVtxbufs(void)
{
    for (int i = 0; i < NUM_VTXBUF_HANDLES; ++i)
        printf("%s (%d)\n", vtxbuf_names[i], i);
}

/*
 * =============================================================================
 * Legacy interface
 * =============================================================================
 */
/** Allocates all memory and initializes the devices visible to the caller. Should be
 * called before any other function in this interface. */
AcResult acInit(const AcMeshInfo mesh_info);

/** Frees all GPU allocations and resets all devices in the node. Should be
 * called at exit. */
AcResult acQuit(void);

/** Checks whether there are any CUDA devices available. Returns AC_SUCCESS if there is 1 or more,
 * AC_FAILURE otherwise. */
AcResult acCheckDeviceAvailability(void);

/** Synchronizes a specific stream. All streams are synchronized if STREAM_ALL is passed as a
 * parameter*/
AcResult acSynchronizeStream(const Stream stream);

/** */
AcResult acSynchronizeMesh(void);

/** Loads a constant to the memories of the devices visible to the caller */
AcResult acLoadDeviceConstant(const AcRealParam param, const AcReal value);

/** Loads an AcMesh to the devices visible to the caller */
AcResult acLoad(const AcMesh host_mesh);

/** Sets the whole mesh to some value */
AcResult acSetVertexBuffer(const VertexBufferHandle handle, const AcReal value);

/** Stores the AcMesh distributed among the devices visible to the caller back to the host*/
AcResult acStore(AcMesh* host_mesh);

/** Performs Runge-Kutta 3 integration. Note: Boundary conditions are not applied after the final
 * substep and the user is responsible for calling acBoundcondStep before reading the data. */
AcResult acIntegrate(const AcReal dt);

/** Performs Runge-Kutta 3 integration. Note: Boundary conditions are not applied after the final
 * substep and the user is responsible for calling acBoundcondStep before reading the data.
 * Has customizable boundary conditions. */
AcResult acIntegrateGBC(const AcMeshInfo config, const AcReal dt);

/** Applies periodic boundary conditions for the Mesh distributed among the devices visible to
 * the caller*/
AcResult acBoundcondStep(void);

/** Applies general outer boundary conditions for the Mesh distributed among the devices visible to
 * the caller*/
AcResult acBoundcondStepGBC(const AcMeshInfo config);

/** Does a scalar reduction with the data stored in some vertex buffer */
AcReal acReduceScal(const ReductionType rtype, const VertexBufferHandle vtxbuf_handle);

/** Does a vector reduction with vertex buffers where the vector components are (a, b, c) */
AcReal acReduceVec(const ReductionType rtype, const VertexBufferHandle a,
                   const VertexBufferHandle b, const VertexBufferHandle c);

/** Does a reduction for an operation which requires a vector and a scalar with vertex buffers
 *  * where the vector components are (a, b, c) and scalr is (d) */
AcReal acReduceVecScal(const ReductionType rtype, const VertexBufferHandle a,
                       const VertexBufferHandle b, const VertexBufferHandle c,
                       const VertexBufferHandle d);

/** Stores a subset of the mesh stored across the devices visible to the caller back to host memory.
 */
AcResult acStoreWithOffset(const int3 dst, const size_t num_vertices, AcMesh* host_mesh);

/** Will potentially be deprecated in later versions. Added only to fix backwards compatibility with
 * PC for now.*/
AcResult acIntegrateStep(const int isubstep, const AcReal dt);
AcResult acIntegrateStepWithOffset(const int isubstep, const AcReal dt, const int3 start,
                                   const int3 end);
AcResult acSynchronize(void);
AcResult acLoadWithOffset(const AcMesh host_mesh, const int3 src, const int num_vertices);

/** */
int acGetNumDevicesPerNode(void);

/** */
Node acGetNode(void);

/*
 * =============================================================================
 * Grid interface
 * =============================================================================
 */
#if AC_MPI_ENABLED
/**
Initializes all available devices.

Must compile and run the code with MPI.

Must allocate exactly one process per GPU. And the same number of processes
per node as there are GPUs on that node.

Devices in the grid are configured based on the contents of AcMesh.
 */
AcResult acGridInit(const AcMeshInfo info);

/**
Resets all devices on the current grid.
 */
AcResult acGridQuit(void);

/** Randomizes the local mesh */
AcResult acGridRandomize(void);

/** */
AcResult acGridSynchronizeStream(const Stream stream);

/** */
AcResult acGridLoadScalarUniform(const Stream stream, const AcRealParam param, const AcReal value);

/** */
AcResult acGridLoadVectorUniform(const Stream stream, const AcReal3Param param,
                                 const AcReal3 value);

/** */
AcResult acGridLoadMesh(const Stream stream, const AcMesh host_mesh);

/** */
AcResult acGridStoreMesh(const Stream stream, AcMesh* host_mesh);

/** */
AcResult acGridIntegrate(const Stream stream, const AcReal dt);

/** */
/*   MV: Commented out for a while, but save for the future when standalone_MPI
         works with periodic boundary conditions.
AcResult
acGridIntegrateNonperiodic(const Stream stream, const AcReal dt)

AcResult acGridIntegrateNonperiodic(const Stream stream, const AcReal dt);
*/

/** */
AcResult acGridPeriodicBoundconds(const Stream stream);

/** */
AcResult acGridGeneralBoundconds(const Device device, const Stream stream);

/** */
AcResult acGridReduceScal(const Stream stream, const ReductionType rtype,
                          const VertexBufferHandle vtxbuf_handle, AcReal* result);

/** */
AcResult acGridReduceVec(const Stream stream, const ReductionType rtype,
                         const VertexBufferHandle vtxbuf0, const VertexBufferHandle vtxbuf1,
                         const VertexBufferHandle vtxbuf2, AcReal* result);

/** */
AcResult acGridReduceVecScal(const Stream stream, const ReductionType rtype,
                             const VertexBufferHandle vtxbuf0, const VertexBufferHandle vtxbuf1,
                             const VertexBufferHandle vtxbuf2, const VertexBufferHandle vtxbuf3,
                             AcReal* result);

/*
 * =============================================================================
 * Task interface (part of the grid interface)
 * =============================================================================
 */

/** */
typedef enum AcTaskType { TASKTYPE_COMPUTE, TASKTYPE_HALOEXCHANGE, TASKTYPE_BOUNDCOND } AcTaskType;

#define BIT(pos) (1U << (pos))

typedef enum AcBoundary {
    BOUNDARY_X_TOP = 0x01,
    BOUNDARY_X_BOT = 0x02,
    BOUNDARY_X     = BOUNDARY_X_TOP | BOUNDARY_X_BOT,
    BOUNDARY_Y_TOP = 0x04,
    BOUNDARY_Y_BOT = 0x08,
    BOUNDARY_Y     = BOUNDARY_Y_TOP | BOUNDARY_Y_BOT,
    BOUNDARY_Z_TOP = 0x10,
    BOUNDARY_Z_BOT = 0x20,
    BOUNDARY_Z     = BOUNDARY_Z_TOP | BOUNDARY_Z_BOT,
    BOUNDARY_XY    = BOUNDARY_X | BOUNDARY_Y,
    BOUNDARY_XZ    = BOUNDARY_X | BOUNDARY_Z,
    BOUNDARY_YZ    = BOUNDARY_Y | BOUNDARY_Z,
    BOUNDARY_XYZ   = BOUNDARY_X | BOUNDARY_Y | BOUNDARY_Z
} AcBoundary;

/** TaskDefinition is a datatype containing information necessary to generate a set of tasks for
 * some operation.*/
typedef struct AcTaskDefinition {
    AcTaskType task_type;
    union {
        AcKernel kernel;
        AcBoundcond bound_cond;
    };
    AcBoundary boundary;

    Field* fields_in;
    size_t num_fields_in;

    Field* fields_out;
    size_t num_fields_out;
} AcTaskDefinition;

/** TaskGraph is an opaque datatype containing information necessary to execute a set of
 * operation.*/
typedef struct AcTaskGraph AcTaskGraph;

/** */
AcTaskDefinition acCompute(const AcKernel kernel, VertexBufferHandle fields_in[],
                           const size_t num_fields_in, VertexBufferHandle fields_out[],
                           const size_t num_fields_out);

/** */
AcTaskDefinition acHaloExchange(VertexBufferHandle fields[], const size_t num_fields);

/** */
AcTaskDefinition acBoundaryCondition(const AcBoundary boundary, const AcBoundcond bound_cond,
                                     VertexBufferHandle fields_in[], const size_t num_fields_in,
                                     VertexBufferHandle fields_out[], const size_t num_fields_out);

/** */
AcTaskGraph* acGridGetDefaultTaskGraph();

/** */
AcTaskGraph* acGridBuildTaskGraph(const AcTaskDefinition ops[], const size_t n_ops);

/** */
AcResult acGridDestroyTaskGraph(AcTaskGraph* graph);

/** */
AcResult acGridExecuteTaskGraph(const AcTaskGraph* graph, const size_t n_iterations);

#endif // AC_MPI_ENABLED

/*
 * =============================================================================
 * Node interface
 * =============================================================================
 */
/**
Initializes all devices on the current node.

Devices on the node are configured based on the contents of AcMesh.

@return Exit status. Places the newly created handle in the output parameter.
@see AcMeshInfo


Usage example:
@code
AcMeshInfo info;
acLoadConfig(AC_DEFAULT_CONFIG, &info);

Node node;
acNodeCreate(0, info, &node);
acNodeDestroy(node);
@endcode
 */
AcResult acNodeCreate(const int id, const AcMeshInfo node_config, Node* node);

/**
Resets all devices on the current node.

@see acNodeCreate()
 */
AcResult acNodeDestroy(Node node);

/**
Prints information about the devices available on the current node.

Requires that Node has been initialized with
@See acNodeCreate().
*/
AcResult acNodePrintInfo(const Node node);

/**



@see DeviceConfiguration
*/
AcResult acNodeQueryDeviceConfiguration(const Node node, DeviceConfiguration* config);

/** */
AcResult acNodeAutoOptimize(const Node node);

/** */
AcResult acNodeSynchronizeStream(const Node node, const Stream stream);

/** Deprecated ? */
AcResult acNodeSynchronizeVertexBuffer(const Node node, const Stream stream,
                                       const VertexBufferHandle vtxbuf_handle); // Not in Device

/** */
AcResult acNodeSynchronizeMesh(const Node node, const Stream stream); // Not in Device

/** */
AcResult acNodeSwapBuffers(const Node node);

/** */
AcResult acNodeLoadConstant(const Node node, const Stream stream, const AcRealParam param,
                            const AcReal value);

/** Deprecated ? Might be useful though if the user wants to load only one vtxbuf. But in this case
 * the user should supply a AcReal* instead of vtxbuf_handle */
AcResult acNodeLoadVertexBufferWithOffset(const Node node, const Stream stream,
                                          const AcMesh host_mesh,
                                          const VertexBufferHandle vtxbuf_handle, const int3 src,
                                          const int3 dst, const int num_vertices);

/** */
AcResult acNodeLoadMeshWithOffset(const Node node, const Stream stream, const AcMesh host_mesh,
                                  const int3 src, const int3 dst, const int num_vertices);

/** Deprecated ? */
AcResult acNodeLoadVertexBuffer(const Node node, const Stream stream, const AcMesh host_mesh,
                                const VertexBufferHandle vtxbuf_handle);

/** */
AcResult acNodeLoadMesh(const Node node, const Stream stream, const AcMesh host_mesh);

/** */
AcResult acNodeSetVertexBuffer(const Node node, const Stream stream,
                               const VertexBufferHandle handle, const AcReal value);

/** Deprecated ? */
AcResult acNodeStoreVertexBufferWithOffset(const Node node, const Stream stream,
                                           const VertexBufferHandle vtxbuf_handle, const int3 src,
                                           const int3 dst, const int num_vertices,
                                           AcMesh* host_mesh);

/** */
AcResult acNodeStoreMeshWithOffset(const Node node, const Stream stream, const int3 src,
                                   const int3 dst, const int num_vertices, AcMesh* host_mesh);

/** Deprecated ? */
AcResult acNodeStoreVertexBuffer(const Node node, const Stream stream,
                                 const VertexBufferHandle vtxbuf_handle, AcMesh* host_mesh);

/** */
AcResult acNodeStoreMesh(const Node node, const Stream stream, AcMesh* host_mesh);

/** */
AcResult acNodeIntegrateSubstep(const Node node, const Stream stream, const int step_number,
                                const int3 start, const int3 end, const AcReal dt);

/** */
AcResult acNodeIntegrate(const Node node, const AcReal dt);

/** */
AcResult acNodeIntegrateGBC(const Node node, const AcMeshInfo config, const AcReal dt);

/** */
AcResult acNodePeriodicBoundcondStep(const Node node, const Stream stream,
                                     const VertexBufferHandle vtxbuf_handle);

/** */
AcResult acNodePeriodicBoundconds(const Node node, const Stream stream);

/** */
AcResult acNodeGeneralBoundcondStep(const Node node, const Stream stream,
                                    const VertexBufferHandle vtxbuf_handle,
                                    const AcMeshInfo config);

/** */
AcResult acNodeGeneralBoundconds(const Node node, const Stream stream, const AcMeshInfo config);

/** */
AcResult acNodeReduceScal(const Node node, const Stream stream, const ReductionType rtype,
                          const VertexBufferHandle vtxbuf_handle, AcReal* result);
/** */
AcResult acNodeReduceVec(const Node node, const Stream stream_type, const ReductionType rtype,
                         const VertexBufferHandle vtxbuf0, const VertexBufferHandle vtxbuf1,
                         const VertexBufferHandle vtxbuf2, AcReal* result);
/** */
AcResult acNodeReduceVecScal(const Node node, const Stream stream_type, const ReductionType rtype,
                             const VertexBufferHandle vtxbuf0, const VertexBufferHandle vtxbuf1,
                             const VertexBufferHandle vtxbuf2, const VertexBufferHandle vtxbuf3,
                             AcReal* result);

/*
 * =============================================================================
 * Device interface
 * =============================================================================
 */
/** */
AcResult acDeviceCreate(const int id, const AcMeshInfo device_config, Device* device);

/** */
AcResult acDeviceDestroy(Device device);

/** */
AcResult acDevicePrintInfo(const Device device);

/** */
// AcResult acDeviceAutoOptimize(const Device device);

/** */
AcResult acDeviceSynchronizeStream(const Device device, const Stream stream);

/** */
AcResult acDeviceSwapBuffer(const Device device, const VertexBufferHandle handle);

/** */
AcResult acDeviceSwapBuffers(const Device device);

/** */
AcResult acDeviceLoadScalarUniform(const Device device, const Stream stream,
                                   const AcRealParam param, const AcReal value);

/** */
AcResult acDeviceLoadVectorUniform(const Device device, const Stream stream,
                                   const AcReal3Param param, const AcReal3 value);

/** */
AcResult acDeviceLoadIntUniform(const Device device, const Stream stream, const AcIntParam param,
                                const int value);

/** */
AcResult acDeviceLoadInt3Uniform(const Device device, const Stream stream, const AcInt3Param param,
                                 const int3 value);

/** */
/*
AcResult acDeviceLoadScalarArray(const Device device, const Stream stream,
                                 const ScalarArrayHandle handle, const size_t start,
                                 const AcReal* data, const size_t num);
                                 */

/** */
AcResult acDeviceLoadMeshInfo(const Device device, const AcMeshInfo device_config);

/** */
AcResult acDeviceLoadDefaultUniforms(const Device device);

/** */
AcResult acDeviceLoadVertexBufferWithOffset(const Device device, const Stream stream,
                                            const AcMesh host_mesh,
                                            const VertexBufferHandle vtxbuf_handle, const int3 src,
                                            const int3 dst, const int num_vertices);

/** Deprecated */
AcResult acDeviceLoadMeshWithOffset(const Device device, const Stream stream,
                                    const AcMesh host_mesh, const int3 src, const int3 dst,
                                    const int num_vertices);

/** */
AcResult acDeviceLoadVertexBuffer(const Device device, const Stream stream, const AcMesh host_mesh,
                                  const VertexBufferHandle vtxbuf_handle);

/** */
AcResult acDeviceLoadMesh(const Device device, const Stream stream, const AcMesh host_mesh);

/** */
AcResult acDeviceSetVertexBuffer(const Device device, const Stream stream,
                                 const VertexBufferHandle handle, const AcReal value);

/** */
AcResult acDeviceStoreVertexBufferWithOffset(const Device device, const Stream stream,
                                             const VertexBufferHandle vtxbuf_handle, const int3 src,
                                             const int3 dst, const int num_vertices,
                                             AcMesh* host_mesh);

/** Deprecated */
AcResult acDeviceStoreMeshWithOffset(const Device device, const Stream stream, const int3 src,
                                     const int3 dst, const int num_vertices, AcMesh* host_mesh);

/** */
AcResult acDeviceStoreVertexBuffer(const Device device, const Stream stream,
                                   const VertexBufferHandle vtxbuf_handle, AcMesh* host_mesh);

/** */
AcResult acDeviceStoreMesh(const Device device, const Stream stream, AcMesh* host_mesh);

/** */
AcResult acDeviceTransferVertexBufferWithOffset(const Device src_device, const Stream stream,
                                                const VertexBufferHandle vtxbuf_handle,
                                                const int3 src, const int3 dst,
                                                const int num_vertices, Device dst_device);

/** Deprecated */
AcResult acDeviceTransferMeshWithOffset(const Device src_device, const Stream stream,
                                        const int3 src, const int3 dst, const int num_vertices,
                                        Device* dst_device);

/** */
AcResult acDeviceTransferVertexBuffer(const Device src_device, const Stream stream,
                                      const VertexBufferHandle vtxbuf_handle, Device dst_device);

/** */
AcResult acDeviceTransferMesh(const Device src_device, const Stream stream, Device dst_device);

/** */
AcResult acDeviceIntegrateSubstep(const Device device, const Stream stream, const int step_number,
                                  const int3 start, const int3 end, const AcReal dt);
/** */
AcResult acDevicePeriodicBoundcondStep(const Device device, const Stream stream,
                                       const VertexBufferHandle vtxbuf_handle, const int3 start,
                                       const int3 end);

/** */
AcResult acDevicePeriodicBoundconds(const Device device, const Stream stream, const int3 start,
                                    const int3 end);

/** */
AcResult acDeviceGeneralBoundcondStep(const Device device, const Stream stream,
                                      const VertexBufferHandle vtxbuf_handle, const int3 start,
                                      const int3 end, const AcMeshInfo config, const int3 bindex);

/** */
AcResult acDeviceGeneralBoundconds(const Device device, const Stream stream, const int3 start,
                                   const int3 end, const AcMeshInfo config, const int3 bindex);

/** */
AcResult acDeviceReduceScal(const Device device, const Stream stream, const ReductionType rtype,
                            const VertexBufferHandle vtxbuf_handle, AcReal* result);
/** */
AcResult acDeviceReduceVec(const Device device, const Stream stream_type, const ReductionType rtype,
                           const VertexBufferHandle vtxbuf0, const VertexBufferHandle vtxbuf1,
                           const VertexBufferHandle vtxbuf2, AcReal* result);
/** */
AcResult acDeviceReduceVecScal(const Device device, const Stream stream_type,
                               const ReductionType rtype, const VertexBufferHandle vtxbuf0,
                               const VertexBufferHandle vtxbuf1, const VertexBufferHandle vtxbuf2,
                               const VertexBufferHandle vtxbuf3, AcReal* result);
/** */
AcResult acDeviceRunMPITest(void);

/** */
AcResult
acDeviceLoadStencils(const Device device, const Stream stream,
                     AcReal stencil[NUM_STENCILS][STENCIL_DEPTH][STENCIL_HEIGHT][STENCIL_WIDTH]);

AcResult acDeviceLaunchKernel(const Device device, const Stream stream, const Kernel kernel,
                              const int3 start, const int3 end);

/*
 * =============================================================================
 * Helper functions
 * =============================================================================
 */
/** Updates the built-in parameters based on nx, ny and nz */
AcResult acHostUpdateBuiltinParams(AcMeshInfo* config);

/** Creates a mesh stored in host memory */
AcResult acHostMeshCreate(const AcMeshInfo mesh_info, AcMesh* mesh);

/** Randomizes a host mesh */
AcResult acHostMeshRandomize(AcMesh* mesh);

/** Destroys a mesh stored in host memory */
AcResult acHostMeshDestroy(AcMesh* mesh);

#ifdef __cplusplus
} // extern "C"
#endif

#ifdef __cplusplus
#if AC_MPI_ENABLED
/** Backwards compatible interface, input fields = output fields*/
template <size_t num_fields>
AcTaskDefinition
acCompute(AcKernel kernel, VertexBufferHandle (&fields)[num_fields])
{
    return acCompute(kernel, fields, num_fields, fields, num_fields);
}

template <size_t num_fields_in, size_t num_fields_out>
AcTaskDefinition
acCompute(AcKernel kernel, VertexBufferHandle (&fields_in)[num_fields_in],
          VertexBufferHandle (&fields_out)[num_fields_out])
{
    return acCompute(kernel, fields_in, num_fields_in, fields_out, num_fields_out);
}

/** */
template <size_t num_fields>
AcTaskDefinition acHaloExchange(VertexBufferHandle (&fields)[num_fields])
{
    return acHaloExchange(fields, num_fields);
}

/** Backwards compatible interface, input fields = output fields*/
template <size_t num_fields>
AcTaskDefinition
acBoundaryCondition(const AcBoundary boundary, const AcBoundcond bound_cond,
                    VertexBufferHandle (&fields)[num_fields])
{
    return acBoundaryCondition(boundary, bound_cond, fields, num_fields,
                               fields, num_fields);
}


/** */
template <size_t num_fields_in, size_t num_fields_out>
AcTaskDefinition
acBoundaryCondition(const AcBoundary boundary, const AcBoundcond bound_cond,
                    VertexBufferHandle (&fields_in)[num_fields_in],
                    VertexBufferHandle (&fields_out)[num_fields_out])
{
    return acBoundaryCondition(boundary, bound_cond, fields_in, num_fields_in,
                               fields_out, num_fields_out);
}

/** */
template <size_t n_ops>
AcTaskGraph*
acGridBuildTaskGraph(const AcTaskDefinition (&ops)[n_ops])
{
    return acGridBuildTaskGraph(ops, n_ops);
}
#endif
#endif
