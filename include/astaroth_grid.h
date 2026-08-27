#pragma once

#include "astaroth_device.h"
#include "astaroth_helpers.h"
#include "func_define.h"

#ifndef UNUSED
#define UNUSED __attribute__((unused)) // Does not give a warning if unused
#endif

#if AC_MPI_ENABLED

AC_BEGIN_C_DECLARATIONS

/**
Calls MPI_Init and creates a separate communicator for Astaroth procs with MPI_Comm_split, color =
666 Any program running in the same MPI process space must also call MPI_Comm_split with some color
!= 666. OTHERWISE this call will hang.

Returns AC_SUCCESS on successfullly initializing MPI and creating a communicator.

Returns AC_FAILURE otherwise.
 */
FUNC_DEFINE(AcResult, ac_MPI_Init,(void));

/**
Calls MPI_Init_thread with the provided thread_level and creates a separate communicator for
Astaroth procs with MPI_Comm_split, color = 666 Any program running in the same MPI process space
must also call MPI_Comm_split with some color != 666. OTHERWISE this call will hang.

Returns AC_SUCCESS on successfullly initializing MPI with the requested thread level and creating a
communicator.

Returns AC_FAILURE otherwise.
 */
FUNC_DEFINE(AcResult, ac_MPI_Init_thread,(int thread_level));

/**
Destroys the communicator and calls MPI_Finalize
*/
FUNC_DEFINE(void, ac_MPI_Finalize,());

/** Returns the rank of the Astaroth communicator */
FUNC_DEFINE(int, ac_MPI_Comm_rank,());

/** If MPI was initialized with MPI_Init* instead of ac_MPI_Init, this will return MPI_COMM_WORLD */
FUNC_DEFINE(MPI_Comm, acGridMPIComm,());

FUNC_DEFINE(AcSubCommunicators,acGridMPISubComms,());
/** Returns the size of the Astaroth communicator */
FUNC_DEFINE(int, ac_MPI_Comm_size,());

/** Calls MPI_Barrier on the Astaroth communicator */
FUNC_DEFINE(void, ac_MPI_Barrier,());

/**
Initializes all available devices.

Must compile and run the code with MPI.

Must allocate exactly one process per GPU. And the same number of processes
per node as there are GPUs on that node.

Devices in the grid are configured based on the contents of AcMesh.
 */

FUNC_DEFINE(AcResult, acGridInitBase, (const AcMesh mesh));
static AcResult UNUSED 
acGridInit(const AcMeshInfo info)
{
	AcMesh mesh;
	for(int i = 0; i < NUM_ALL_FIELDS; ++i) mesh.vertex_buffer[i] = NULL;
	mesh.info = info;
	return acGridInitBase(mesh);
}

/**
Resets all devices on the current grid.
 */
FUNC_DEFINE(AcResult, acGridQuit,(void));

/** Get the local device */
FUNC_DEFINE(Device, acGridGetDevice,(void));

/** Randomizes the local mesh */
FUNC_DEFINE(AcResult, acGridRandomize,(void));

FUNC_DEFINE(AcResult, acGridSynchronizeStream,(const Stream stream));

FUNC_DEFINE(AcResult, acGridLoadScalarUniform,(const Stream stream, const AcRealParam param, const AcReal value));

FUNC_DEFINE(AcResult, acGridLoadVectorUniform,(const Stream stream, const AcReal3Param param,
                                 const AcReal3 value));

FUNC_DEFINE(AcResult, acGridLoadIntUniform,(const Stream stream, const AcIntParam param, const int value));

FUNC_DEFINE(AcResult, acGridLoadInt3Uniform,(const Stream stream, const AcInt3Param param, const int3 value));

FUNC_DEFINE(AcResult, acGridLoadMesh,(const Stream stream, const AcMesh host_mesh));

FUNC_DEFINE(AcResult, acGridStoreMesh,(const Stream stream, AcMesh* host_mesh));

FUNC_DEFINE(AcResult, acGridIntegrate,(const Stream stream, const AcReal dt));

FUNC_DEFINE(AcResult, acGridSwapBuffers,(void));

/*   MV: Commented out for a while, but save for the future when standalone_MPI
         works with periodic boundary conditions.
AcResult
acGridIntegrateNonperiodic(const Stream stream, const AcReal dt)

AcResult acGridIntegrateNonperiodic(const Stream stream, const AcReal dt);
*/

FUNC_DEFINE(AcResult, acGridHaloExchange,());

FUNC_DEFINE(AcResult, acGridPeriodicBoundconds,(const Stream stream));


FUNC_DEFINE(AcResult, acGridReduceScal,(const Stream stream, const AcReduction reduction,
                          const VertexBufferHandle vtxbuf_handle, AcReal* result));

FUNC_DEFINE(AcResult, acGridReduceVec,(const Stream stream, const AcReduction reduction,
                         const VertexBufferHandle vtxbuf0, const VertexBufferHandle vtxbuf1,
                         const VertexBufferHandle vtxbuf2, AcReal* result));

FUNC_DEFINE(AcResult, acGridReduceVecScal,(const Stream stream, const AcReduction reduction,
                             const VertexBufferHandle vtxbuf0, const VertexBufferHandle vtxbuf1,
                             const VertexBufferHandle vtxbuf2, const VertexBufferHandle vtxbuf3,
                             AcReal* result));

FUNC_DEFINE(AcResult, acGridReduceXY, (const Stream stream, const Field field, const Profile profile, const AcReduction reduction));

typedef enum {
    ACCESS_READ,
    ACCESS_WRITE,
} AccessType;

FUNC_DEFINE(AcResult, acGridAccessMeshOnDiskSynchronous,(const VertexBufferHandle field, const char* dir,
                                           const char* label, const AccessType type));

FUNC_DEFINE(AcResult, acGridDiskAccessLaunch,(const AccessType type));

/* Asynchronous. Need to call acGridDiskAccessSync afterwards */
FUNC_DEFINE(AcResult, acGridWriteSlicesToDiskLaunch,(const char* dir, const int step_number, const AcReal simulation_time));
/* Synchronous. No need to call acGridDiskAccessSync afterwards */
FUNC_DEFINE(AcResult, acGridWriteSlicesToDiskSynchronous,(const char* dir, const int step_number, const AcReal simulation_time));

/* Synchronous */
FUNC_DEFINE(AcResult, acGridWriteSlicesToDiskCollectiveSynchronous,(const char* dir, const int step_number, const AcReal simulation_time));

/* Asynchronous. Need to call acGridDiskAccessSync afterwards */
FUNC_DEFINE(AcResult, acGridWriteMeshToDiskLaunch,(const char* dir, const char* label));

FUNC_DEFINE(AcResult, acGridDiskAccessSync,(void));

FUNC_DEFINE(AcResult, acGridReadVarfileToMesh,(const char* file, const Field fields[], const size_t num_fields,
                                 const int3 nn, const int3 rr));

/* Quick hack for the hero run, will be removed in future builds */
FUNC_DEFINE(AcResult, acGridAccessMeshOnDiskSynchronousDistributed,(const VertexBufferHandle vtxbuf,
                                                      const char* dir, const char* label,
                                                      const AccessType type));

/* Quick hack for the hero run, will be removed in future builds */
FUNC_DEFINE(AcResult, acGridAccessMeshOnDiskSynchronousCollective,(const VertexBufferHandle vtxbuf,
                                                     const char* dir, const char* label,
                                                     const AccessType type));

// Bugged
// AcResult acGridLoadFieldFromFile(const char* path, const VertexBufferHandle field);

// Bugged
// AcResult acGridStoreFieldToFile(const char* path, const VertexBufferHandle field);

typedef enum AcTaskType {
    TASKTYPE_NO_OP,
    TASKTYPE_COMPUTE,
    TASKTYPE_HALOEXCHANGE,
    TASKTYPE_BOUNDCOND,
    TASKTYPE_REDUCE,
    TASKTYPE_SCAN,
    TASKTYPE_RAY_UPDATE,
    TASKTYPE_PERIODIC_RAY,
} AcTaskType;



static UNUSED const char*
ac_boundary_to_str(const AcBoundary boundary)
{
	if(boundary == BOUNDARY_NONE)  return  "BOUNDARY_NONE";

	if(boundary == BOUNDARY_X_BOT) return "BOUNDARY_X_BOT";
	if(boundary == BOUNDARY_X_TOP) return "BOUNDARY_X_TOP";
	if(boundary == BOUNDARY_X)     return "BOUNDARY_X";

	if(boundary == BOUNDARY_Y_BOT) return "BOUNDARY_Y_BOT";
	if(boundary == BOUNDARY_Y_TOP) return "BOUNDARY_Y_TOP";
	if(boundary == BOUNDARY_Y)     return "BOUNDARY_Y";

	if(boundary == BOUNDARY_Z_BOT) return "BOUNDARY_Z_BOT";
	if(boundary == BOUNDARY_Z_TOP) return "BOUNDARY_Z_TOP";
	if(boundary == BOUNDARY_Z)     return "BOUNDARY_Z";

	if(boundary == BOUNDARY_XY)    return "BOUNDARY_XY";
	if(boundary == BOUNDARY_XZ)    return "BOUNDARY_XZ";
	if(boundary == BOUNDARY_YZ)    return "BOUNDARY_YZ";
	if(boundary == BOUNDARY_XYZ)   return "BOUNDARY_XYZ";

        if(boundary == BOUNDARY_X_BOT_Y_BOT_Z_BOT) return "BOUNDARY_X_BOT_Y_BOT_Z_BOT";
        if(boundary == BOUNDARY_X_BOT_Y_BOT_Z_TOP) return "BOUNDARY_X_BOT_Y_BOT_Z_TOP";
        if(boundary == BOUNDARY_X_BOT_Y_TOP_Z_BOT) return "BOUNDARY_X_BOT_Y_TOP_Z_BOT";
        if(boundary == BOUNDARY_X_BOT_Y_TOP_Z_TOP) return "BOUNDARY_X_BOT_Y_TOP_Z_TOP";
        if(boundary == BOUNDARY_X_TOP_Y_BOT_Z_BOT) return "BOUNDARY_X_TOP_Y_BOT_Z_BOT";
        if(boundary == BOUNDARY_X_TOP_Y_BOT_Z_TOP) return "BOUNDARY_X_TOP_Y_BOT_Z_TOP";
        if(boundary == BOUNDARY_X_TOP_Y_TOP_Z_BOT) return "BOUNDARY_X_TOP_Y_TOP_Z_BOT";
        if(boundary == BOUNDARY_X_TOP_Y_TOP_Z_TOP) return "BOUNDARY_X_TOP_Y_TOP_Z_TOP";

        if(boundary == BOUNDARY_XY_Z_BOT) return "BOUNDARY_XY_Z_BOT"; 
        if(boundary == BOUNDARY_XY_Z_TOP) return "BOUNDARY_XY_Z_TOP";
        if(boundary == BOUNDARY_XZ_Y_BOT) return "BOUNDARY_XZ_Y_BOT";
        if(boundary == BOUNDARY_XZ_Y_TOP) return "BOUNDARY_XZ_Y_TOP";
        if(boundary == BOUNDARY_YZ_X_BOT) return "BOUNDARY_YZ_X_BOT";
        if(boundary == BOUNDARY_YZ_X_TOP) return "BOUNDARY_YZ_X_TOP";

        if(boundary == BOUNDARY_X_Z_BOT) return "BOUNDARY_X_Z_BOT";
        if(boundary == BOUNDARY_X_Z_TOP) return "BOUNDARY_X_Z_TOP";
        if(boundary == BOUNDARY_X_Y_BOT) return "BOUNDARY_X_Y_BOT";
        if(boundary == BOUNDARY_X_Y_TOP) return "BOUNDARY_X_Y_TOP";

        if(boundary == BOUNDARY_Y_X_BOT) return "BOUNDARY_Y_X_BOT";
        if(boundary == BOUNDARY_Y_X_TOP) return "BOUNDARY_Y_X_TOP";
        if(boundary == BOUNDARY_Y_Z_BOT) return "BOUNDARY_Y_Z_BOT";
        if(boundary == BOUNDARY_Y_Z_TOP) return "BOUNDARY_Y_Z_TOP";

        if(boundary == BOUNDARY_Z_X_BOT) return "BOUNDARY_Z_X_BOT";
        if(boundary == BOUNDARY_Z_X_TOP) return "BOUNDARY_Z_X_TOP";
        if(boundary == BOUNDARY_Z_Y_BOT) return "BOUNDARY_Z_Y_BOT";
        if(boundary == BOUNDARY_Z_Y_TOP) return "BOUNDARY_Z_Y_TOP";
	return "UNKNOWN_BOUNDARY";
}




FUNC_DEFINE(acAnalysisBCInfo, acAnalysisGetBCInfo,(const AcMeshInfo info, const AcKernel bc, const AcBoundary boundary));

typedef struct ParamLoadingInfo {
        acKernelInputParams* params;
        Device device;
        const int step_number;
        const int3 boundary_normal;
        const Field vtxbuf;
	const AcKernel kernel;
} ParamLoadingInfo;

//opaque for C to enable C++ lambdas
typedef struct LoadKernelParamsFunc LoadKernelParamsFunc;


typedef struct 
{
	int min,max;
} facet_class_range;
/** TaskDefinition is a datatype containing information necessary to generate a set of tasks for
 * some operation.*/
typedef struct AcTaskDefinition {
    AcTaskType task_type;
    AcKernel kernel_enum;
    AcBoundary boundary;

    Field* fields_in;
    size_t num_fields_in;

    Field* fields_out;
    size_t num_fields_out;

    Profile* profiles_in;
    size_t num_profiles_in;

    Profile* profiles_reduce_out;
    size_t num_profiles_reduce_out;

    Profile* profiles_write_out;
    size_t num_profiles_write_out;

    AcRealParam* parameters;
    size_t num_parameters;
    LoadKernelParamsFunc* load_kernel_params_func;
    bool fieldwise;

    KernelReduceOutput* outputs_in;
    size_t num_outputs_in;

    KernelReduceOutput* outputs_out;
    size_t num_outputs_out;
    AcBoundary computes_on_halos;
    Volume start;
    Volume end;
    bool given_launch_bounds;
    Volume halo_sizes;
    bool sending;
    bool receiving;
    int3 ray_direction;
    bool include_boundaries;
    facet_class_range* halo_types;
    int3 id;
    KernelAnalysisInfo analysis_info;
    AcBoundary communicates_boundary;
    AcRedBlackState red_black_state;
} AcTaskDefinition;

/** TaskGraph is an opaque datatype containing information necessary to execute a set of
 * operations.*/
typedef struct AcTaskGraph AcTaskGraph;

OVERLOADED_FUNC_DEFINE(AcTaskDefinition, acCompute,(const AcKernel kernel, Field fields_in[], const size_t num_fields_in,
                           Field fields_out[], const size_t num_fields_out,Profile profiles_in[], const size_t num_profiles_in, Profile profiles_out[], const size_t num_profiles_out));

OVERLOADED_FUNC_DEFINE(AcTaskDefinition, acHaloExchange,(Field fields[], const size_t num_fields));
FUNC_DEFINE(AcTaskDefinition, acHaloExchangeBoundary,(Field fields[], const size_t num_fields, const AcBoundary boundary));

OVERLOADED_FUNC_DEFINE(AcTaskDefinition, acScan,(Field fields[], const size_t num_fields, const int3 direction));

FUNC_DEFINE(AcTaskDefinition, acPeriodicRay,(Field fields[], const size_t num_fields, const int3 direction));

FUNC_DEFINE(AcTaskDefinition,acHaloExchangeWithBounds,(Field fields[], const size_t num_fields, const Volume start, const Volume end, const int3 ray_direction, const bool sending, const bool receiving, const AcBoundary boundary, const bool include_boundaries,const facet_class_range halo_types[], const Volume halo_size));

FUNC_DEFINE(AcTaskGraph*, acGridGetDefaultTaskGraph,());

FUNC_DEFINE(bool, acGridTaskGraphIsEmpty,(AcTaskGraph* graph));

FUNC_DEFINE(bool, acGridTaskGraphHasPeriodicBoundcondsX,(AcTaskGraph* graph));

FUNC_DEFINE(bool, acGridTaskGraphHasPeriodicBoundcondsY,(AcTaskGraph* graph));

FUNC_DEFINE(bool, acGridTaskGraphHasPeriodicBoundcondsZ,(AcTaskGraph* graph));

OVERLOADED_FUNC_DEFINE(AcTaskGraph*, acGridBuildTaskGraph,(const AcTaskDefinition ops[], const size_t n_ops));
OVERLOADED_FUNC_DEFINE(AcTaskGraph*, acGridBuildTaskGraphWithBounds,(const AcTaskDefinition ops[], const size_t n_ops, const Volume start, const Volume end, const bool globally_imposed_bcs));

FUNC_DEFINE(AcTaskGraph*, acGetDSLTaskGraph,(const AcDSLTaskGraph));
FUNC_DEFINE(AcTaskGraph*, acGetDSLTaskGraphWithBounds,(const AcDSLTaskGraph, const Volume start, const Volume end));
OVERLOADED_FUNC_DEFINE(AcTaskGraph*, acGetOptimizedDSLTaskGraph,(const AcDSLTaskGraph));
FUNC_DEFINE(AcTaskGraph*, acGetOptimizedDSLTaskGraphWithBounds,(const AcDSLTaskGraph, const Volume start, const Volume end, const bool no_communication, const AcDSLTaskGraph bc_graph));

FUNC_DEFINE(AcDSLTaskGraph, acGetComputeStepsBCs,(const AcDSLTaskGraph graph));


FUNC_DEFINE(AcResult, acGridDestroyTaskGraph,(AcTaskGraph* graph));

FUNC_DEFINE(AcResult, acGridClearTaskGraphCache,());

FUNC_DEFINE(AcResult, acGridExecuteTaskGraph,(AcTaskGraph* graph, const size_t n_iterations));

FUNC_DEFINE(AcResult, acGridExecuteTaskGraphBase,(AcTaskGraph* graph, const size_t n_iterations, const bool include_all));

FUNC_DEFINE(AcResult, acGridFinalizeReduceLocal,(AcTaskGraph* graph));

FUNC_DEFINE(AcResult, acGridFinalizeReduce,(AcTaskGraph* graph));

FUNC_DEFINE(AcResult, acGridLaunchKernel,(const Stream stream, const AcKernel kernel, const Volume start,
                            const Volume end));


FUNC_DEFINE(AcResult, acGridLoadStencil,(const Stream stream, const Stencil stencil,
                           const AcReal data[STENCIL_DEPTH][STENCIL_HEIGHT][STENCIL_WIDTH]));

FUNC_DEFINE(AcResult, acGridStoreStencil,(const Stream stream, const Stencil stencil,
                            AcReal data[STENCIL_DEPTH][STENCIL_HEIGHT][STENCIL_WIDTH]));

FUNC_DEFINE(AcResult, acGridLoadStencils,(const Stream stream,
                   const AcReal data[NUM_STENCILS][STENCIL_DEPTH][STENCIL_HEIGHT][STENCIL_WIDTH]));

FUNC_DEFINE(AcResult, acGridStoreStencils,(const Stream stream,
                    AcReal data[NUM_STENCILS][STENCIL_DEPTH][STENCIL_HEIGHT][STENCIL_WIDTH]));
static UNUSED bool
ac_function_always_false()
{
    return false;
}

#if AC_RUNTIME_COMPILATION
static UNUSED bool (*acGridInitialized)() = ac_function_always_false;
#else
FUNC_DEFINE(bool, acGridInitialized, ());
#endif

AC_END_C_DECLARATIONS

#if __cplusplus

static UNUSED AcResult
acGridInit(const AcMesh mesh)
{
	return acGridInitBase(mesh);
}

using KernelParamsLoader = std::function<void(ParamLoadingInfo step_info)>;
OVERLOADED_FUNC_DEFINE(AcTaskDefinition, acComputeWithParams,
                       (const AcKernel kernel, Field fields_in[], const size_t num_fields_in,
                        Field fields_out[], const size_t num_fields_out, Profile profiles_in[],
                        const size_t num_profiles_in, Profile profiles_reduce_out[],
                        const size_t num_profiles_reduce_out, Profile profiles_write_out[],
                        const size_t num_profiles_write_out, KernelReduceOutput reduce_outputs_in[],
                        size_t num_outputs_in, KernelReduceOutput reduce_outputs_out[],
                        size_t num_outputs_out, const Volume start, const Volume end,
                        const int onion_level, KernelParamsLoader loader));

OVERLOADED_FUNC_DEFINE(AcTaskDefinition, acBoundaryCondition,
                       (const AcBoundary boundary, const AcKernel kernel, const Field fields_in[],
                        const size_t num_fields_in, const Field fields_out[],
                        const size_t num_fields_out, const KernelParamsLoader));
FUNC_DEFINE(AcTaskDefinition, acBoundaryConditionWithBounds,
            (const AcBoundary boundary, const AcKernel kernel, const Field fields_in[],
             const size_t num_fields_in, const Field fields_out[], const size_t num_fields_out,
             const Volume start, const Volume end, const facet_class_range halo_types[],
             const int3 id, const KernelParamsLoader));

OVERLOADED_FUNC_DEFINE(AcTaskDefinition, acRayUpdate,
                       (const AcKernel kernel, const AcBoundary boundary, const int3 ray_direction,
                        Field fields_in[], const size_t num_fields_in, Field fields_out[],
                        const size_t num_fields_out, KernelParamsLoader loader));

static UNUSED AcTaskGraph*
acGetOptimizedDSLTaskGraph(const AcDSLTaskGraph graph, const Volume start, const Volume end)
{
    return acGetOptimizedDSLTaskGraphWithBounds(graph, start, end, false,
                                                acGetComputeStepsBCs(graph));
}

static UNUSED AcTaskGraph*
acGetOptimizedDSLTaskGraph(const AcDSLTaskGraph graph, const Volume start, const Volume end,
                           const bool globally_imposed_bcs)
{
    return acGetOptimizedDSLTaskGraphWithBounds(graph, start, end, true,
                                                acGetComputeStepsBCs(graph));
}

static UNUSED AcTaskGraph*
acGetOptimizedDSLTaskGraph(const AcDSLTaskGraph graph, const bool globally_imposed_bcs)
{
    return acGetOptimizedDSLTaskGraphWithBounds(graph,
                                                acGetMinNN(
                                                    acDeviceGetLocalConfig(acGridGetDevice())),
                                                acGetMaxNN(
                                                    acDeviceGetLocalConfig(acGridGetDevice())),
                                                globally_imposed_bcs, acGetComputeStepsBCs(graph));
}

static UNUSED AcTaskGraph*
acGetOptimizedDSLTaskGraph(const AcDSLTaskGraph graph, const bool globally_imposed_bcs,
                           const AcDSLTaskGraph bc_graph)
{
    return acGetOptimizedDSLTaskGraphWithBounds(graph,
                                                acGetMinNN(
                                                    acDeviceGetLocalConfig(acGridGetDevice())),
                                                acGetMaxNN(
                                                    acDeviceGetLocalConfig(acGridGetDevice())),
                                                globally_imposed_bcs, bc_graph);
}

/** Backwards compatible interface, input fields = output fields*/
template <size_t num_fields>
static AcTaskDefinition
acCompute(AcKernel kernel, Field (&fields)[num_fields])
{
    return BASE_FUNC_NAME(acCompute)(kernel, fields, num_fields, fields, num_fields, NULL, 0, NULL,
                                     0);
}
static UNUSED AcTaskDefinition
acCompute(AcKernel kernel, std::vector<Field> fields)
{
    return BASE_FUNC_NAME(acCompute)(kernel, fields.data(), fields.size(), fields.data(),
                                     fields.size(), NULL, 0, NULL, 0);
}

static UNUSED AcTaskDefinition
acRayUpdate(AcKernel kernel, const int3 ray_direction, std::vector<Field> fields,
            KernelParamsLoader loader)
{
    return BASE_FUNC_NAME(acRayUpdate)(kernel, BOUNDARY_XYZ, ray_direction, fields.data(),
                                       fields.size(), fields.data(), fields.size(), loader);
}

static UNUSED AcTaskDefinition
acRayUpdate(AcKernel kernel, const AcBoundary boundary, const int3 ray_direction,
            std::vector<Field> fields, KernelParamsLoader loader)
{
    return BASE_FUNC_NAME(acRayUpdate)(kernel, boundary, ray_direction, fields.data(),
                                       fields.size(), fields.data(), fields.size(), loader);
}

static UNUSED AcTaskDefinition
acRayUpdate(AcKernel kernel, const int3 ray_direction, std::vector<Field> fields)
{
    KernelParamsLoader loader = [&](ParamLoadingInfo p) { (void)p; };
    return BASE_FUNC_NAME(acRayUpdate)(kernel, BOUNDARY_XYZ, ray_direction, fields.data(),
                                       fields.size(), fields.data(), fields.size(),
                                       (KernelParamsLoader){});
}

static UNUSED AcTaskDefinition
acRayUpdate(AcKernel kernel, const AcBoundary boundary, const int3 ray_direction,
            std::vector<Field> fields)
{
    KernelParamsLoader loader = [&](ParamLoadingInfo p) { (void)p; };
    return BASE_FUNC_NAME(acRayUpdate)(kernel, boundary, ray_direction, fields.data(),
                                       fields.size(), fields.data(), fields.size(),
                                       (KernelParamsLoader){});
}

template <size_t num_fields>
static AcTaskDefinition
acComputeWithParams(AcKernel kernel, Field (&fields)[num_fields], KernelParamsLoader loader)
{
    return BASE_FUNC_NAME(acComputeWithParams)(kernel, fields, num_fields, fields, num_fields, NULL,
                                               0, NULL, 0, (Volume){0, 0, 0}, (Volume){0, 0, 0}, 1,
                                               loader);
}

template <size_t num_fields_in, size_t num_fields_out>
static AcTaskDefinition
acBoundaryCondition(const AcBoundary boundary, AcKernel kernel, Field (&fields_in)[num_fields_in],
                    Field (&fields_out)[num_fields_out],
                    std::function<void(ParamLoadingInfo)> loader)
{
    return BASE_FUNC_NAME(aBoundaryCondition)(boundary, kernel, fields_in, num_fields_in,
                                              fields_out, num_fields_out, loader);
}

template <size_t num_fields>
static AcTaskDefinition
acBoundaryCondition(const AcBoundary boundary, AcKernel kernel, Field (&fields)[num_fields],
                    std::function<void(ParamLoadingInfo)> loader)
{
    return BASE_FUNC_NAME(acBoundaryCondition)(boundary, kernel, fields, num_fields, fields,
                                               num_fields, loader);
}

template <size_t num_fields_in, size_t num_fields_out>
static AcTaskDefinition
acCompute(AcKernel kernel, Field (&fields_in)[num_fields_in], Field (&fields_out)[num_fields_out])
{
    return BASE_FUNC_NAME(acCompute)(kernel, fields_in, num_fields_in, fields_out, num_fields_out);
}

template <size_t num_fields_in, size_t num_fields_out>
static AcTaskDefinition
acComputeWithParams(AcKernel kernel, Field (&fields_in)[num_fields_in],
                    Field (&fields_out)[num_fields_out],
                    std::function<void(ParamLoadingInfo)> loader)
{
    return BASE_FUNC_NAME(acComputeWithParams)(kernel, fields_in, num_fields_in, fields_out,
                                               num_fields_out, 1, loader);
}

static inline AcTaskDefinition
acComputeWithParams(AcKernel kernel, std::vector<Field> fields_in, std::vector<Field> fields_out,
                    std::function<void(ParamLoadingInfo)> loader)
{
    return BASE_FUNC_NAME(acComputeWithParams)(kernel, fields_in.data(), fields_in.size(),
                                               fields_out.data(), fields_out.size(), NULL, 0, NULL,
                                               0, NULL, 0, NULL, 0, NULL, 0, (Volume){0, 0, 0},
                                               (Volume){0, 0, 0}, 1, loader);
}

static inline AcTaskDefinition
acCompute(AcKernel kernel, std::vector<Field> fields_in, std::vector<Field> fields_out,
          std::function<void(ParamLoadingInfo)> loader)
{
    return BASE_FUNC_NAME(acComputeWithParams)(kernel, fields_in.data(), fields_in.size(),
                                               fields_out.data(), fields_out.size(), NULL, 0, NULL,
                                               0, NULL, 0, NULL, 0, NULL, 0, (Volume){0, 0, 0},
                                               (Volume){0, 0, 0}, 1, loader);
}

static inline AcTaskDefinition
acCompute(AcKernel kernel, std::vector<Field> fields_in, std::vector<Field> fields_out,
          std::vector<Profile> profile_in, std::vector<Profile> profile_out,
          std::function<void(ParamLoadingInfo)> loader)
{
    return BASE_FUNC_NAME(
        acComputeWithParams)(kernel, fields_in.data(), fields_in.size(), fields_out.data(),
                             fields_out.size(), profile_in.data(), profile_in.size(),
                             profile_out.data(), profile_out.size(), NULL, 0, NULL, 0, NULL, 0,
                             (Volume){0, 0, 0}, (Volume){0, 0, 0}, 1, loader);
}

static inline AcTaskDefinition
acCompute(AcKernel kernel, std::vector<Field> fields_in, std::vector<Field> fields_out,
          std::vector<Profile> profile_in, std::vector<Profile> profile_reduce_out,
          std::vector<Profile> profile_write_out, std::vector<KernelReduceOutput> reduce_outputs_in,
          std::vector<KernelReduceOutput> reduce_outputs_out, const Volume start, const Volume end,
          std::function<void(ParamLoadingInfo)> loader)
{
    return BASE_FUNC_NAME(acComputeWithParams)(kernel, fields_in.data(), fields_in.size(),
                                               fields_out.data(), fields_out.size(),
                                               profile_in.data(), profile_in.size(),
                                               profile_reduce_out.data(), profile_reduce_out.size(),
                                               profile_write_out.data(), profile_write_out.size(),
                                               reduce_outputs_in.data(), reduce_outputs_in.size(),
                                               reduce_outputs_out.data(), reduce_outputs_out.size(),
                                               start, end, 1, loader);
}

static inline AcTaskDefinition
acCompute(AcKernel kernel, std::vector<Field> fields_in, std::vector<Field> fields_out,
          std::vector<Profile> profile_in, std::vector<Profile> profile_reduce_out,
          std::vector<Profile> profile_write_out, std::vector<KernelReduceOutput> reduce_outputs_in,
          std::vector<KernelReduceOutput> reduce_outputs_out, const Volume start, const Volume end,
          const int onion_level, std::function<void(ParamLoadingInfo)> loader)
{
    return BASE_FUNC_NAME(acComputeWithParams)(kernel, fields_in.data(), fields_in.size(),
                                               fields_out.data(), fields_out.size(),
                                               profile_in.data(), profile_in.size(),
                                               profile_reduce_out.data(), profile_reduce_out.size(),
                                               profile_write_out.data(), profile_write_out.size(),
                                               reduce_outputs_in.data(), reduce_outputs_in.size(),
                                               reduce_outputs_out.data(), reduce_outputs_out.size(),
                                               start, end, onion_level, loader);
}

static inline AcTaskDefinition
acCompute(AcKernel kernel, std::vector<Field> fields_in, std::vector<Field> fields_out)
{
    return BASE_FUNC_NAME(acCompute)(kernel, fields_in.data(), fields_in.size(), fields_out.data(),
                                     fields_out.size(), NULL, 0, NULL, 0);
}

static inline AcTaskDefinition
acCompute(AcKernel kernel, std::vector<Field> fields, std::function<void(ParamLoadingInfo)> loader)
{
    return BASE_FUNC_NAME(acComputeWithParams)(kernel, fields.data(), fields.size(), fields.data(),
                                               fields.size(), NULL, 0, NULL, 0, NULL, 0, NULL, 0,
                                               NULL, 0, (Volume){0, 0, 0}, (Volume){0, 0, 0}, 1,
                                               loader);
}

template <size_t num_fields_in, size_t num_fields_out>
static AcTaskDefinition
acCompute(AcKernel kernel, Field (&fields_in)[num_fields_in], Field (&fields_out)[num_fields_out],
          std::function<void(ParamLoadingInfo)> loader)
{
    return BASE_FUNC_NAME(acComputeWithParams)(kernel, fields_in, num_fields_in, fields_out,
                                               num_fields_out, NULL, 0, NULL, 0, NULL, 0, NULL, 0,
                                               (Volume){0, 0, 0}, (Volume){0, 0, 0}, 1, loader);
}

template <size_t num_fields>
static AcTaskDefinition
acHaloExchange(Field (&fields)[num_fields])
{
    return BASE_FUNC_NAME(acHaloExchange)(fields, num_fields);
}

static inline AcTaskDefinition
acHaloExchange(std::vector<Field> fields)
{
    return BASE_FUNC_NAME(acHaloExchange)(fields.data(), fields.size());
}

static inline AcTaskDefinition
acHaloExchange(std::vector<Field> fields, std::vector<facet_class_range> halo_types)
{
    const auto start = acGetMinNN(acDeviceGetLocalConfig(acGridGetDevice()));
    const auto end   = acGetMaxNN(acDeviceGetLocalConfig(acGridGetDevice()));
    return acHaloExchangeWithBounds(fields.data(), fields.size(), start, end, (int3){0, 0, 0}, true,
                                    true, BOUNDARY_XYZ, false, halo_types.data(),
                                    (Volume){NGHOST, NGHOST, NGHOST});
}

static inline AcTaskDefinition
acScan(std::vector<Field> fields, const int3 ray_direction)
{
    return BASE_FUNC_NAME(acScan)(fields.data(), fields.size(), ray_direction);
}

static inline AcTaskDefinition
acHaloExchange(std::vector<Field> fields, const Volume start, const Volume end)
{
    std::vector<facet_class_range> halo_types{};
    for (size_t i = 0; i < fields.size(); ++i) halo_types.push_back((facet_class_range){1, 2});
    return acHaloExchangeWithBounds(fields.data(), fields.size(), start, end, (int3){0, 0, 0}, true,
                                    true, BOUNDARY_XYZ, false, halo_types.data(),
                                    (Volume){NGHOST, NGHOST, NGHOST});
}

static inline AcTaskDefinition
acHaloExchange(std::vector<Field> fields, const Volume start, const Volume end,
               const int3 ray_direction, const bool sending, const bool receiving)
{
    std::vector<facet_class_range> halo_types{};
    for (size_t i = 0; i < fields.size(); ++i) halo_types.push_back((facet_class_range){1, 2});
    return acHaloExchangeWithBounds(fields.data(), fields.size(), start, end, ray_direction,
                                    sending, receiving, BOUNDARY_XYZ, false, halo_types.data(),
                                    (Volume){NGHOST, NGHOST, NGHOST});
}

static inline AcTaskDefinition
acHaloExchange(std::vector<Field> fields, const int3 ray_direction, const bool sending,
               const bool receiving)
{
    const auto start = acGetMinNN(acDeviceGetLocalConfig(acGridGetDevice()));
    const auto end   = acGetMaxNN(acDeviceGetLocalConfig(acGridGetDevice()));
    std::vector<facet_class_range> halo_types{};
    for (size_t i = 0; i < fields.size(); ++i) halo_types.push_back((facet_class_range){1, 2});
    return acHaloExchangeWithBounds(fields.data(), fields.size(), start, end, ray_direction,
                                    sending, receiving, BOUNDARY_XYZ, false, halo_types.data(),
                                    (Volume){NGHOST, NGHOST, NGHOST});
}

static inline AcTaskDefinition
acHaloExchange(std::vector<Field> fields, const int3 ray_direction, const bool sending,
               const bool receiving, const AcBoundary boundary)
{
    const auto start = acGetMinNN(acDeviceGetLocalConfig(acGridGetDevice()));
    const auto end   = acGetMaxNN(acDeviceGetLocalConfig(acGridGetDevice()));
    std::vector<facet_class_range> halo_types{};
    for (size_t i = 0; i < fields.size(); ++i) halo_types.push_back((facet_class_range){1, 2});
    return acHaloExchangeWithBounds(fields.data(), fields.size(), start, end, ray_direction,
                                    sending, receiving, boundary, false, halo_types.data(),
                                    (Volume){NGHOST, NGHOST, NGHOST});
}

static inline AcTaskDefinition
acHaloExchange(std::vector<Field> fields, const Volume start, const Volume end,
               const int3 ray_direction, const bool sending, const bool receiving,
               const AcBoundary boundary)
{
    std::vector<facet_class_range> halo_types{};
    for (size_t i = 0; i < fields.size(); ++i) halo_types.push_back((facet_class_range){1, 2});
    return acHaloExchangeWithBounds(fields.data(), fields.size(), start, end, ray_direction,
                                    sending, receiving, boundary, false, halo_types.data(),
                                    (Volume){NGHOST, NGHOST, NGHOST});
}

static inline AcTaskDefinition
acHaloExchange(std::vector<Field> fields, const Volume start, const Volume end,
               const int3 ray_direction, const bool sending, const bool receiving,
               const AcBoundary boundary, const std::vector<facet_class_range> halo_types)
{
    return acHaloExchangeWithBounds(fields.data(), fields.size(), start, end, ray_direction,
                                    sending, receiving, boundary, false, halo_types.data(),
                                    (Volume){NGHOST, NGHOST, NGHOST});
}

static inline AcTaskDefinition
acHaloExchange(std::vector<Field> fields, const Volume start, const Volume end,
               const int3 ray_direction, const bool sending, const bool receiving,
               const AcBoundary boundary, const std::vector<facet_class_range> halo_types,
               const Volume halo_size)
{
    return acHaloExchangeWithBounds(fields.data(), fields.size(), start, end, ray_direction,
                                    sending, receiving, boundary, false, halo_types.data(),
                                    halo_size);
}

static inline AcTaskDefinition
acHaloExchange(std::vector<Field> fields, const int3 ray_direction, const AcBoundary boundary)
{
    const auto start = acGetMinNN(acDeviceGetLocalConfig(acGridGetDevice()));
    const auto end   = acGetMaxNN(acDeviceGetLocalConfig(acGridGetDevice()));
    std::vector<facet_class_range> halo_types{};
    for (size_t i = 0; i < fields.size(); ++i) halo_types.push_back((facet_class_range){1, 2});
    return acHaloExchangeWithBounds(fields.data(), fields.size(), start, end, ray_direction, true,
                                    true, boundary, false, halo_types.data(),
                                    (Volume){NGHOST, NGHOST, NGHOST});
}

static inline AcTaskDefinition
acHaloExchange(std::vector<Field> fields, const int3 ray_direction, const AcBoundary boundary,
               const bool include_boundaries)
{
    const auto start = acGetMinNN(acDeviceGetLocalConfig(acGridGetDevice()));
    const auto end   = acGetMaxNN(acDeviceGetLocalConfig(acGridGetDevice()));
    std::vector<facet_class_range> halo_types{};
    for (size_t i = 0; i < fields.size(); ++i) halo_types.push_back((facet_class_range){1, 2});
    return acHaloExchangeWithBounds(fields.data(), fields.size(), start, end, ray_direction, true,
                                    true, boundary, include_boundaries, halo_types.data(),
                                    (Volume){NGHOST, NGHOST, NGHOST});
}

static inline AcTaskDefinition
acBoundaryCondition(const AcBoundary boundary, AcKernel kernel, std::vector<Field> fields,
                    std::function<void(ParamLoadingInfo)> loader)
{
    return BASE_FUNC_NAME(acBoundaryCondition)(boundary, kernel, fields.data(), fields.size(),
                                               fields.data(), fields.size(), loader);
}
template <typename T>
static inline AcTaskDefinition
acBoundaryCondition(const AcBoundary boundary, AcKernel kernel, std::vector<Field> fields,
                    const T param)
{
    auto loader = [&](ParamLoadingInfo p) {
        auto config = acDeviceGetLocalConfig(p.device);
        acLoadKernelParams(*p.params, kernel, p.vtxbuf, config[param]);
    };
    return BASE_FUNC_NAME(acBoundaryCondition)(boundary, kernel, fields.data(), fields.size(),
                                               fields.data(), fields.size(), loader);
}

static inline AcTaskDefinition
acBoundaryCondition(const AcBoundary boundary, AcKernel kernel, std::vector<Field> fields,
                    const std::vector<facet_class_range> halo_types)
{
    const auto start = acGetMinNN(acDeviceGetLocalConfig(acGridGetDevice()));
    const auto end   = acGetMaxNN(acDeviceGetLocalConfig(acGridGetDevice()));
    std::function<void(ParamLoadingInfo)> loader = [](const ParamLoadingInfo& p) { (void)p; };
    return acBoundaryConditionWithBounds(boundary, kernel, fields.data(), fields.size(),
                                         fields.data(), fields.size(), start, end,
                                         halo_types.data(), (int3){0, 0, 0}, loader);
}

static inline AcTaskDefinition
acBoundaryCondition(const AcBoundary boundary, AcKernel kernel, std::vector<Field> fields_in,
                    std::vector<Field> fields_out)
{
    std::function<void(ParamLoadingInfo)> loader = [](const ParamLoadingInfo& p) { (void)p; };
    return BASE_FUNC_NAME(acBoundaryCondition)(boundary, kernel, fields_in.data(), fields_in.size(),
                                               fields_out.data(), fields_out.size(), loader);
}

static inline AcTaskDefinition
acBoundaryCondition(const AcBoundary boundary, AcKernel kernel, std::vector<Field> fields_in,
                    std::vector<Field> fields_out, const Volume start, const Volume end)
{
    std::function<void(ParamLoadingInfo)> loader = [](const ParamLoadingInfo& p) { (void)p; };
    std::vector<facet_class_range> halo_types{};
    for (size_t i = 0; i < fields_out.size(); ++i) halo_types.push_back((facet_class_range){1, 2});
    return acBoundaryConditionWithBounds(boundary, kernel, fields_in.data(), fields_in.size(),
                                         fields_out.data(), fields_out.size(), start, end,
                                         halo_types.data(), (int3){0, 0, 0}, loader);
}

static inline AcTaskDefinition
acBoundaryCondition(const AcBoundary boundary, AcKernel kernel, std::vector<Field> fields_in,
                    std::vector<Field> fields_out, const Volume start, const Volume end,
                    const int3 id)
{
    std::function<void(ParamLoadingInfo)> loader = [](const ParamLoadingInfo& p) { (void)p; };
    std::vector<facet_class_range> halo_types{};
    for (size_t i = 0; i < fields_out.size(); ++i) halo_types.push_back((facet_class_range){1, 2});
    return acBoundaryConditionWithBounds(boundary, kernel, fields_in.data(), fields_in.size(),
                                         fields_out.data(), fields_out.size(), start, end,
                                         halo_types.data(), id, loader);
}

static inline AcTaskDefinition
acBoundaryCondition(const AcBoundary boundary, AcKernel kernel, std::vector<Field> fields_in,
                    std::vector<Field> fields_out, const Volume start, const Volume end,
                    const std::vector<facet_class_range> halo_types)
{
    std::function<void(ParamLoadingInfo)> loader = [](const ParamLoadingInfo& p) { (void)p; };
    return acBoundaryConditionWithBounds(boundary, kernel, fields_in.data(), fields_in.size(),
                                         fields_out.data(), fields_out.size(), start, end,
                                         halo_types.data(), (int3){0, 0, 0}, loader);
}

static inline AcTaskDefinition
acBoundaryCondition(const AcBoundary boundary, AcKernel kernel, std::vector<Field> fields_in,
                    std::vector<Field> fields_out, const Volume start, const Volume end,
                    const std::vector<facet_class_range> halo_types, const int3 id)
{
    std::function<void(ParamLoadingInfo)> loader = [](const ParamLoadingInfo& p) { (void)p; };
    return acBoundaryConditionWithBounds(boundary, kernel, fields_in.data(), fields_in.size(),
                                         fields_out.data(), fields_out.size(), start, end,
                                         halo_types.data(), id, loader);
}

static inline AcTaskDefinition
acBoundaryCondition(const AcBoundary boundary, AcKernel kernel, std::vector<Field> fields,
                    const Volume start, const Volume end)
{
    std::function<void(ParamLoadingInfo)> loader = [](const ParamLoadingInfo& p) { (void)p; };
    std::vector<facet_class_range> halo_types{};
    for (size_t i = 0; i < fields.size(); ++i) halo_types.push_back((facet_class_range){1, 2});
    return acBoundaryConditionWithBounds(boundary, kernel, fields.data(), fields.size(),
                                         fields.data(), fields.size(), start, end,
                                         halo_types.data(), (int3){0, 0, 0}, loader);
}

static inline AcTaskDefinition
acBoundaryCondition(const AcBoundary boundary, AcKernel kernel, std::vector<Field> fields,
                    const Volume start, const Volume end,
                    const std::vector<facet_class_range> halo_types)
{
    std::function<void(ParamLoadingInfo)> loader = [](const ParamLoadingInfo& p) { (void)p; };
    return acBoundaryConditionWithBounds(boundary, kernel, fields.data(), fields.size(),
                                         fields.data(), fields.size(), start, end,
                                         halo_types.data(), (int3){0, 0, 0}, loader);
}

static inline AcTaskDefinition
acBoundaryCondition(const AcBoundary boundary, AcKernel kernel, std::vector<Field> fields)
{
    std::function<void(ParamLoadingInfo)> loader = [](const ParamLoadingInfo& p) { (void)p; };
    return BASE_FUNC_NAME(acBoundaryCondition)(boundary, kernel, fields.data(), fields.size(),
                                               fields.data(), fields.size(), loader);
}

template <size_t n_ops>
static AcTaskGraph*
acGridBuildTaskGraph(const AcTaskDefinition (&ops)[n_ops])
{
    return BASE_FUNC_NAME(acGridBuildTaskGraph)(ops, n_ops);
}

static UNUSED AcTaskGraph*
acGridBuildTaskGraph(const std::vector<AcTaskDefinition> ops)
{
    return BASE_FUNC_NAME(acGridBuildTaskGraph)(ops.data(), ops.size());
}

template <size_t n_ops>
static AcTaskGraph*
acGridBuildTaskGraphWithBounds(const AcTaskDefinition (&ops)[n_ops], const Volume start,
                               const Volume end)
{
    return BASE_FUNC_NAME(acGridBuildTaskGraphWithBounds)(ops, n_ops, start, end, false);
}

static UNUSED AcTaskGraph*
acGridBuildTaskGraph(const std::vector<AcTaskDefinition> ops, const Volume start, const Volume end)
{
    return BASE_FUNC_NAME(acGridBuildTaskGraphWithBounds)(ops.data(), ops.size(), start, end,
                                                          false);
}

template <size_t n_ops>
static AcTaskGraph*
acGridBuildTaskGraphWithBounds(const AcTaskDefinition (&ops)[n_ops], const Volume start,
                               const Volume end, const bool globally_imposed_bcs)
{
    return BASE_FUNC_NAME(acGridBuildTaskGraphWithBounds)(ops, n_ops, start, end,
                                                          globally_imposed_bcs);
}

static UNUSED AcTaskGraph*
acGridBuildTaskGraph(const std::vector<AcTaskDefinition> ops, const Volume start, const Volume end,
                     const bool globally_imposed_bcs)
{
    return BASE_FUNC_NAME(acGridBuildTaskGraphWithBounds)(ops.data(), ops.size(), start, end,
                                                          globally_imposed_bcs);
}

#if AC_RUNTIME_COMPILATION
static UNUSED AcTaskGraph*
acGetOptimizedDSLTaskGraph(const AcDSLTaskGraph graph)
{
    return BASE_FUNC_NAME(acGetOptimizedDSLTaskGraph)(graph);
}
#endif

#else

FUNC_DEFINE(AcTaskDefinition, acComputeWithParams,
            (const AcKernel kernel, Field fields_in[], const size_t num_fields_in,
             Field fields_out[], const size_t num_fields_out, Profile profiles_in[],
             const size_t num_profiles_in, Profile profiles_out[], const size_t num_profiles_out,
             const Volume start, const Volume dims, const int onion_level,
             void (*load_func)(ParamLoadingInfo step_info)));

OVERLOADED_FUNC_DEFINE(AcTaskDefinition, acBoundaryCondition,
                       (const AcBoundary boundary, AcKernel kernel, Field fields_in[],
                        const size_t num_fields_in, Field fields_out[], const size_t num_fields_out,
                        void (*load_func)(ParamLoadingInfo step_info)));

FUNC_DEFINE(AcTaskDefinition, acBoundaryConditionWithBounds,
            (const AcBoundary boundary, AcKernel kernel, Field fields_in[],
             const size_t num_fields_in, Field fields_out[], const size_t num_fields_out,
             const Volume start, const Volume end, const facet_class_range halo_types[],
             const int3 id, void (*load_func)(ParamLoadingInfo step_info)));

OVERLOADED_FUNC_DEFINE(AcTaskDefinition, acRayUpdate,
                       (const AcKernel kernel, const AcBoundary boundary, const int3 ray_direction,
                        Field fields_in[], const size_t num_fields_in, Field fields_out[],
                        const size_t num_fields_out,
                        void (*load_func)(ParamLoadingInfo step_info)));

#endif

#endif // AC_MPI_ENABLED
