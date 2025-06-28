/*
    Copyright (C) 2020-2021, Johannes Pekkilä, Miikka Väisälä, Oskar Lappi

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
#if AC_MPI_ENABLED
/**
 * Quick overview of the MPI implementation:
 *
 * The halo is partitioned into segments, each segment is assigned a HaloExchangeTask.
 * A HaloExchangeTask sends local data as a halo to a neighbor
 * and receives halo data from a (possibly different) neighbor.
 *
 * struct PackedData is used for packing and unpacking. Holds the actual data in
 *                   the halo partition (wrapped by HaloMessage)
 * struct Grid contains information about the local GPU device, decomposition,
 *             the total mesh dimensions, default tasks, and MPI requests
 * struct AcTaskGraph contains *Tasks*, encapsulated pieces of work that depend on each other.
 *                  Users can create their own AcTaskGraphs, but the struct implementation is
 *                  hidden from them.

 * Basic steps:
 *   1) Distribute the mesh among ranks
 *   2) Integrate & communicate
 *     - start inner integration and at the same time, pack halo data and send it to neighbors
 *     - as halo data is received and unpacked, integrate segments whose dependencies are ready
 *     - tasks in the task graph are run for three iterations. They are started early as possible
 *   3) Gather the mesh to rank 0 for postprocessing
 *
 * This file contains the grid interface, with algorithms and high level functionality
 * The nitty gritty of the MPI communication and the Task interface is defined in task.h/task.cc
 */

#include "astaroth.h"
#include "astaroth_utils.h"
#include "task.h"
#include "ac_helpers.h"

#include <algorithm>
#include <cstring> //memcpy
#include <mpi.h>
#include <queue>
#include <vector>
#include <stack>

#include "analysis_grid_helpers.h"

#include "decomposition.h" //getPid3D, morton3D
#include "errchk.h"
#include "math_utils.h"
#include "timer_hires.h"

#define ARRAY_SIZE(x) (sizeof(x) / sizeof(x[0]))

AcResult acAnalysisLoadMeshInfo(const AcMeshInfo info);

#ifdef USE_PERFSTUBS
#define PERFSTUBS_USE_TIMER
#include "perfstubs_api/timer.h"
#endif

#define fatal(MESSAGE, ...) \
        { \
	acLogFromRootProc(ac_pid(),MESSAGE,__VA_ARGS__); \
	fflush(stderr); \
	fflush(stdout); \
	ERRCHK_ALWAYS(false); \
	} 
/* Internal interface to grid (a global variable)  */
typedef struct Grid {
    Device device;
    AcMesh submesh;         // Submesh in host memory. Used as scratch space.
    uint3_64 decomposition; // For backwards compatibility. Should use AcDecompositionInfo.
    bool initialized;
    Volume nn;
    std::shared_ptr<AcTaskGraph> default_tasks;
    std::shared_ptr<AcTaskGraph> halo_exchange_tasks;
    std::shared_ptr<AcTaskGraph> periodic_bc_tasks;
    size_t mpi_tag_space_count;
    bool mpi_initialized;
    bool vertex_buffer_copied_from_user[NUM_VTXBUF_HANDLES]{};
    std::vector<KernelAnalysisInfo> kernel_analysis_info{};
} Grid;


#include "internal_device_funcs.h"

static Grid grid = {};

static constexpr int astaroth_comm_split_key = 666;

// In case some old programs still  use MPI_Init or MPI_Init_thread, we don't want to break them
static MPI_Comm astaroth_comm;
static AcSubCommunicators astaroth_sub_comms;


static int
ac_pid()
{
    int pid;
    MPI_Comm_rank(astaroth_comm, &pid);
    return pid;
}


static AcProcMappingStrategy
ac_proc_mapping_strategy()
{
	return grid.submesh.info[AC_proc_mapping_strategy];
}

bool
acGridInitialized()
{
	return grid.initialized;
}

AcMeshInfo
acGridGetLocalMeshInfo(void)
{
    ERRCHK_ALWAYS(grid.initialized);
    return acDeviceGetLocalConfig(grid.device);
}
Device
acGridGetDevice(void)
{
    ERRCHK_ALWAYS(grid.initialized);
    return grid.device;
}

static AcDecomposeStrategy
ac_decomp_strategy()
{
	return (AcDecomposeStrategy)grid.submesh.info[AC_decompose_strategy];
}
static int 
ac_nprocs()
{
    int nprocs;
    MPI_Comm_size(astaroth_comm, &nprocs);
    return nprocs;
}



static uint3_64
get_decomp(const AcMeshInfo global_config)
{
    return uint3_64(acDecompose(ac_nprocs(), global_config));
}
static int3
getPid3D(const AcMeshInfo config)
{
    return getPid3D(ac_pid(), get_decomp(config), 
                   config[AC_proc_mapping_strategy]);
}


static int3
getPid3D(const int pid, const uint3_64 decomp)
{
    return getPid3D(pid, decomp,ac_proc_mapping_strategy()); 
}

static int3
getPid3D(const int pid)
{
    return getPid3D(pid, grid.decomposition,ac_proc_mapping_strategy()); 
}

AcResult
ac_MPI_Init()
{
    ERRCHK_ALWAYS(!grid.mpi_initialized);
    if (MPI_Init(NULL, NULL)) {
        return AC_FAILURE;
    }

    // Get rank for new communicator
    int rank = -1;
    if (MPI_Comm_rank(MPI_COMM_WORLD, &rank) != MPI_SUCCESS) {
        return AC_FAILURE;
    }

    // Split MPI_COMM_WORLD
    if (MPI_Comm_split(MPI_COMM_WORLD, astaroth_comm_split_key, rank, &astaroth_comm) !=
        MPI_SUCCESS) {
        return AC_FAILURE;
    }
    grid.mpi_initialized = true;
    return AC_SUCCESS;
}

AcResult
ac_MPI_Init_thread(int thread_level)
{
    ERRCHK_ALWAYS(!grid.mpi_initialized);
    int thread_support_level = -1;
    int result               = MPI_Init_thread(NULL, NULL, thread_level, &thread_support_level);
    if (thread_support_level < thread_level || result != MPI_SUCCESS) {
        fprintf(stderr, "Thread level %d not supported by the MPI implementation\n", thread_level);
        return AC_FAILURE;
    }

    // Get rank for new communicator
    int rank = -1;
    if (MPI_Comm_rank(MPI_COMM_WORLD, &rank) != MPI_SUCCESS) {
        return AC_FAILURE;
    }

    // Split MPI_COMM_WORLD
    if (MPI_Comm_split(MPI_COMM_WORLD, astaroth_comm_split_key, rank, &astaroth_comm) !=
        MPI_SUCCESS) {
        return AC_FAILURE;
    }
    grid.mpi_initialized = true;
    return AC_SUCCESS;
}

void
ac_MPI_Finalize()
{
    if (astaroth_comm != MPI_COMM_WORLD && astaroth_comm != MPI_COMM_NULL) {
        MPI_Comm_free(&astaroth_comm);
        astaroth_comm = MPI_COMM_NULL;
    }
    MPI_Finalize();
}

MPI_Comm
acGridMPIComm()
{
    ERRCHK_ALWAYS(grid.mpi_initialized);
    return astaroth_comm;
}
AcSubCommunicators
acGridMPISubComms()
{
    	ERRCHK_ALWAYS(grid.initialized);
	return astaroth_sub_comms;
}


int
ac_MPI_Comm_rank()
{
    ERRCHK_ALWAYS(grid.initialized);
    int rank;
    MPI_Comm_rank(acGridMPIComm(), &rank);
    return rank;
}

int
ac_MPI_Comm_size()
{
    ERRCHK_ALWAYS(grid.initialized);
    int nprocs;
    MPI_Comm_size(acGridMPIComm(), &nprocs);
    return nprocs;
}

void
ac_MPI_Barrier()
{
    ERRCHK_ALWAYS(grid.initialized);
    MPI_Barrier(acGridMPIComm());
}

AcResult
acGridSynchronizeStream(const Stream stream)
{
    ERRCHK(grid.initialized);
    acDeviceSynchronizeStream(grid.device, stream);
    MPI_Barrier(astaroth_comm);
    return AC_SUCCESS;
}

AcResult
acGridRandomize(void)
{
    ERRCHK(grid.initialized);

    const Stream stream = STREAM_DEFAULT;

    AcMesh host;
    acHostMeshCreate(grid.submesh.info, &host);
    acHostMeshRandomize(&host);
    acDeviceLoadMesh(grid.device, stream, host);
    acDeviceSynchronizeStream(grid.device, stream);
    acHostMeshDestroy(&host);

    if(grid.submesh.info[AC_fully_periodic_grid]) acGridPeriodicBoundconds(stream);
    acGridSynchronizeStream(stream);

    return AC_SUCCESS;
}

static inline void UNUSED
set_info_val(AcMeshInfo& info, const AcIntParam param, const int value)
{
	acPushToConfig(info,param,value);
}
static inline void UNUSED
set_info_val(AcMeshInfo& info, const AcInt3Param param, const int3 value)
{
	acPushToConfig(info,param,value);
}

static inline void UNUSED
set_info_val(AcMeshInfo& , const int , const int ){}

static inline void UNUSED
set_info_val(AcMeshInfo& , const int3 , const int3 ){}


AcMeshInfo
acGridDecomposeMeshInfo(const AcMeshInfo global_config)
{
    AcMeshInfo submesh_config = global_config;

    const uint3_64 decomp = get_decomp(global_config);

    ERRCHK_ALWAYS(submesh_config[AC_ngrid].x % decomp.x == 0);
    ERRCHK_ALWAYS(submesh_config[AC_ngrid].y % decomp.y == 0);
    ERRCHK_ALWAYS(submesh_config[AC_ngrid].z % decomp.z == 0);

    const Volume nn = acGetGridNN(submesh_config);
    const auto submesh_n_size_t = nn / decomp;
    int3 submesh_n = (int3){(int)submesh_n_size_t.x,(int)submesh_n_size_t.y,(int)submesh_n_size_t.z};

    set_info_val(submesh_config,AC_nlocal,submesh_n);
    const int3 pid3d = getPid3D(global_config);
    const int3 offset = pid3d*submesh_n;
    set_info_val(submesh_config,AC_multigpu_offset,offset);
    submesh_config[AC_multigpu_offset] = pid3d*submesh_n;
    set_info_val(submesh_config,AC_domain_decomposition,(int3){(int)decomp.x, (int)decomp.y, (int)decomp.z});
    set_info_val(submesh_config,AC_domain_coordinates,pid3d);
    acHostUpdateParams(&submesh_config);
    return submesh_config;
}
static Volume
get_global_nn()
{
    const Device device   = grid.device;
    const AcMeshInfo info = acDeviceGetLocalConfig(device);
    return to_volume(info[AC_ngrid]);
}


static void
check_that_decomp_valid(const AcMeshInfo info)
{
    const uint3_64 decomp = get_decomp(info);
    const Volume nn = acGetGridNN(info);
    const bool nx_valid = nn.x % decomp.x == 0;
    const bool ny_valid = nn.y % decomp.y == 0;
    const bool nz_valid = nn.z % decomp.z == 0;
    if (!nx_valid || !ny_valid || !nz_valid) {
        WARNING("Mesh dimensions must be divisible by the decomposition\n");
        fprintf(stderr, "Decomposition: (%lu, %lu, %lu)\n", decomp.x, decomp.y, decomp.z);
        fprintf(stderr, "Mesh dimensions: (%ld, %ld, %ld)\n", nn.x, nn.y, nn.z);
        fprintf(stderr, "Divisible: (%d, %d, %d)\n", nx_valid, ny_valid, nz_valid);
    }
}
static AcBoundary
get_full_boundary()
{
    const auto inactive = acGridGetLocalMeshInfo()[AC_dimension_inactive];
    return
	    (!inactive.x && inactive.y && inactive.z) ? BOUNDARY_X :
	    (inactive.x && !inactive.y && inactive.z) ? BOUNDARY_Y :
	    (inactive.x && inactive.y && !inactive.z) ? BOUNDARY_Z :
	    (!inactive.x && !inactive.y && inactive.z) ? BOUNDARY_XY :
	    (!inactive.x && inactive.y && !inactive.z) ? BOUNDARY_XZ :
	    (inactive.x && !inactive.y && !inactive.z) ? BOUNDARY_YZ :
	    (!inactive.x && !inactive.y && !inactive.z) ? BOUNDARY_XYZ :
	    BOUNDARY_NONE;
}
#ifdef AC_INTEGRATION_ENABLED
static void
gen_default_taskgraph()
{
    acLogFromRootProc(ac_pid(), "acGridInit: Creating default task graph\n");
    std::vector<Field> all_fields_vec{};
    std::vector<Field> all_comm_fields_vec{};
    for (int i = 0; i < NUM_VTXBUF_HANDLES; i++) {
        all_fields_vec.push_back(Field(i));
	if(vtxbuf_is_communicated[i]) all_comm_fields_vec.push_back(Field(i));
    }
#if AC_SINGLEPASS_INTEGRATION
    auto single_loader = [](ParamLoadingInfo l){
	    l.params -> singlepass_solve.step_num = AC_SUBSTEP_NUMBER(l.step_number);
	    l.params -> singlepass_solve.time_params.dt= 
	    acDeviceGetInput(l.device,AC_dt);
	    l.params -> singlepass_solve.time_params.current_time= 
		   acDeviceGetInput(l.device,AC_current_time);
    };
#else
    auto intermediate_loader = [](ParamLoadingInfo l){
	    l.params -> twopass_solve_intermediate.step_num = AC_SUBSTEP_NUMBER(l.step_number);
	    l.params -> twopass_solve_intermediate.dt= 
	    acDeviceGetInput(l.device,AC_dt);
    };
    auto final_loader = [](ParamLoadingInfo l){
	    l.params -> twopass_solve_final.step_num = AC_SUBSTEP_NUMBER(l.step_number);
	    l.params -> twopass_solve_final.current_time= 
		   acDeviceGetInput(l.device,AC_current_time);
    };
#endif

    AcTaskDefinition default_ops[] = {
	    acHaloExchange(all_comm_fields_vec),
            acBoundaryCondition(get_full_boundary(), BOUNDCOND_PERIODIC,all_fields_vec),
#if AC_SINGLEPASS_INTEGRATION
	    acCompute(singlepass_solve, all_comm_fields_vec,all_fields_vec,single_loader),
#else
	    acCompute(twopass_solve_intermediate, all_comm_fields_vec,all_fields_vec,intermediate_loader),
	    acCompute(twopass_solve_final, all_comm_fields_vec,all_comm_fields_vec,final_loader)
#endif
    };
    
    grid.default_tasks = std::shared_ptr<AcTaskGraph>(acGridBuildTaskGraph(default_ops));
    acLogFromRootProc(ac_pid(), "acGridInit: Done creating default task graph\n");
}
#endif
static void
initialize_random_number_generation(const AcMeshInfo submesh_info)
{
    // Random number generator
    // const auto rr            = (int3){STENCIL_WIDTH, STENCIL_HEIGHT, STENCIL_DEPTH};
    // const auto local_m       = acConstructInt3Param(AC_mx, AC_my, AC_mz, submesh_info);
    // const auto global_m      = acConstructInt3Param(AC_nxgrid, AC_nygrid, AC_nzgrid) + 2 * rr;
    // const auto global_offset = submesh_info[AC_multigpu_offset];
    // acRandInit(1234UL, to_volume(local_m), to_volume(global_m), to_volume(global_offset));
    const size_t count = acVertexBufferCompdomainSize(submesh_info);
    acRandInitAlt(1234UL, count, ac_pid());
}
static UNUSED void
log_grid_debug_info(const AcMeshInfo info)
{

    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);

    const Volume nn = acGetLocalNN(info);
    const int3 pid3d = getPid3D(info);
    const auto decomp = get_decomp(info);

    printf("Processor %s. Process %d of %d: (%d, %d, %d)\n", processor_name, ac_pid(), ac_nprocs(), pid3d.x,
           pid3d.y, pid3d.z);
    printf("Decomposition: %lu, %lu, %lu\n", decomp.x, decomp.y, decomp.z);
    printf("Mesh size: %ld, %ld, %ld\n", nn.x,nn.y,nn.z);
    fflush(stdout);
    MPI_Barrier(astaroth_comm);
}

static int
ac_x_color()
{
    const int3 my_coordinates = grid.submesh.info[AC_domain_coordinates];
    return my_coordinates.y + my_coordinates.z*grid.decomposition.y;
}

static int
ac_y_color()
{
    const int3 my_coordinates = grid.submesh.info[AC_domain_coordinates];
    return my_coordinates.x + my_coordinates.z*grid.decomposition.x;
}

static int
ac_z_color()
{
    const int3 my_coordinates = grid.submesh.info[AC_domain_coordinates];
    return my_coordinates.x + my_coordinates.y*grid.decomposition.x;
}

static int
ac_xy_color()
{
    const int3 my_coordinates = grid.submesh.info[AC_domain_coordinates];
    return my_coordinates.z;
}

static int
ac_xz_color()
{
    const int3 my_coordinates = grid.submesh.info[AC_domain_coordinates];
    return my_coordinates.y;
}

static int
ac_yz_color()
{
    const int3 my_coordinates = grid.submesh.info[AC_domain_coordinates];
    return my_coordinates.x;
}

static void
create_astaroth_sub_communicators()
{
	ERRCHK_ALWAYS(MPI_Comm_split(astaroth_comm,ac_x_color(),ac_pid(),&astaroth_sub_comms.x) == MPI_SUCCESS);
	ERRCHK_ALWAYS(MPI_Comm_split(astaroth_comm,ac_y_color(),ac_pid(),&astaroth_sub_comms.y) == MPI_SUCCESS);
	ERRCHK_ALWAYS(MPI_Comm_split(astaroth_comm,ac_z_color(),ac_pid(),&astaroth_sub_comms.z) == MPI_SUCCESS);

	ERRCHK_ALWAYS(MPI_Comm_split(astaroth_comm,ac_xy_color(),ac_pid(),&astaroth_sub_comms.xy) == MPI_SUCCESS);
	ERRCHK_ALWAYS(MPI_Comm_split(astaroth_comm,ac_xz_color(),ac_pid(),&astaroth_sub_comms.xz) == MPI_SUCCESS);
	ERRCHK_ALWAYS(MPI_Comm_split(astaroth_comm,ac_yz_color(),ac_pid(),&astaroth_sub_comms.yz) == MPI_SUCCESS);
}

static void
create_astaroth_comm(const AcMeshInfo info)
{
      switch(info[AC_MPI_comm_strategy])
      {
	case AC_MPI_COMM_STRATEGY_DUP_WORLD:
      		ERRCHK_ALWAYS(MPI_Comm_dup(MPI_COMM_WORLD,&astaroth_comm) == MPI_SUCCESS);
		break;
	case AC_MPI_COMM_STRATEGY_DUP_USER:
	{
		if(info.comm == NULL || info.comm->handle == MPI_COMM_NULL) fatal("%s","Cannot duplicate communicator since it is not loaded!\n");
      		ERRCHK_ALWAYS(MPI_Comm_dup(info.comm->handle,&astaroth_comm) == MPI_SUCCESS);
		break;
	}
	default:
		fatal("%s","Unknown MPICommStrategy\n");
      }
      MPI_Barrier(astaroth_comm);
      

      grid.mpi_initialized = true;
}

static void
check_that_device_allocation_valid()
{
    int device_count = -1;
    acGetDeviceCount(&device_count);
    if (device_count > ac_nprocs()) {
        fprintf(stderr,
                "Invalid device-task allocation: Must allocate one MPI task per GPU but got %d "
                "devices per node and only %d task(s).",
                device_count, ac_nprocs());
        ERRCHK_ALWAYS(device_count <= ac_nprocs());
    }
    MPI_Barrier(astaroth_comm);
}

static void 
check_that_submesh_large_enough(const AcMeshInfo info)
{
    const Volume nn = acGetLocalNN(info);
    if (nn.x < STENCIL_WIDTH && !info[AC_dimension_inactive].x)
        fprintf(stderr, "nn.x %ld too small, must be >= %d (stencil width)\n", nn.x, STENCIL_WIDTH);
    if (nn.y < STENCIL_HEIGHT && !info[AC_dimension_inactive].y)
        fprintf(stderr, "nn.y %ld too small, must be >= %d (stencil height)\n", nn.y,
                STENCIL_HEIGHT);
    if (nn.z < STENCIL_DEPTH && !info[AC_dimension_inactive].z)
        fprintf(stderr, "nn.z %ld too small, must be >= %d (stencil depth)\n", nn.z, STENCIL_DEPTH);
}

static void 
check_that_mesh_large_enough(const AcMeshInfo info)
{
    const Volume nn = acGetGridNN(info);
    if (nn.x < STENCIL_WIDTH && !info[AC_dimension_inactive].x)
        fprintf(stderr, "nn.x %ld too small, must be >= %d (stencil width)\n", nn.x, STENCIL_WIDTH);
    if (nn.y < STENCIL_HEIGHT && !info[AC_dimension_inactive].y)
        fprintf(stderr, "nn.y %ld too small, must be >= %d (stencil height)\n", nn.y,
                STENCIL_HEIGHT);
    if (nn.z < STENCIL_DEPTH && !info[AC_dimension_inactive].z)
        fprintf(stderr, "nn.z %ld too small, must be >= %d (stencil depth)\n", nn.z, STENCIL_DEPTH);
}

static AcMesh
create_grid_submesh(const AcMeshInfo submesh_info, const AcMesh user_mesh)
{
    AcMesh submesh;
    size_t n_allocated_vtxbufs = 0;
    for(int i = 0; i < NUM_FIELDS; ++i)
    {
	    if(user_mesh.vertex_buffer[i] != NULL)
	    {
			submesh.vertex_buffer[i] = user_mesh.vertex_buffer[i];
    			grid.vertex_buffer_copied_from_user[i] = true;
	    }
	    else if(!vtxbuf_is_device_only[i])
	    {
		        ++n_allocated_vtxbufs;
			submesh.vertex_buffer[i] = acHostCreateVertexBuffer(submesh_info,Field(i));
	    }
    }

    if(n_allocated_vtxbufs) acLogFromRootProc(ac_pid(), "acGridInit: Allocated %ld VertexBuffers on the host\n",n_allocated_vtxbufs);
    submesh.info = submesh_info;
    acHostMeshCreateProfiles(&submesh);
    return submesh;
}

static void
check_compile_info_matches_runtime_info(const std::vector<KernelAnalysisInfo> info)
{
[[maybe_unused]] constexpr int AC_STENCIL_CALL        = (1 << 2);
#include "stencil_accesses.h"
	for(int k = 0; k < NUM_KERNELS; ++k)
		for(int j= 0; j< NUM_ALL_FIELDS; ++j)
			for(int i = 0; i < NUM_STENCILS; ++i)
			{
				//TP: in case some fields are dead and Fields are re-ordered because of that
				const int j_old = field_remappings[j];
				//TP: we are a little bit messy by storing info about array reads of vertexbuffers to stencils_accessed so have to check is the stencil call bit set to filter array reads
				const bool comp_time = (stencils_accessed[k][j_old][i] & AC_STENCIL_CALL);
				const bool run_time  = (info[k].stencils_accessed[j][i] & AC_STENCIL_CALL);
				if(run_time && !comp_time)
					fatal("In Kernel %s Stencil %s used for %s at runtime but not generated!\n"
					      "Most likely that stencill is executed in a control-flow path that was not taken.\n"
					      "Consider either runtime-compilation or rewriting the kernel do avoid the conditional stencil call\n"
					      ,kernel_names[k], stencil_names[i], field_names[j]
					      );
				if(!run_time && comp_time)
					acLogFromRootProc(ac_pid(), "PERF WARNING: In Kernel %s Stencil %s generated for %s but not accessed at runtime!\n"
							            "Most likely because the stencil call is performed inside conditional control-flow.\n"
								    "Consider refactoring the code,turning OPTIMIZE_MEM_ACCESSES and/or using runtime-compilation to skip the unnecessary Stencil computation\n"
					     ,kernel_names[k], stencil_names[i], field_names[j]
					      );
				if(run_time)
				{
					if(acGridGetLocalMeshInfo()[AC_dimension_inactive].x && (BOUNDARY_X & get_stencil_boundaries(Stencil(i))))
					{
						fatal("In Kernel %s Used Stencil %s on Field %s even though X is inactive!\n",kernel_names[k],stencil_names[i], field_names[j]);
					}
					if(acGridGetLocalMeshInfo()[AC_dimension_inactive].y && (BOUNDARY_Y & get_stencil_boundaries(Stencil(i))))
					{
						fatal("In Kernel %s Used Stencil %s on Field %s even though Y is inactive!\n",kernel_names[k],stencil_names[i], field_names[j]);
					}
					if(acGridGetLocalMeshInfo()[AC_dimension_inactive].z && (BOUNDARY_Z & get_stencil_boundaries(Stencil(i))))
					{
						fatal("In Kernel %s Used Stencil %s on Field %s even though Z is inactive!\n",kernel_names[k],stencil_names[i], field_names[j]);
					}
				}
			}

}
static AcAutotuneMeasurement
grid_gather_best_measurement(const AcAutotuneMeasurement local_best)
{
        ERRCHK_ALWAYS(grid.initialized);
        float* time_buffer = (float*)malloc(sizeof(float)*ac_nprocs());
        int* x_dim = (int*)malloc(sizeof(int)*ac_nprocs());
        int* y_dim = (int*)malloc(sizeof(int)*ac_nprocs());
        int* z_dim = (int*)malloc(sizeof(int)*ac_nprocs());

        int tpb_x = local_best.tpb.x;
        int tpb_y = local_best.tpb.y;
        int tpb_z = local_best.tpb.z;
        MPI_Gather(&tpb_x,1,MPI_INT,x_dim,1,MPI_INT,0,astaroth_comm);
        MPI_Gather(&tpb_y,1,MPI_INT,y_dim,1,MPI_INT,0,astaroth_comm);
        MPI_Gather(&tpb_z,1,MPI_INT,z_dim,1,MPI_INT,0,astaroth_comm);
        MPI_Gather(&local_best.time,1,MPI_FLOAT,time_buffer,1,MPI_FLOAT,0,astaroth_comm);

        dim3 first_tpb(x_dim[0],y_dim[0],z_dim[0]);
        AcAutotuneMeasurement res{time_buffer[0], first_tpb};
        if(ac_pid() == 0)
        {
                for(int i = 0; i < ac_nprocs(); ++i)
                {
                        if(time_buffer[i]  < res.time)
                        {
                                res.time = time_buffer[i];
                                dim3 res_tpb(x_dim[i],y_dim[i],z_dim[i]);
                                res.tpb = res_tpb;
                        }
                }
                MPI_Bcast(&res.time, 1, MPI_FLOAT, 0, astaroth_comm);
                int x_res = res.tpb.x;
                int y_res = res.tpb.y;
                int z_res = res.tpb.z;
                MPI_Bcast(&x_res, 1, MPI_INT, 0, astaroth_comm);
                MPI_Bcast(&y_res, 1, MPI_INT, 0, astaroth_comm);
                MPI_Bcast(&z_res, 1, MPI_INT, 0, astaroth_comm);
        }
        else
        {
                int x_res;
                int y_res;
                int z_res;
                MPI_Bcast(&res.time, 1, MPI_FLOAT, 0, astaroth_comm);
                MPI_Bcast(&x_res, 1, MPI_INT, 0, astaroth_comm);
                MPI_Bcast(&y_res, 1, MPI_INT, 0, astaroth_comm);
                MPI_Bcast(&z_res, 1, MPI_INT, 0, astaroth_comm);
                res.tpb = dim3(x_res,y_res,z_res);
        }
        free(time_buffer);
        free(x_dim);
        free(y_dim);
        free(z_dim);
	return res;
}
static void
gen_postprocessing_metadata()
{
    if (ac_pid() == 0) {
        FILE* header_file = fopen("ac_grid_info.csv", "w");
	fprintf(header_file,"nxgrid,nygrid,nzgrid,dsx,dsy,dsz\n");
	fprintf(header_file,"%d,%d,%d,%g,%g,%g\n"
			,grid.submesh.info[AC_ngrid].x,grid.submesh.info[AC_ngrid].y,grid.submesh.info[AC_ngrid].z
			,grid.submesh.info[AC_ds].x,grid.submesh.info[AC_ds].y,grid.submesh.info[AC_ds].z
			);
	fclose(header_file);
    }
}

AcResult
acGridInitBase(const AcMesh user_mesh)
{
    int mpi_has_been_initialized{};
    MPI_Initialized(&mpi_has_been_initialized);
    if(!mpi_has_been_initialized)
    {
	    int provided{};
	    MPI_Init_thread(NULL,NULL,MPI_THREAD_MULTIPLE,&provided);
	    ERRCHK_ALWAYS(provided >= MPI_THREAD_MULTIPLE);
    }
    const AcMeshInfo info = user_mesh.info;
    ERRCHK(!grid.initialized);
    if (!grid.mpi_initialized)
      create_astaroth_comm(info);

    if(!mpi_has_been_initialized) acLogFromRootProc(ac_pid(), "acGridInit: MPI was not initialized so called MPI_Init_thread with MPI_THREAD_MULTIPLE\n");

    if(
	info[AC_ngrid].x < 0 ||
	info[AC_ngrid].y < 0 ||
	info[AC_ngrid].z < 0
      )
    	{
    	        fatal("acGridInit: Incorrect AC_ngrid: (%d,%d,%d)!\n",
    	    			info[AC_ngrid].x,
    	    			info[AC_ngrid].y,
    	    			info[AC_ngrid].z
    	    		    );
    	}

    acInitializeRuntimeMPI(ac_pid(), ac_nprocs() ,grid_gather_best_measurement);
    // Check that MPI is initialized
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);

    check_that_device_allocation_valid();


    const int n_active_dimensions = !info[AC_dimension_inactive].x + !info[AC_dimension_inactive].y + !info[AC_dimension_inactive].z;
    acInitDecomposition(n_active_dimensions == 2);
    if(info[AC_decompose_strategy] == AC_DECOMPOSE_STRATEGY_HIERARCHICAL)
    {
        int device_count = -1;
        acGetDeviceCount(&device_count);
    	// Decompose
    	const AcMeshDims mesh_dims = acGetMeshDims(info);
    	const size_t global_dims[] = {
        	as_size_t(mesh_dims.nn.x),
        	as_size_t(mesh_dims.nn.y),
        	as_size_t(mesh_dims.nn.z),
    	};
    	const size_t ndims                  = ARRAY_SIZE(global_dims);
    	const size_t node_count             = as_size_t((ac_nprocs() + device_count - 1) / device_count);
    	const size_t partitions_per_layer[] = {as_size_t(device_count), as_size_t(node_count)};
    	const size_t nlayers                = ARRAY_SIZE(partitions_per_layer);
    	compat_acDecompositionInit(ndims, global_dims, nlayers, partitions_per_layer);
    	// grid.decomposition_info = acDecompositionInit(ndims, global_dims,
    	// nlayers,partitions_per_layer);
    	acVerifyDecomposition(decompose(ac_nprocs(),AC_DECOMPOSE_STRATEGY_HIERARCHICAL),ac_proc_mapping_strategy());
    }

    // grid.decomposition_info = acDecompositionInit(ndims, global_dims,
    // nlayers,partitions_per_layer);
    check_that_decomp_valid(info);

    check_that_mesh_large_enough(info);
    MPI_Barrier(astaroth_comm);

#if AC_VERBOSE
    log_grid_debug_info(info);
#endif

    // Check that mixed precision is correctly configured, AcRealPacked == AC_REAL_MPI_TYPE
    // CAN BE REMOVED IF MIXED PRECISION IS SUPPORTED AS A PREPROCESSOR FLAG
    int mpi_type_size;
    MPI_Type_size(AC_REAL_MPI_TYPE, &mpi_type_size);
    ERRCHK_ALWAYS(sizeof(AcRealPacked) == mpi_type_size);

    // Decompose config (divide dimensions by decomposition)
    AcMeshInfo submesh_info = acGridDecomposeMeshInfo(info);
    check_that_submesh_large_enough(submesh_info);

    // GPU alloc
    int devices_per_node = -1;
    acGetDeviceCount(&devices_per_node);
    if(devices_per_node == 0) fatal("%s", "acGridInit: No devices found!\n");

    const int pid = ac_pid();
    acLogFromRootProc(pid, "acGridInit: n[xyz]grid: (%d,%d,%d) n[xyz]local (%d,%d,%d)\n",
		    submesh_info[AC_ngrid].x,submesh_info[AC_ngrid].y,submesh_info[AC_ngrid].z,
		    submesh_info[AC_nlocal].x,submesh_info[AC_nlocal].y,submesh_info[AC_nlocal].z
		    );
    acLogFromRootProc(pid, "acGridInit: Calling acDeviceCreate\n");
    acVerboseLogFromRootProc(pid,"memusage before acDeviceCreate = %f MBytes\n",acMemUsage()/1024.0);

    Device device;
    acDeviceCreate(pid % devices_per_node, submesh_info, &device);

    acVerboseLogFromRootProc(ac_pid(),"memusage after acDeviceCreate = %f MBytes\n", acMemUsage()/1024.0);
    acLogFromRootProc(ac_pid() , "acGridInit: Returned from acDeviceCreate\n");

    // Setup the global grid structure
    grid.device        = device;
    grid.submesh       = create_grid_submesh(submesh_info, user_mesh);
    grid.decomposition = get_decomp(info);
    acLogFromRootProc(ac_pid(), "acGridInit: decomp: (%ld,%ld,%ld)\n",
		    grid.decomposition.x,
		    grid.decomposition.y,
		    grid.decomposition.z
		    );

    // Configure
    grid.nn = acGetLocalNN(acDeviceGetLocalConfig(device));
    grid.mpi_tag_space_count = 0;


    acDeviceUpdate(device,acDeviceGetLocalConfig(device));

    initialize_random_number_generation(submesh_info);

    create_astaroth_sub_communicators();
    grid.initialized   = true;

    acVerboseLogFromRootProc(ac_pid(), "acGridInit: Synchronizing streams\n");
    acVerboseLogFromRootProc(ac_pid(), "memusage before synchronize stream= %f MBytes\n", acMemUsage()/1024.0);
    acGridSynchronizeStream(STREAM_ALL);
    acVerboseLogFromRootProc(ac_pid(), "memusage after synchronize stream= %f MBytes\n", acMemUsage()/1024.0);
    acVerboseLogFromRootProc(ac_pid(), "acGridInit: Done synchronizing streams\n");

    grid.kernel_analysis_info = get_kernel_analysis_info();
    if(ac_pid() == 0) acAnalysisCheckForDSLErrors(acGridGetLocalMeshInfo());	
    check_compile_info_matches_runtime_info(grid.kernel_analysis_info);

#ifdef AC_INTEGRATION_ENABLED
    if(grid.submesh.info[AC_fully_periodic_grid])
    	gen_default_taskgraph();
#endif
    {
	    std::vector<Field> all_comm_fields_vec{};
	    std::vector<facet_class_range> halo_types{};
	    for (int i = 0; i < NUM_VTXBUF_HANDLES; i++) {
		if(vtxbuf_is_communicated[i]) 
		{
			all_comm_fields_vec.push_back(Field(i));
			halo_types.push_back((facet_class_range){1,3});
		}
	    }
	    grid.halo_exchange_tasks = std::shared_ptr<AcTaskGraph>(acGridBuildTaskGraph({
		    			acHaloExchange(all_comm_fields_vec,halo_types)
					}));
    	    if(grid.submesh.info[AC_fully_periodic_grid])
	    {
	    		grid.periodic_bc_tasks = std::shared_ptr<AcTaskGraph>(acGridBuildTaskGraph({
            				acBoundaryCondition(get_full_boundary(), BOUNDCOND_PERIODIC,all_comm_fields_vec,halo_types),
					}));
	    }

    }
    acAnalysisLoadMeshInfo(acGridGetLocalMeshInfo());
    //Refresh log files
    if(!ac_pid())
    {
    	FILE* fp = fopen("taskgraph_log.txt","w");
    	fclose(fp);
    }

    fflush(stdout);
    fflush(stderr);

    gen_postprocessing_metadata();
    acLogFromRootProc(ac_pid(), "acGridInit: Done\n");

    return AC_SUCCESS;
}

AcResult
acGridQuit(void)
{
    ERRCHK_ALWAYS(grid.initialized);
    acGridSynchronizeStream(STREAM_ALL);

    acGridClearTaskGraphCache();
    // Random number generator
    acRandQuit();

    grid.default_tasks = nullptr;
    grid.halo_exchange_tasks    = nullptr;
    grid.periodic_bc_tasks = nullptr;

    grid.initialized   = false;
    grid.decomposition = (uint3_64){0, 0, 0};
    for(int i = 0; i < NUM_VTXBUF_HANDLES; ++i)
    {
	if(!grid.vertex_buffer_copied_from_user[i])
    		acHostMeshDestroyVertexBuffer(&grid.submesh.vertex_buffer[i]);
	else
		grid.submesh.vertex_buffer[i] = NULL;
	grid.vertex_buffer_copied_from_user[i] = false;
    }
    acDeviceDestroy(&grid.device);
    compat_acDecompositionQuit();
    acRuntimeQuit();
    // acDecompositionInfoDestroy(&grid.decomposition_info);

    return AC_SUCCESS;
}

AcResult
acGridLoadScalarUniform(const Stream stream, const AcRealParam param, const AcReal value)
{
    ERRCHK(grid.initialized);
    acGridSynchronizeStream(stream);

    const int root_proc = 0;
    AcReal buffer       = value;
    MPI_Bcast(&buffer, 1, AC_REAL_MPI_TYPE, root_proc, astaroth_comm);

    return acDeviceLoadScalarUniform(grid.device, stream, param, buffer);
}

AcResult
acGridLoadVectorUniform(const Stream stream, const AcReal3Param param, const AcReal3 value)
{
    ERRCHK(grid.initialized);
    acGridSynchronizeStream(stream);

    const int root_proc = 0;
    AcReal3 buffer      = value;
    MPI_Bcast(&buffer, 3, AC_REAL_MPI_TYPE, root_proc, astaroth_comm);

    return acDeviceLoadVectorUniform(grid.device, stream, param, buffer);
}

AcResult
acGridLoadIntUniform(const Stream stream, const AcIntParam param, const int value)
{
    ERRCHK(grid.initialized);
    acGridSynchronizeStream(stream);

    const int root_proc = 0;
    int buffer          = value;
    MPI_Bcast(&buffer, 1, MPI_INT, root_proc, astaroth_comm);

    return acDeviceLoadIntUniform(grid.device, stream, param, buffer);
}

AcResult
acGridLoadInt3Uniform(const Stream stream, const AcInt3Param param, const int3 value)
{
    ERRCHK(grid.initialized);
    acGridSynchronizeStream(stream);

    const int root_proc = 0;
    int3 buffer         = value;
    MPI_Bcast(&buffer, 3, MPI_INT, root_proc, astaroth_comm);

    return acDeviceLoadInt3Uniform(grid.device, stream, param, buffer);
}

static Volume
get_ghost_zone_sizes()
{
	return to_volume(acGridGetLocalMeshInfo()[AC_nmin]);
}

AcResult
acGridLoadMeshWorking(const Stream stream, const AcMesh host_mesh)
{
    ERRCHK(grid.initialized);
    acGridSynchronizeStream(stream);


    const Volume rr = get_ghost_zone_sizes();
    const Volume monolithic_mm = get_global_nn() + 2 * rr;
    const Volume monolithic_nn = acGetLocalNN(acGridGetLocalMeshInfo());
    const Volume monolithic_offset = rr;

    MPI_Datatype monolithic_subarray;
    const int monolithic_mm_arr[]     = {(int)monolithic_mm.z, (int)monolithic_mm.y, (int)monolithic_mm.x};
    const int monolithic_nn_arr[]     = {(int)monolithic_nn.z, (int)monolithic_nn.y, (int)monolithic_nn.x};
    const int monolithic_offset_arr[] = {(int)monolithic_offset.z, (int)monolithic_offset.y,
                                        (int) monolithic_offset.x};
    MPI_Type_create_subarray(3, monolithic_mm_arr, monolithic_nn_arr, monolithic_offset_arr,
                             MPI_ORDER_C, AC_REAL_MPI_TYPE, &monolithic_subarray);
    MPI_Type_commit(&monolithic_subarray);

    const Volume distributed_mm = acGetLocalMM(acGridGetLocalMeshInfo());
    const Volume distributed_nn = acGetLocalNN(acGridGetLocalMeshInfo());
    const Volume distributed_offset = rr;

    MPI_Datatype distributed_subarray;
    const int distributed_mm_arr[]     = {(int)distributed_mm.z, (int)distributed_mm.y, (int)distributed_mm.x};
    const int distributed_nn_arr[]     = {(int)distributed_nn.z, (int)distributed_nn.y, (int)distributed_nn.x};
    const int distributed_offset_arr[] = {(int)distributed_offset.z, (int)distributed_offset.y,
                                          (int)distributed_offset.x};
    MPI_Type_create_subarray(3, distributed_mm_arr, distributed_nn_arr, distributed_offset_arr,
                             MPI_ORDER_C, AC_REAL_MPI_TYPE, &distributed_subarray);
    MPI_Type_commit(&distributed_subarray);


    MPI_Request recv_reqs[NUM_VTXBUF_HANDLES];
    for (int vtxbuf = 0; vtxbuf < NUM_VTXBUF_HANDLES; ++vtxbuf) {
        MPI_Irecv(grid.submesh.vertex_buffer[vtxbuf], 1, distributed_subarray, 0, vtxbuf,
                  acGridMPIComm(), &recv_reqs[vtxbuf]);
        if (ac_pid() == 0) {
            for (int tgt = 0; tgt < ac_nprocs(); ++tgt) {
                const int3 tgt_pid3d = getPid3D(tgt);
                const size_t idx     = acVertexBufferIdx(tgt_pid3d.x * distributed_nn.x, //
                                                         tgt_pid3d.y * distributed_nn.y, //
                                                         tgt_pid3d.z * distributed_nn.z, //
                                                         host_mesh.info,
							 VertexBufferHandle(vtxbuf));
                MPI_Send(&host_mesh.vertex_buffer[vtxbuf][idx], 1, monolithic_subarray, tgt, vtxbuf,
                         acGridMPIComm());
            }
        }
    }
    MPI_Waitall(NUM_VTXBUF_HANDLES, recv_reqs, MPI_STATUSES_IGNORE);
    /*
        Strategy:
            1) Select a subarray from the input mesh
            2) Select a subarray from the output mesh
            3) Scatter

        Notes:
            1) Check that subarray divisible by number of procs (required in init iirc)
    MPI_Datatype input_subarray_resized;
    MPI_Type_create_resized(input_subarray, 0, sizeof(AcReal), &input_subarray_resized);
    MPI_Type_commit(&input_subarray_resized);

    // Scatter host_mesh from proc 0
    for (int vtxbuf = 0; vtxbuf < NUM_VTXBUF_HANDLES; ++vtxbuf) {
        const AcReal* src = host_mesh.vertex_buffer[vtxbuf];
        AcReal* dst       = grid.submesh.vertex_buffer[vtxbuf];
        //MPI_Scatter(src, 1, input_subarray, dst, 1, output_subarray, 0, acGridMPIComm());

        int nprocs;
        MPI_Comm_size(acGridMPIComm(), &nprocs);
        const uint3_64 p = morton3D(nprocs - 1) + (uint3_64){1, 1, 1};
        int counts[nprocs];
        int displacements[nprocs];
        for (int i = 0; i < nprocs; ++i) {
            counts[i]    = 1;

            const uint3_64 block = morton3D(i);
            const size_t block_offset = block.x * output_nn.x + block.y * output_nn.y * output_nn.x
    * p.x + block.z * output_nn.z * output_nn.x * output_nn.y; displacements[i] = block_offset;
        }

        //MPI_Scatterv(src, counts, displacements, input_subarray, dst, 1, output_subarray, 0,
        //             acGridMPIComm());
        MPI_Scatterv(src, counts, displacements, input_subarray_resized, dst, output_nn.z *
    output_nn.y * output_nn.x, AC_REAL_MPI_TYPE, 0, acGridMPIComm());

    }*/

    MPI_Type_free(&monolithic_subarray);
    MPI_Type_free(&distributed_subarray);
    return acDeviceLoadMesh(grid.device, stream, grid.submesh);
}
/*
// has some illegal memory access issue (create_subarray overwrites block value and loop fails)
AcResult
acGridStoreMesh(const Stream stream, AcMesh* host_mesh)
{
    ERRCHK(grid.initialized);

    const Device device   = grid.device;
    const AcMeshInfo info = acDeviceGetLocalConfig(device);

    acGridSynchronizeStream(stream);
    acDeviceStoreMesh(device, stream, &grid.submesh);
    acDeviceSynchronizeStream(device, stream);

    int pid, nprocs;
    MPI_Comm_rank(acGridMPIComm(), &pid);
    MPI_Comm_size(acGridMPIComm(), &nprocs);

    const int3 pid3d   = getPid3D(pid, grid.decomposition);
    const uint3_64 min = (uint3_64){0, 0, 0};
    const uint3_64 max = morton3D(nprocs - 1); // inclusive

    const int3 rr = (int3){
        (STENCIL_WIDTH - 1) / 2,
        (STENCIL_HEIGHT - 1) / 2,
        (STENCIL_DEPTH - 1) / 2,
    };
    const int3 monolithic_mm  = acConstructInt3Param(AC_nxgrid, AC_nygrid, AC_nzgrid) + 2 * rr;
    const int3 distributed_mm = acConstructInt3Param(AC_mx, AC_my, AC_mz, info);

    for (int block = 0; block < nprocs; ++block) {
        ERRCHK_ALWAYS(block < nprocs);
        int3 distributed_nn     = acConstructInt3Param(AC_nx, AC_ny, AC_nz, info);
        int3 distributed_offset = rr;
        ERRCHK_ALWAYS(block < nprocs);
        if (pid3d.x == min.x) {
            distributed_nn.x += rr.x;
            distributed_offset.x = 0;
        }
        if (pid3d.x == max.x) {
            distributed_nn.x += rr.x;
        }
        if (pid3d.y == min.y) {
            distributed_nn.y += rr.y;
            distributed_offset.y = 0;
        }
        if (pid3d.y == max.y) {
            distributed_nn.y += rr.y;
        }
        if (pid3d.z == min.z) {
            distributed_nn.z += rr.z;
            distributed_offset.z = 0;
        }
        if (pid3d.z == max.z) {
            distributed_nn.z += rr.z;
        }
        // fprintf(stderr, "proc %d, pid %d %d %d, box size %d %d %d\n", pid, pid3d.x, pid3d.y,
        // pid3d.z,
        //         distributed_nn.x, distributed_nn.y, distributed_nn.z);
        ERRCHK_ALWAYS(block < nprocs);
        MPI_Datatype monolithic_subarray;
        const int monolithic_mm_arr[]     = {monolithic_mm.z, monolithic_mm.y, monolithic_mm.x};
        const int monolithic_nn_arr[]     = {distributed_nn.z, distributed_nn.y, distributed_nn.x};
        const int monolithic_offset_arr[] = {distributed_offset.z, distributed_offset.y,
                                             distributed_offset.x};
        ERRCHK_ALWAYS(block < nprocs);
        MPI_Type_create_subarray(3, monolithic_mm_arr, monolithic_nn_arr, monolithic_offset_arr,
                                 MPI_ORDER_C, AC_REAL_MPI_TYPE, &monolithic_subarray);
        ERRCHK_ALWAYS(block < nprocs);
        MPI_Type_commit(&monolithic_subarray);
        ERRCHK_ALWAYS(block < nprocs);

        // const int3 distributed_mm     = acConstructInt3Param(AC_mx, AC_my, AC_mz, info);
        // const int3 distributed_nn     = acConstructInt3Param(AC_nx, AC_ny, AC_nz, info);
        // const int3 distributed_offset = rr;
        ERRCHK_ALWAYS(block < nprocs);
        MPI_Datatype distributed_subarray;
        const int distributed_mm_arr[]     = {distributed_mm.z, distributed_mm.y, distributed_mm.x};
        const int distributed_nn_arr[]     = {distributed_nn.z, distributed_nn.y, distributed_nn.x};
        const int distributed_offset_arr[] = {distributed_offset.z, distributed_offset.y,
                                             distributed_offset.x};
        MPI_Type_create_subarray(3, distributed_mm_arr, distributed_nn_arr, distributed_offset_arr,
                                 MPI_ORDER_C, AC_REAL_MPI_TYPE, &distributed_subarray);
        MPI_Type_commit(&distributed_subarray);

        ERRCHK_ALWAYS(block < nprocs);
        MPI_Request send_reqs[NUM_VTXBUF_HANDLES];
        for (int vtxbuf = 0; vtxbuf < NUM_VTXBUF_HANDLES; ++vtxbuf) {
            ERRCHK_ALWAYS(block < nprocs);
            // send to 0
            if (pid == block)
                MPI_Isend(grid.submesh.vertex_buffer[vtxbuf], 1, distributed_subarray, 0, vtxbuf,
                          acGridMPIComm(), &send_reqs[vtxbuf]);
            ERRCHK_ALWAYS(block < nprocs);
            if (pid == 0) {
                // recv from block
                ERRCHK_ALWAYS(block < nprocs);
                const int3 block_pid3d = getPid3D(block, grid.decomposition);
                const int3 nn          = acConstructInt3Param(AC_nx, AC_ny, AC_nz, info);
                const size_t idx       = acVertexBufferIdx(block_pid3d.x * nn.x, //
                                                           block_pid3d.y * nn.y, //
                                                           block_pid3d.z * nn.z, //
                                                           host_mesh->info);
                MPI_Recv(&host_mesh->vertex_buffer[vtxbuf][idx], 1, monolithic_subarray, block,
                         vtxbuf, acGridMPIComm(), MPI_STATUS_IGNORE);
            }
        }
        if (pid == block)
            MPI_Waitall(NUM_VTXBUF_HANDLES, send_reqs, MPI_STATUSES_IGNORE);
        // Free
        MPI_Type_free(&monolithic_subarray);
        MPI_Type_free(&distributed_subarray);
        // TODO
        // for each block
        //      declare the mapping
        //      all send
        //      if pid == 0
        //          recv
        //
        // could possibly do with scatter/gather but not that important and
        // diminishing returns/no-point finetuning because this is just a
        // simple debug function anyways.
        // More important to focus on getting science/meaningful work done!
    }
}
*/

static void
to_mpi_array_order_c(const int3 v, int arr[3])
{
    arr[0] = v.z;
    arr[1] = v.y;
    arr[2] = v.x;
}

//static void
//print_mpi_array(const char* str, const int arr[3])
//{
//    printf("%s: (%d, %d, %d)\n", str, arr[2], arr[1], arr[0]);
//}

static void
get_subarray(const int pid, //
             int monolithic_mm_arr[3], int monolithic_nn_arr[3],
             int monolithic_offset_arr[3], //
             int distributed_mm_arr[3], int distributed_nn_arr[3], int distributed_offset_arr[3])
{
    int nprocs;
    MPI_Comm_size(acGridMPIComm(), &nprocs);

    const Device device   = grid.device;
    const AcMeshInfo info = acDeviceGetLocalConfig(device);

    const int3 pid3d = getPid3D(pid);
    const Volume rr    = get_ghost_zone_sizes(); 

    const int3 min = (int3){0, 0, 0};
    const int3 max = getPid3D(nprocs - 1); // inclusive

    const int3 base_distributed_nn = to_int3(acGetLocalNN(info));
    int3 distributed_nn     = to_int3(acGetLocalNN(info));
    Volume distributed_offset = rr;

    if (pid3d.x == min.x) {
        distributed_offset.x -= rr.x;
        distributed_nn.x += rr.x;
    }
    if (pid3d.x == max.x) {
        distributed_nn.x += rr.x;
    }
    if (pid3d.y == min.y) {
        distributed_offset.y -= rr.y;
        distributed_nn.y += rr.y;
    }
    if (pid3d.y == max.y) {
        distributed_nn.y += rr.y;
    }
    if (pid3d.z == min.z) {
        distributed_offset.z -= rr.z;
        distributed_nn.z += rr.z;
    }
    if (pid3d.z == max.z) {
        distributed_nn.z += rr.z;
    }

    // Monolithic
    to_mpi_array_order_c(to_int3(acGetGridNN(info) + 2 * rr), monolithic_mm_arr);
    to_mpi_array_order_c(distributed_nn, monolithic_nn_arr);
    to_mpi_array_order_c(to_int3(pid3d * base_distributed_nn + distributed_offset), monolithic_offset_arr);

    // Distributed
    to_mpi_array_order_c(to_int3(acGetLocalMM(info)), distributed_mm_arr);
    to_mpi_array_order_c(distributed_nn, distributed_nn_arr);
    to_mpi_array_order_c(to_int3(distributed_offset), distributed_offset_arr);

    /*
    printf("------\n");
    printf("pid %d\n", pid);
    print_mpi_array("monol mm", monolithic_mm_arr);
    print_mpi_array("monol nn", monolithic_nn_arr);
    print_mpi_array("monol os", monolithic_offset_arr);

    print_mpi_array("distr mm", distributed_mm_arr);
    print_mpi_array("distr nn", distributed_nn_arr);
    print_mpi_array("distr os", distributed_offset_arr);
    printf("------\n");
    */
}

/**
static void
get_subarray(const int pid, //
             int monolithic_mm_arr[3], int monolithic_nn_arr[3],
             int monolithic_offset_arr[3], //
             int distributed_mm_arr[3], int distributed_nn_arr[3], int distributed_offset_arr[3])
{

    const Device device   = grid.device;
    const AcMeshInfo info = acDeviceGetLocalConfig(device);

    const int3 pid3d = getPid3D(pid);
    const int3 rr = get_ghost_zone_sizes();

    const int3 min = (int3){0, 0, 0};
    const int3 max = getPid3D(ac_nprocs() - 1); // inclusive
    const int3 base_distributed_nn = acGetLocalNN(acGridGetLocalMeshInfo());
    int3 distributed_nn     = acGetLocalNN(acGridGetLocalMeshInfo());
    int3 distributed_offset = rr;

    if (pid3d.x == min.x) {
        distributed_offset.x -= rr.x;
        distributed_nn.x += rr.x;
    }
    if (pid3d.x == max.x) {
        distributed_nn.x += rr.x;
    }
    if (pid3d.y == min.y) {
        distributed_offset.y -= rr.y;
        distributed_nn.y += rr.y;
    }
    if (pid3d.y == max.y) {
        distributed_nn.y += rr.y;
    }
    if (pid3d.z == min.z) {
        distributed_offset.z -= rr.z;
        distributed_nn.z += rr.z;
    }
    if (pid3d.z == max.z) {
        distributed_nn.z += rr.z;
    }

    // Monolithic
    to_mpi_array_order_c(get_global_nn() + 2 * rr, monolithic_mm_arr);
    to_mpi_array_order_c(distributed_nn, monolithic_nn_arr);
    to_mpi_array_order_c(pid3d * base_distributed_nn + distributed_offset, monolithic_offset_arr);

    // Distributed
    to_mpi_array_order_c(acGetLocalMM(info), distributed_mm_arr);
    to_mpi_array_order_c(distributed_nn, distributed_nn_arr);
    to_mpi_array_order_c(distributed_offset, distributed_offset_arr);

    //printf("------\n");
    //printf("pid %d\n", pid);
    //print_mpi_array("monol mm", monolithic_mm_arr);
    //print_mpi_array("monol nn", monolithic_nn_arr);
    //print_mpi_array("monol os", monolithic_offset_arr);

    //print_mpi_array("distr mm", distributed_mm_arr);
    //print_mpi_array("distr nn", distributed_nn_arr);
    //print_mpi_array("distr os", distributed_offset_arr);
    //printf("------\n");
}
**/

// With ghost zone
//
AcResult
acGridLoadMesh(const Stream stream, const AcMesh host_mesh)
{
    ERRCHK(grid.initialized);
    acGridSynchronizeStream(stream);

    const int pid = ac_pid();
    const int nprocs = ac_nprocs();
    // Datatype:
    // 1) All processes: Local subarray (sending)
    //  1.1) function that takes the pid and outputs the local subarray
    // 2) Root process:  Global array (receiving)
    // 3) Root process:  Local subarrays for all procs (same as used for sending)

    // Receive the local subarray
    MPI_Request recv_reqs[NUM_VTXBUF_HANDLES];
    for (int i = 0; i < NUM_VTXBUF_HANDLES; ++i) recv_reqs[i] = MPI_REQUEST_NULL;
    for (int vtxbuf = 0; vtxbuf < NUM_VTXBUF_HANDLES; ++vtxbuf) {
    	if(!vtxbuf_is_alive[vtxbuf] || vtxbuf_is_device_only[vtxbuf]) continue;
        int monolithic_mm[3], monolithic_nn[3], monolithic_offset[3];
        int distributed_mm[3], distributed_nn[3], distributed_offset[3];
        get_subarray(pid, monolithic_mm, monolithic_nn,
                     monolithic_offset, //
                     distributed_mm, distributed_nn, distributed_offset);

        MPI_Datatype distributed_subarray;
        MPI_Type_create_subarray(3, distributed_mm, distributed_nn, distributed_offset, MPI_ORDER_C,
                                 AC_REAL_MPI_TYPE, &distributed_subarray);
        MPI_Type_commit(&distributed_subarray);

        MPI_Irecv(grid.submesh.vertex_buffer[vtxbuf], 1, distributed_subarray, 0, vtxbuf,
                  acGridMPIComm(), &recv_reqs[vtxbuf]);

        MPI_Type_free(&distributed_subarray);
    }

    if (pid == 0) {
        for (int tgt = 0; tgt < nprocs; ++tgt) {
            for (int vtxbuf = 0; vtxbuf < NUM_VTXBUF_HANDLES; ++vtxbuf) {
    	        if(!vtxbuf_is_alive[vtxbuf] || vtxbuf_is_device_only[vtxbuf]) continue;
                int monolithic_mm[3], monolithic_nn[3], monolithic_offset[3];
                int distributed_mm[3], distributed_nn[3], distributed_offset[3];
                get_subarray(tgt, monolithic_mm, monolithic_nn,
                             monolithic_offset, //
                             distributed_mm, distributed_nn, distributed_offset);

                MPI_Datatype monolithic_subarray;
                MPI_Type_create_subarray(3, monolithic_mm, monolithic_nn, monolithic_offset,
                                         MPI_ORDER_C, AC_REAL_MPI_TYPE, &monolithic_subarray);
                MPI_Type_commit(&monolithic_subarray);

                MPI_Send(host_mesh.vertex_buffer[vtxbuf], 1, monolithic_subarray, tgt, vtxbuf,
                         acGridMPIComm());

                MPI_Type_free(&monolithic_subarray);
            }
        }
    }
    MPI_Waitall(NUM_VTXBUF_HANDLES, recv_reqs, MPI_STATUSES_IGNORE);

    // TODO: Should apply halo exchange here without touching the ghost zones, how?
    // Currently the users need to update halos after each load, which is error-prone
    // acDeviceLoadMesh(grid.device, stream, grid.submesh);
    // return acGridPeriodicBoundconds(STREAM_DEFAULT);

    return acDeviceLoadMesh(grid.device, stream, grid.submesh);
}


// Working with ghost zone
AcResult
acGridStoreMesh(const Stream stream, AcMesh* host_mesh)
{
    ERRCHK(grid.initialized);
    acGridSynchronizeStream(stream);
    acDeviceStoreMesh(grid.device, stream, &grid.submesh);
    acDeviceSynchronizeStream(grid.device, stream);

    const int pid = ac_pid();
    // Datatype:
    // 1) All processes: Local subarray (sending)
    //  1.1) function that takes the pid and outputs the local subarray
    // 2) Root process:  Global array (receiving)
    // 3) Root process:  Local subarrays for all procs (same as used for sending)

    // Send the local subarray
    MPI_Request send_reqs[NUM_VTXBUF_HANDLES];
    for (int i = 0; i < NUM_VTXBUF_HANDLES; ++i) send_reqs[i] = MPI_REQUEST_NULL;
    for (int vtxbuf = 0; vtxbuf < NUM_VTXBUF_HANDLES; ++vtxbuf) {
    	if(!vtxbuf_is_alive[vtxbuf] || vtxbuf_is_device_only[vtxbuf]) continue;
        int monolithic_mm[3], monolithic_nn[3], monolithic_offset[3];
        int distributed_mm[3], distributed_nn[3], distributed_offset[3];
        get_subarray(pid, monolithic_mm, monolithic_nn,
                     monolithic_offset, //
                     distributed_mm, distributed_nn, distributed_offset);

        MPI_Datatype distributed_subarray;
        MPI_Type_create_subarray(3, distributed_mm, distributed_nn, distributed_offset, MPI_ORDER_C,
                                 AC_REAL_MPI_TYPE, &distributed_subarray);
        MPI_Type_commit(&distributed_subarray);

        MPI_Isend(grid.submesh.vertex_buffer[vtxbuf], 1, distributed_subarray, 0, vtxbuf,
                  acGridMPIComm(), &send_reqs[vtxbuf]);

        MPI_Type_free(&distributed_subarray);
    }

    if (pid == 0) {
        for (int src = 0; src < ac_nprocs(); ++src) {
            for (int vtxbuf = 0; vtxbuf < NUM_VTXBUF_HANDLES; ++vtxbuf) {
    	        if(!vtxbuf_is_alive[vtxbuf] || vtxbuf_is_device_only[vtxbuf]) continue;
                int monolithic_mm[3], monolithic_nn[3], monolithic_offset[3];
                int distributed_mm[3], distributed_nn[3], distributed_offset[3];
                get_subarray(src, monolithic_mm, monolithic_nn,
                             monolithic_offset, //
                             distributed_mm, distributed_nn, distributed_offset);

                MPI_Datatype monolithic_subarray;
                MPI_Type_create_subarray(3, monolithic_mm, monolithic_nn, monolithic_offset,
                                         MPI_ORDER_C, AC_REAL_MPI_TYPE, &monolithic_subarray);
                MPI_Type_commit(&monolithic_subarray);

                MPI_Recv(host_mesh->vertex_buffer[vtxbuf], 1, monolithic_subarray, src, vtxbuf,
                         acGridMPIComm(), MPI_STATUS_IGNORE);

                MPI_Type_free(&monolithic_subarray);
            }
        }
    }
    MPI_Waitall(NUM_VTXBUF_HANDLES, send_reqs, MPI_STATUSES_IGNORE);

    return AC_SUCCESS;
}


AcResult
acGridStoreMeshWorking(const Stream stream, AcMesh* host_mesh)
{
    ERRCHK(grid.initialized);
    acGridSynchronizeStream(stream);
    acDeviceStoreMesh(grid.device, stream, &grid.submesh);
    acDeviceSynchronizeStream(grid.device, stream);

    /*
    const Device device   = grid.device;
    const AcMeshInfo info = acDeviceGetLocalConfig(device);

    const int3 rr = (int3){
        (STENCIL_WIDTH - 1) / 2,
        (STENCIL_HEIGHT - 1) / 2,
        (STENCIL_DEPTH - 1) / 2,
    };
    const int3 input_nn     = acConstructInt3Param(AC_nxgrid, AC_nygrid, AC_nzgrid); // Without halo
    const int3 input_mm     = input_nn + 2 * rr;
    const int3 input_offset = rr; //  + info[AC_multigpu_offset];

    MPI_Datatype input_subarray;
    const int input_mm_arr[]     = {input_mm.z, input_mm.y, input_mm.x};
    const int input_nn_arr[]     = {input_nn.z, input_nn.y, input_nn.x};
    const int input_offset_arr[] = {input_offset.z, input_offset.y, input_offset.x};
    MPI_Type_create_subarray(3, input_mm_arr, input_nn_arr, input_offset_arr, MPI_ORDER_C,
                             AC_REAL_MPI_TYPE, &input_subarray);
    MPI_Type_commit(&input_subarray);

    const int3 output_nn     = acConstructInt3Param(AC_nx, AC_ny, AC_nz, info);
    const int3 output_mm     = acConstructInt3Param(AC_mx, AC_my, AC_mz, info);
    const int3 output_offset = rr;

    MPI_Datatype output_subarray;
    const int output_mm_arr[]     = {output_mm.z, output_mm.y, output_mm.x};
    const int output_nn_arr[]     = {output_nn.z, output_nn.y, output_nn.x};
    const int output_offset_arr[] = {output_offset.z, output_offset.y, output_offset.x};
    MPI_Type_create_subarray(3, output_mm_arr, output_nn_arr, output_offset_arr, MPI_ORDER_C,
                             AC_REAL_MPI_TYPE, &output_subarray);
    MPI_Type_commit(&output_subarray);

    // Scatter host_mesh from proc 0
    for (int vtxbuf = 0; vtxbuf < NUM_VTXBUF_HANDLES; ++vtxbuf) {
        const AcReal* src = grid.submesh.vertex_buffer[vtxbuf];
        AcReal* dst       = host_mesh->vertex_buffer[vtxbuf];
        MPI_Gather(src, 1, output_subarray, dst, 1, input_subarray, 0, acGridMPIComm());
        // MPI_Scatter(src, 1, input_subarray, dst, 1, output_subarray, 0, acGridMPIComm());
    }

    MPI_Type_free(&input_subarray);
    MPI_Type_free(&output_subarray);
    */


    const Volume rr = get_ghost_zone_sizes();
    const Volume monolithic_mm     = get_global_nn() + 2 * rr;
    const Volume monolithic_nn     = acGetLocalNN(acGridGetLocalMeshInfo());
    const Volume monolithic_offset = rr;

    MPI_Datatype monolithic_subarray;
    const int monolithic_mm_arr[]     = {(int)monolithic_mm.z, (int)monolithic_mm.y, (int)monolithic_mm.x};
    const int monolithic_nn_arr[]     = {(int)monolithic_nn.z, (int)monolithic_nn.y, (int)monolithic_nn.x};
    const int monolithic_offset_arr[] = {(int)monolithic_offset.z, (int)monolithic_offset.y,
                                         (int)monolithic_offset.x};
    MPI_Type_create_subarray(3, monolithic_mm_arr, monolithic_nn_arr, monolithic_offset_arr,
                             MPI_ORDER_C, AC_REAL_MPI_TYPE, &monolithic_subarray);
    MPI_Type_commit(&monolithic_subarray);

    const Volume distributed_mm     = acGetLocalMM(acGridGetLocalMeshInfo());
    const Volume distributed_nn     = acGetLocalNN(acGridGetLocalMeshInfo());
    const Volume distributed_offset = rr;

    MPI_Datatype distributed_subarray;
    const int distributed_mm_arr[]     = {(int)distributed_mm.z, (int)distributed_mm.y, (int)distributed_mm.x};
    const int distributed_nn_arr[]     = {(int)distributed_nn.z, (int)distributed_nn.y, (int)distributed_nn.x};
    const int distributed_offset_arr[] = {(int)distributed_offset.z, (int)distributed_offset.y,
                                          (int)distributed_offset.x};
    MPI_Type_create_subarray(3, distributed_mm_arr, distributed_nn_arr, distributed_offset_arr,
                             MPI_ORDER_C, AC_REAL_MPI_TYPE, &distributed_subarray);
    MPI_Type_commit(&distributed_subarray);

    const int pid = ac_pid();

    MPI_Request send_reqs[NUM_VTXBUF_HANDLES];
    for (int vtxbuf = 0; vtxbuf < NUM_VTXBUF_HANDLES; ++vtxbuf) {
        MPI_Isend(grid.submesh.vertex_buffer[vtxbuf], 1, distributed_subarray, 0, vtxbuf,
                  acGridMPIComm(), &send_reqs[vtxbuf]);
        if (pid == 0) {
            for (int tgt = 0; tgt < ac_nprocs(); ++tgt) {
                const int3 tgt_pid3d = getPid3D(tgt);
                const size_t idx     = acVertexBufferIdx(tgt_pid3d.x * distributed_nn.x, //
                                                         tgt_pid3d.y * distributed_nn.y, //
                                                         tgt_pid3d.z * distributed_nn.z, //
                                                         host_mesh->info,
							 VertexBufferHandle(vtxbuf)
							 );
                MPI_Recv(&host_mesh->vertex_buffer[vtxbuf][idx], 1, monolithic_subarray, tgt,
                         vtxbuf, acGridMPIComm(), MPI_STATUS_IGNORE);
            }
        }
    }
    MPI_Waitall(NUM_VTXBUF_HANDLES, send_reqs, MPI_STATUSES_IGNORE);
    /*
        Strategy:
            1) Select a subarray from the input mesh
            2) Select a subarray from the output mesh
            3) Scatter

        Notes:
            1) Check that subarray divisible by number of procs (required in init iirc)
    MPI_Datatype input_subarray_resized;
    MPI_Type_create_resized(input_subarray, 0, sizeof(AcReal), &input_subarray_resized);
    MPI_Type_commit(&input_subarray_resized);

    // Scatter host_mesh from proc 0
    for (int vtxbuf = 0; vtxbuf < NUM_VTXBUF_HANDLES; ++vtxbuf) {
        const AcReal* src = host_mesh.vertex_buffer[vtxbuf];
        AcReal* dst       = grid.submesh.vertex_buffer[vtxbuf];
        //MPI_Scatter(src, 1, input_subarray, dst, 1, output_subarray, 0, acGridMPIComm());

        int nprocs;
        MPI_Comm_size(acGridMPIComm(), &nprocs);
        const uint3_64 p = morton3D(nprocs - 1) + (uint3_64){1, 1, 1};
        int counts[nprocs];
        int displacements[nprocs];
        for (int i = 0; i < nprocs; ++i) {
            counts[i]    = 1;

            const uint3_64 block = morton3D(i);
            const size_t block_offset = block.x * output_nn.x + block.y * output_nn.y * output_nn.x
    * p.x + block.z * output_nn.z * output_nn.x * output_nn.y; displacements[i] = block_offset;
        }

        //MPI_Scatterv(src, counts, displacements, input_subarray, dst, 1, output_subarray, 0,
        //             acGridMPIComm());
        MPI_Scatterv(src, counts, displacements, input_subarray_resized, dst, output_nn.z *
    output_nn.y * output_nn.x, AC_REAL_MPI_TYPE, 0, acGridMPIComm());

    }*/

    MPI_Type_free(&monolithic_subarray);
    MPI_Type_free(&distributed_subarray);

    return AC_SUCCESS;
}
AcResult
acGridLoadMeshOld(const Stream stream, const AcMesh host_mesh)
{
    ERRCHK(grid.initialized);
    acGridSynchronizeStream(stream);
    acGridDiskAccessSync(); // Note: syncs all streams

#if AC_VERBOSE
    printf("Distributing mesh...\n");
    fflush(stdout);
#endif

    const int pid = ac_pid();
    ERRCHK_ALWAYS(&grid.submesh);

    // Submesh nn
    const Volume nn = acGetLocalNN(acGridGetLocalMeshInfo());

    // Send to self
    auto ghost_zone_sizes = get_ghost_zone_sizes();
    if (pid == 0) {
        for (int vtxbuf = 0; vtxbuf < NUM_VTXBUF_HANDLES; ++vtxbuf) {
            // For pencils
            for (size_t k = ghost_zone_sizes.z; k < ghost_zone_sizes.z + nn.z; ++k) {
                for (size_t j = ghost_zone_sizes.y; j < ghost_zone_sizes.y + nn.y; ++j) {
                    const int i       = ghost_zone_sizes.x;
                    const int count   = nn.x;
                    const int src_idx = acVertexBufferIdx(i, j, k, host_mesh.info,VertexBufferHandle(vtxbuf));
							 
                    const int dst_idx = acVertexBufferIdx(i, j, k, grid.submesh.info,VertexBufferHandle(vtxbuf));
                    memcpy(&grid.submesh.vertex_buffer[vtxbuf][dst_idx], //
                           &host_mesh.vertex_buffer[vtxbuf][src_idx],    //
                           count * sizeof(host_mesh.vertex_buffer[i][0]));
                }
            }
        }
    }

    for (int vtxbuf = 0; vtxbuf < NUM_VTXBUF_HANDLES; ++vtxbuf) {
        // For pencils
        for (size_t k = ghost_zone_sizes.z; k < ghost_zone_sizes.z + nn.z; ++k) {
            for (size_t j = ghost_zone_sizes.y; j < ghost_zone_sizes.y + nn.y; ++j) {
                const int i     = ghost_zone_sizes.x;
                const int count = nn.x;

                if (pid != 0) {
                    const int dst_idx = acVertexBufferIdx(i, j, k, grid.submesh.info,VertexBufferHandle(vtxbuf));
                    // Recv
                    MPI_Status status;
                    MPI_Recv(&grid.submesh.vertex_buffer[vtxbuf][dst_idx], count, AC_REAL_MPI_TYPE,
                             0, 0, astaroth_comm, &status);
                }
                else {
                    for (int tgt_pid = 1; tgt_pid < ac_nprocs(); ++tgt_pid) {
                        const int3 tgt_pid3d = getPid3D(tgt_pid);
                        const int src_idx    = acVertexBufferIdx(i + tgt_pid3d.x * nn.x, //
                                                                 j + tgt_pid3d.y * nn.y, //
                                                                 k + tgt_pid3d.z * nn.z, //
                                                                 host_mesh.info,
								 VertexBufferHandle(vtxbuf)
								 );

                        // Send
                        MPI_Send(&host_mesh.vertex_buffer[vtxbuf][src_idx], count, AC_REAL_MPI_TYPE,
                                 tgt_pid, 0, astaroth_comm);
                    }
                }
            }
        }
    }

    acDeviceLoadMesh(grid.device, stream, grid.submesh);
    return AC_SUCCESS;
}

// TODO: do with packed data
AcResult
acGridStoreMeshAA(const Stream stream, AcMesh* host_mesh)
{
    ERRCHK(grid.initialized);
    acGridSynchronizeStream(stream);
    acGridDiskAccessSync(); // Note: syncs all streams

    acDeviceStoreMesh(grid.device, stream, &grid.submesh);
    acGridSynchronizeStream(stream);

#if AC_VERBOSE
    printf("Gathering mesh...\n");
    fflush(stdout);
#endif

    const int pid = ac_pid();

    if (pid == 0)
        ERRCHK_ALWAYS(host_mesh);

    // Submesh nn and mm
    const Volume nn = acGetLocalNN(acGridGetLocalMeshInfo());
    const Volume mm = acGetLocalMM(acGridGetLocalMeshInfo());

    // Send to self
    if (pid == 0) {
        for (int vtxbuf = 0; vtxbuf < NUM_VTXBUF_HANDLES; ++vtxbuf) {
            // For pencils
            for (size_t k = 0; k < mm.z; ++k) {
                for (size_t j = 0; j < mm.y; ++j) {
                    const int i       = 0;
                    const int count   = mm.x;
                    const int src_idx = acVertexBufferIdx(i, j, k, grid.submesh.info,VertexBufferHandle(vtxbuf));
                    const int dst_idx = acVertexBufferIdx(i, j, k, host_mesh->info,VertexBufferHandle(vtxbuf));
                    memcpy(&host_mesh->vertex_buffer[vtxbuf][dst_idx],   //
                           &grid.submesh.vertex_buffer[vtxbuf][src_idx], //
                           count * sizeof(grid.submesh.vertex_buffer[i][0]));
                }
            }
        }
    }

    for (int vtxbuf = 0; vtxbuf < NUM_VTXBUF_HANDLES; ++vtxbuf) {
        // For pencils
        for (size_t k = 0; k < mm.z; ++k) {
            for (size_t j = 0; j < mm.y; ++j) {
                const int i     = 0;
                const int count = mm.x;

                if (pid == 0) {
                    for (int tgt_pid = 1; tgt_pid < ac_nprocs(); ++tgt_pid) {
                        const int3 tgt_pid3d = getPid3D(tgt_pid);
                        const int dst_idx    = acVertexBufferIdx(i + tgt_pid3d.x * nn.x, //
                                                                 j + tgt_pid3d.y * nn.y, //
                                                                 k + tgt_pid3d.z * nn.z, //
                                                                 host_mesh->info,
								 VertexBufferHandle(vtxbuf)
								 );

                        // Recv
                        MPI_Status status;
                        MPI_Recv(&host_mesh->vertex_buffer[vtxbuf][dst_idx], count,
                                 AC_REAL_MPI_TYPE, tgt_pid, 0, astaroth_comm, &status);
                    }
                }
                else {
                    // Send
                    const int src_idx = acVertexBufferIdx(i, j, k, grid.submesh.info,VertexBufferHandle(vtxbuf));
                    MPI_Send(&grid.submesh.vertex_buffer[vtxbuf][src_idx], count, AC_REAL_MPI_TYPE,
                             0, 0, astaroth_comm);
                }
            }
        }
    }
    MPI_Barrier(astaroth_comm);

    return AC_SUCCESS;
}

AcTaskGraph*
acGridGetDefaultTaskGraph()
{
    ERRCHK(grid.initialized);
    if(!grid.submesh.info[AC_fully_periodic_grid])
	    fatal("%s","Default taskgraph assumes fully periodic grid!\n");
    return grid.default_tasks.get();
}

static void
check_ops(const std::vector<AcTaskDefinition> ops)
{
    if (ops.size() == 0) {
	if(ac_pid() == 0) WARNING("\nUnusual task graph {}:\n - Task graph is empty.\n")
	return;
    }

    bool found_halo_exchange        = false;
    unsigned int boundaries_defined = 0x00;
    // bool found_compute              = false;

    bool boundary_condition_before_halo_exchange = false;
    bool compute_before_halo_exchange            = false;
    bool compute_before_boundary_condition       = false;

    [[maybe_unused]] bool error   = false;
    [[maybe_unused]] bool warning = false;

    std::string task_graph_repr = "{";
    const auto inactive = acGridGetLocalMeshInfo()[AC_dimension_inactive];

    for (size_t i = 0; i < ops.size() ; i++) {
        AcTaskDefinition op = ops[i];
        switch (op.task_type) {
        case TASKTYPE_HALOEXCHANGE:
            found_halo_exchange = true;
            task_graph_repr += "HaloExchange,";
            break;
        case TASKTYPE_BOUNDCOND:
            if (!found_halo_exchange) {
                boundary_condition_before_halo_exchange = true;
                error                                   = true;
            }
            boundaries_defined |= (unsigned int)op.boundary;
            task_graph_repr += "BoundCond,";
            break;
        case TASKTYPE_COMPUTE:
            if (!found_halo_exchange) {
                compute_before_halo_exchange = true;
                warning                      = true;
            }
            if (boundaries_defined != get_full_boundary()) {
                compute_before_boundary_condition = true;
                warning                           = true;
            }
            // found_compute = true;
            task_graph_repr += "Compute,";
            break;
        case TASKTYPE_REDUCE:
          task_graph_repr += "Reduce,";
	  break;
        case TASKTYPE_RAY_REDUCE:
          task_graph_repr += "RayReduce,";
	  break;
        case TASKTYPE_RAY_UPDATE:
          task_graph_repr += "RayUpdate,";
	  break;
	}
    }

    task_graph_repr += "}";

    std::string msg = "";

    if (!found_halo_exchange) {
        msg += " - No halo exchange defined in task graph.\n";
        error = true;
    }

    if (boundaries_defined != get_full_boundary()) {
        error = true;
    }
    if ((boundaries_defined & BOUNDARY_X_TOP) != BOUNDARY_X_TOP && !inactive.x) {
        msg += " - Boundary conditions not defined for top X boundary.\n";
    }
    if ((boundaries_defined & BOUNDARY_X_BOT) != BOUNDARY_X_BOT && !inactive.x) {
        msg += " - Boundary conditions not defined for bottom X boundary.\n";
    }
    if ((boundaries_defined & BOUNDARY_Y_TOP) != BOUNDARY_Y_TOP && !inactive.y) {
        msg += " - Boundary conditions not defined for top Y boundary.\n";
    }
    if ((boundaries_defined & BOUNDARY_Y_BOT) != BOUNDARY_Y_BOT && !inactive.y) {
        msg += " - Boundary conditions not defined for bottom Y boundary.\n";
    }
    if ((boundaries_defined & BOUNDARY_Z_TOP) != BOUNDARY_Z_TOP && !inactive.z) {
        msg += " - Boundary conditions not defined for top Z boundary.\n";
    }
    if ((boundaries_defined & BOUNDARY_Z_BOT) != BOUNDARY_Z_BOT && !inactive.z) {
        msg += " - Boundary conditions not defined for bottom Z boundary.\n";
    }

    // This warning is probably unnecessary
    /*
      if (!found_compute) {
          //msg += " - No compute kernel defined in task graph.\n";
          //warning = true;
      }
      */

    if (found_halo_exchange && boundary_condition_before_halo_exchange) {
        msg += " - Boundary condition before halo exchange. Halo exchange must come first.\n";
    }
    if (boundaries_defined == BOUNDARY_XYZ && compute_before_boundary_condition) {
        msg += " - Compute ordered before boundary conditions. Boundary conditions must usually be "
               "resolved before running kernels.\n";
    }
    if (found_halo_exchange && compute_before_halo_exchange) {
        msg += " - Compute ordered before halo exchange. Halo exchange must usually be performed "
               "before running kernels.\n";
    }

#if AC_VERBOSE
    if (error && NGHOST>0) {
        // ERROR(("\nIncorrect task graph " + task_graph_repr + ":\n" + msg).c_str())
        WARNING(("\nIncorrect task graph " + task_graph_repr + ":\n" + msg).c_str())
    }
    if (warning && NGHOST>0) {
        WARNING(("\nUnusual task graph " + task_graph_repr + ":\n" + msg).c_str())
    }
#endif
}

static AcReal3 
get_spacings()
{
	return acGridGetLocalMeshInfo()[AC_ds];
}


AcTaskGraph*
acGridBuildTaskGraphWithBounds(const AcTaskDefinition ops_in[], const size_t n_ops, const Volume start_in, const Volume end_in, const bool globally_imposed_bcs)
{ 

    // ERRCHK(grid.initialized);
    std::vector<AcTaskDefinition> ops{};
    //TP: insert reduce tasks in between of tasks
    //If kernel A outputs P(profile or scalar to be reduced) then a reduce task reducing P should be inserted
    for(size_t i = 0; i < n_ops; ++i)
    {
	    auto op = ops_in[i];
	    ops.push_back(op);
	    if(op.task_type == TASKTYPE_COMPUTE && (op.num_profiles_reduce_out > 0 || op.num_outputs_out > 0))
	    {
	  		auto reduce_op = op;
			reduce_op.task_type = TASKTYPE_REDUCE;

	  		reduce_op.profiles_in      = op.profiles_reduce_out;
	  		reduce_op.num_profiles_in  = op.num_profiles_reduce_out;
	  		reduce_op.profiles_reduce_out     = op.profiles_reduce_out;
	  		reduce_op.num_profiles_reduce_out = op.num_profiles_reduce_out;

			reduce_op.num_outputs_in  = op.num_outputs_out;
			reduce_op.num_outputs_out = op.num_outputs_out;
			reduce_op.outputs_in      = op.outputs_out;
			reduce_op.outputs_out     = op.outputs_out;

			reduce_op.start = op.start;
			reduce_op.end   = op.end;
			reduce_op.given_launch_bounds = op.given_launch_bounds;
			ops.push_back(reduce_op);
	    }
    }
    std::vector<AcKernel> kernels{};
    for(size_t i = 0; i < ops.size(); ++i) kernels.push_back(AC_NULL_KERNEL);
    for(size_t i = 0; i < ops.size(); ++i)
    {
	    const auto& op = ops[i];
	    if(op.task_type == TASKTYPE_COMPUTE)
	    {
		    kernels[i] = op.kernel_enum;
	    }
    }
    const auto& kernel_computes_profile_on_halos = compute_kernel_call_computes_profile_across_halos(kernels);
    for(size_t i = 0; i < ops.size(); ++i)
    {
	    ops[i].computes_on_halos = BOUNDARY_NONE;
	    for(int profile = 0; profile < NUM_PROFILES; ++profile)
	    	ops[i].computes_on_halos = (AcBoundary)((int)ops[i].computes_on_halos | kernel_computes_profile_on_halos[i][profile]);
	    const auto& boundary = ops[i].computes_on_halos;
	    int num_boundaries_included = 0;
	    if(boundary & BOUNDARY_X) ++num_boundaries_included;
	    if(boundary & BOUNDARY_Y) ++num_boundaries_included;
	    if(boundary & BOUNDARY_Z) ++num_boundaries_included;
	    if(num_boundaries_included > 1) fatal("%s","For now kernels can include only a single halo region in their input\n");
    }
    for(size_t i = 0; i < ops.size(); ++i)
    {
	    const auto& op = ops[i];
	    const AcKernel kernel = op.kernel_enum;
	    bool calls_stencil = false;
	    for(size_t field = 0; field < NUM_VTXBUF_HANDLES; ++field)
		    calls_stencil |= grid.kernel_analysis_info[kernel].field_has_stencil_op[field];
	    if(calls_stencil && op.computes_on_halos != BOUNDARY_NONE)
		    fatal("Kernel %s should be computed on halos but also uses Stencils!!\n",kernel_names[kernel]);
    }

    int rank;
    MPI_Comm_rank(astaroth_comm, &rank);
    int comm_size;
    MPI_Comm_size(astaroth_comm, &comm_size);

    check_ops(ops);
    acVerboseLogFromRootProc(rank, "acGridBuildTaskGraph: Allocating task graph\n");

    AcTaskGraph* graph = new AcTaskGraph();

    graph->periodic_boundaries = BOUNDARY_NONE;

    graph->halo_tasks.reserve(ops.size() * Region::n_halo_regions);
    graph->all_tasks.reserve(ops.size() * max(Region::n_halo_regions, Region::n_comp_regions));

    // Create tasks for each operation & store indices to ranges of tasks belonging to operations
    std::vector<size_t> op_indices;
    op_indices.reserve(ops.size());

    const AcGridInfo grid_info = {grid.nn, get_global_nn()*get_spacings()};



    uint3_64 decomp = grid.decomposition;
    int3 pid3d      = getPid3D(rank);
    Device device   = grid.device;

    auto boundary_normal = [&decomp, &pid3d](int tag, const AcBoundary boundary) -> int3 {
        int3 neighbor = pid3d + Region::tag_to_id(tag);
        if (neighbor.z == -1 && (boundary & BOUNDARY_Z_BOT) != 0) {
            return int3{0, 0, -1};
        }
        else if (neighbor.z == (int)decomp.z && (boundary & BOUNDARY_Z_TOP) != 0) {
            return int3{0, 0, 1};
        }
        else if (neighbor.y == -1 && (boundary & BOUNDARY_Y_BOT) != 0) {
            return int3{0, -1, 0};
        }
        else if (neighbor.y == (int)decomp.y && (boundary & BOUNDARY_Y_TOP) != 0) {
            return int3{0, 1, 0};
        }
        else if (neighbor.x == -1 && (boundary & BOUNDARY_X_BOT) != 0) {
            return int3{-1, 0, 0};
        }
        else if (neighbor.x == (int)decomp.x && (boundary & BOUNDARY_X_TOP) != 0) {
            return int3{1, 0, 0};
        }
        else {
            // Something went wrong, this tag does not identify a boundary region.
            return int3{0, 0, 0};
        }
    };
    

    // The tasks start at different offsets from the beginning of the iteration
    // this array of bools keep track of that state
    std::array<bool, NUM_VTXBUF_HANDLES+NUM_PROFILES> swap_offset{};
    //int num_comp_tasks = 0;
    std::array<int,NUM_FIELDS> fields_already_depend_on_boundaries{};
    const auto compute_task_poststep = [&](const auto& op, const auto& task)
    {
        //done here since we want to write only to out not to in what launching the taskgraph would do
        //always remember to call the loader since otherwise might not be safe to execute taskgraph
        op.load_kernel_params_func->loader({acDeviceGetKernelInputParams(grid.device),grid.device, 0, {}, {}, op.kernel_enum});

	const int3 old_tpb = acReadOptimTBConfig(op.kernel_enum,task->output_region.dims,to_volume(acGridGetLocalMeshInfo()[AC_thread_block_loop_factors]));
        acDeviceSetReduceOffset(grid.device, op.kernel_enum, task->output_region.position, task->output_region.position + task->output_region.dims);
        //acDeviceLaunchKernel(grid.device, STREAM_DEFAULT, op.kernel_enum, task->output_region.position, task->output_region.position + task->output_region.dims);
        fields_already_depend_on_boundaries = get_fields_kernel_depends_on_boundaries(op.kernel_enum,fields_already_depend_on_boundaries);
    	//make sure after autotuning that out is 0
	if(old_tpb == (int3){-1,-1,-1})
	{
    		AcMeshDims dims = acGetMeshDims(acGridGetLocalMeshInfo());
    		acGridLaunchKernel(STREAM_DEFAULT, AC_BUILTIN_RESET, dims.n0,dims.n1);
    		acGridSynchronizeStream(STREAM_ALL);
	}

    };


    acVerboseLogFromRootProc(rank, "acGridBuildTaskGraph: Creating tasks: %lu ops\n", ops.size());
    for (size_t i = 0; i < ops.size(); i++) {
        acVerboseLogFromRootProc(rank, "acGridBuildTaskGraph: Creating tasks for op %lu\n", i);
        auto op = ops[i];

        const Volume dims = 
			    op.given_launch_bounds ?
			    op.end-op.start :
			    end_in-start_in;
	const Volume start =
				op.given_launch_bounds ?
				op.start : start_in;

        op_indices.push_back(graph->all_tasks.size());

        if (op.task_type == TASKTYPE_BOUNDCOND && op.kernel_enum == BOUNDCOND_PERIODIC) {
            graph->periodic_boundaries = (AcBoundary)(graph->periodic_boundaries | op.boundary);
        }
        switch (op.task_type) {

        case TASKTYPE_RAY_UPDATE: {
            acVerboseLogFromRootProc(rank, "Creating ray updates\n");
	    std::vector<Field> fields_out(op.fields_out,op.fields_out+op.num_fields_out);
	    std::vector<Field> fields_in(op.fields_in,op.fields_in+op.num_fields_in);
	    if(
		  (op.ray_direction.x == -1 && ((op.boundary & BOUNDARY_X_BOT) != 0))
		||(op.ray_direction.x == +1 && ((op.boundary & BOUNDARY_X_TOP) != 0))
	      )
	    {
		    int3 boundary = (int3){op.ray_direction.x,0,0};
		    auto task = ComputeTask::RayUpdate(op,i,boundary,op.ray_direction,device,swap_offset,fields_already_depend_on_boundaries);
              	    graph->all_tasks.push_back(task);
	      	    compute_task_poststep(op,task);
	    }
	    if(
		  (op.ray_direction.y == -1 && ((op.boundary & BOUNDARY_Y_BOT) != 0))
		||(op.ray_direction.y == +1 && ((op.boundary & BOUNDARY_Y_TOP) != 0))
	      )
	    {
		    int3 boundary = (int3){0,op.ray_direction.y,0};
	      	    auto task = ComputeTask::RayUpdate(op,i,boundary,op.ray_direction,device,swap_offset,fields_already_depend_on_boundaries);
              	    graph->all_tasks.push_back(task);
	      	    compute_task_poststep(op,task);
	    }
	    if(
		  (op.ray_direction.z == -1 && ((op.boundary & BOUNDARY_Z_BOT) != 0))
		||(op.ray_direction.z == +1 && ((op.boundary & BOUNDARY_Z_TOP) != 0))
	      )
	    {
		    int3 boundary = (int3){0,0,op.ray_direction.z};
	      	    auto task = ComputeTask::RayUpdate(op,i,boundary,op.ray_direction,device,swap_offset,fields_already_depend_on_boundaries);
              	    graph->all_tasks.push_back(task);
	      	    compute_task_poststep(op,task);
	    }
            acVerboseLogFromRootProc(rank, "Ray updates created\n");
            for (size_t buf = 0; buf < op.num_fields_out; buf++) {
		if(kernel_writes_to_output(op.kernel_enum,op.fields_out[buf]))
		{
                	swap_offset[op.fields_out[buf]] = !swap_offset[op.fields_out[buf]];
		}
            }
            for (size_t buf = 0; buf < op.num_profiles_write_out; buf++) {
                swap_offset[op.profiles_write_out[buf]+NUM_VTXBUF_HANDLES] = !swap_offset[op.profiles_write_out[buf]+NUM_VTXBUF_HANDLES];
            }
            break;
        }
	
        case TASKTYPE_COMPUTE: {
            acVerboseLogFromRootProc(rank, "Creating compute tasks\n");
	    std::vector<Field> fields_out(op.fields_out,op.fields_out+op.num_fields_out);
	    std::vector<Field> fields_in(op.fields_in,op.fields_in+op.num_fields_in);

	    std::vector<Profile> profiles_out(op.profiles_reduce_out,op.profiles_reduce_out+op.num_profiles_reduce_out);
	    std::vector<Profile> profiles_in(op.profiles_in,op.profiles_in+op.num_profiles_in);

	    std::vector<KernelReduceOutput> reduce_output_in(op.outputs_in,  op.outputs_in +op.num_outputs_in);
	    std::vector<KernelReduceOutput> reduce_output_out(op.outputs_out,op.outputs_out+op.num_outputs_out);


	    const bool raytracing = is_raytracing_kernel(op.kernel_enum);
	    const bool oned_launch = kernel_only_writes_profile(op.kernel_enum,PROFILE_X) || kernel_only_writes_profile(op.kernel_enum,PROFILE_Y) || kernel_only_writes_profile(op.kernel_enum,PROFILE_Z);
	    const bool single_gpu_optim = ((comm_size == 1) || (NGHOST == 0)) && !grid.submesh.info[AC_skip_single_gpu_optim];
	    const int max_comp_facet_class = (oned_launch || raytracing || single_gpu_optim) ? 0 : 3;
	    {
            	for (int tag = Region::min_comp_tag; tag < Region::max_comp_tag; tag++) {
		    if(acGridGetLocalMeshInfo()[AC_dimension_inactive].x  && Region::tag_to_id(tag).x != 0) continue;
		    if(acGridGetLocalMeshInfo()[AC_dimension_inactive].y  && Region::tag_to_id(tag).y != 0) continue;
		    if(acGridGetLocalMeshInfo()[AC_dimension_inactive].z  && Region::tag_to_id(tag).z != 0) continue;
		    else if(max_comp_facet_class == 3)
		    {
			const AcBoundary bc_dependencies = get_kernel_depends_on_boundaries(op.kernel_enum,fields_already_depend_on_boundaries);
			if(!(bc_dependencies & BOUNDARY_X) && !(op.computes_on_halos & BOUNDARY_X))
				if(Region::tag_to_id(tag).x != 0) continue;
			if(!(bc_dependencies & BOUNDARY_Y) && !(op.computes_on_halos & BOUNDARY_Y))
				if(Region::tag_to_id(tag).y != 0) continue;
			if(!(bc_dependencies & BOUNDARY_Z) && !(op.computes_on_halos & BOUNDARY_Z))
				if(Region::tag_to_id(tag).z != 0) continue;
		    }
	    	    //auto task = std::make_shared<ComputeTask>(op,tag,full_input_region,full_region,device,swap_offset);
            	    //graph->all_tasks.push_back(task);
    	            ERRCHK_ALWAYS((int)dims.x > grid.submesh.info[AC_nmin].x*2);
    	            ERRCHK_ALWAYS((int)dims.y > grid.submesh.info[AC_nmin].y*2);
    	            ERRCHK_ALWAYS((int)dims.z > grid.submesh.info[AC_nmin].z*2);
		    if(Region::tag_to_facet_class(tag) > max_comp_facet_class) continue;
            	    auto task = std::make_shared<ComputeTask>(op, i, tag, start, dims, device, swap_offset,fields_already_depend_on_boundaries,max_comp_facet_class);
            	    graph->all_tasks.push_back(task);
		    compute_task_poststep(op,task);
            	}
	    }
            acVerboseLogFromRootProc(rank, "Compute tasks created\n");
            for (size_t buf = 0; buf < op.num_fields_out; buf++) {
		if(kernel_writes_to_output(op.kernel_enum,op.fields_out[buf]))
		{
                	swap_offset[op.fields_out[buf]] = !swap_offset[op.fields_out[buf]];
		}
            }
            for (size_t buf = 0; buf < op.num_profiles_write_out; buf++) {
                swap_offset[op.profiles_write_out[buf]+NUM_VTXBUF_HANDLES] = !swap_offset[op.profiles_write_out[buf]+NUM_VTXBUF_HANDLES];
            }
	    //++num_comp_tasks;
            break;
        }

        case TASKTYPE_HALOEXCHANGE: {
	    if(globally_imposed_bcs)
	    {
		    fatal("%s","Tried to generate taskgraph with globally imposed bcs and halo exchanges!\n");
	    }
            acVerboseLogFromRootProc(rank, "Creating halo exchange tasks\n");
            int tag0 = grid.mpi_tag_space_count * Region::max_halo_tag;
            for (int tag = Region::min_halo_tag; tag < Region::max_halo_tag; tag++) {

		if(acGridGetLocalMeshInfo()[AC_dimension_inactive].x  && Region::tag_to_id(tag).x != 0) continue;
		if(acGridGetLocalMeshInfo()[AC_dimension_inactive].y  && Region::tag_to_id(tag).y != 0) continue;
		if(acGridGetLocalMeshInfo()[AC_dimension_inactive].z  && Region::tag_to_id(tag).z != 0) continue;

		bool included = false;
		included |= (Region::tag_to_id(tag).x == -1 && (op.boundary & BOUNDARY_X_BOT) != 0);
		included |= (Region::tag_to_id(tag).x == +1 && (op.boundary & BOUNDARY_X_TOP) != 0);
		included |= (Region::tag_to_id(tag).y == -1 && (op.boundary & BOUNDARY_Y_BOT) != 0);
		included |= (Region::tag_to_id(tag).y == +1 && (op.boundary & BOUNDARY_Y_TOP) != 0);
		included |= (Region::tag_to_id(tag).z == -1 && (op.boundary & BOUNDARY_Z_BOT) != 0);
		included |= (Region::tag_to_id(tag).z == +1 && (op.boundary & BOUNDARY_Z_TOP) != 0);
		if(!included) continue;

                if (op.include_boundaries || !Region::is_on_boundary(decomp, rank, tag, BOUNDARY_XYZ, ac_proc_mapping_strategy())) {
                    auto task = std::make_shared<HaloExchangeTask>(op, i, start, dims, tag0, tag, grid_info,
                                                                   device, swap_offset,false);
                    graph->halo_tasks.push_back(task);
                    graph->all_tasks.push_back(task);
                }
            }
            acVerboseLogFromRootProc(rank, "Halo exchange tasks created\n");
            grid.mpi_tag_space_count++;
            break;
        }

        case TASKTYPE_RAY_REDUCE: {
            acVerboseLogFromRootProc(rank, "Creating ray reduce tasks\n");
            int tag0 = grid.mpi_tag_space_count * Region::max_halo_tag;
            auto task = std::make_shared<MPIReduceTask>(op, i, start, dims, tag0, op.ray_direction, grid_info,
                                                           device, swap_offset);
            graph->all_tasks.push_back(task);

            acVerboseLogFromRootProc(rank, "Ray reduce tasks created\n");
            grid.mpi_tag_space_count++;
            break;
        }


        case TASKTYPE_BOUNDCOND: {
	    int tag0       = grid.mpi_tag_space_count * Region::max_halo_tag;
	    //TP: this is because bcs of higher facet class can depend on the bcs of the lower facet classes
	    //Take for example symmetric bc. At (-1,1,0) with boundary_normal (1,0,0) the bc would depend on (0,1,0)
	    //If that bc was generated first there would be no dependency between the bcs and cause a race condition
	    std::array<int,26> correct_tag_order {

	    	    Region::id_to_tag((int3){+1,0,0}),
	    	    Region::id_to_tag((int3){0,+1,0}),
	    	    Region::id_to_tag((int3){0,0,+1}),
	    	    Region::id_to_tag((int3){-1,0,0}),
	    	    Region::id_to_tag((int3){0,-1,0}),
	    	    Region::id_to_tag((int3){0,0,-1}),

	    	    Region::id_to_tag((int3){+1,+1,0}),
	    	    Region::id_to_tag((int3){+1,-1,0}),
	    	    Region::id_to_tag((int3){-1,+1,0}),
	    	    Region::id_to_tag((int3){-1,-1,0}),
	    	    Region::id_to_tag((int3){+1,0,+1}),
	    	    Region::id_to_tag((int3){+1,0,-1}),
	    	    Region::id_to_tag((int3){-1,0,+1}),
	    	    Region::id_to_tag((int3){-1,0,-1}),
	    	    Region::id_to_tag((int3){0,+1,+1}),
	    	    Region::id_to_tag((int3){0,+1,-1}),
	    	    Region::id_to_tag((int3){0,-1,+1}),
	    	    Region::id_to_tag((int3){0,-1,-1}),

	    	    Region::id_to_tag((int3){+1,+1,+1}),
	    	    Region::id_to_tag((int3){-1,+1,+1}),
	    	    Region::id_to_tag((int3){+1,-1,+1}),
	    	    Region::id_to_tag((int3){-1,-1,+1}),
	    	    Region::id_to_tag((int3){+1,+1,-1}),
	    	    Region::id_to_tag((int3){-1,+1,-1}),
	    	    Region::id_to_tag((int3){+1,-1,-1}),
	    	    Region::id_to_tag((int3){-1,-1,-1}),
	    };
            for (const int tag : correct_tag_order) {

		//TP: this is because bcs of higher facet class can depend on the bcs of the lower facet classes
		//Take for example symmetric bc. At (-1,1,0) with boundary_normal (1,0,0) the bc would depend on (0,1,0)
		//If that bc was generated first there would be no dependency between the bcs and cause a race condition
		const auto id = Region::tag_to_id(tag);
		if(op.id != (int3){0,0,0} && id != op.id) continue;
		if(acGridGetLocalMeshInfo()[AC_dimension_inactive].x  && id.x != 0) continue;
		if(acGridGetLocalMeshInfo()[AC_dimension_inactive].y  && id.y != 0) continue;
		if(acGridGetLocalMeshInfo()[AC_dimension_inactive].z  && id.z != 0) continue;

                acVerboseLogFromRootProc(rank,
                                         "tag %d, decomp %i %i %i, rank %i, op.boundary  %i \n ",
                                         tag, decomp.x, decomp.y, decomp.z, rank, op.boundary);
                acVerboseLogFromRootProc(rank,
                                         "acGridBuildTaskGraph: Region::is_on_boundary(decomp, "
                                         "rank, tag, op.boundary) = %i \n",
                                         Region::is_on_boundary(decomp, rank, tag, op.boundary, ac_proc_mapping_strategy()));
		const bool is_on_boundary = Region::is_on_boundary(decomp, rank, tag, op.boundary, ac_proc_mapping_strategy());
		if(!is_on_boundary && !globally_imposed_bcs) continue;
                if (op.kernel_enum == BOUNDCOND_PERIODIC) {
		    if(globally_imposed_bcs)
		    {
		    	fatal("%s","Can not use periodic bcs and globally imposed bcs at the same time!\n");
		    }
                    acVerboseLogFromRootProc(rank, "Creating periodic bc task with tag%d\n",
                                             tag);
		    const bool shear_periodic = acGridGetLocalMeshInfo()[AC_shear] && (
		    				   (id.x == -1 && acGridGetLocalMeshInfo()[AC_domain_coordinates].x == 0)
		    				|| (id.x == 1  && acGridGetLocalMeshInfo()[AC_domain_coordinates].x == int(decomp.x-1))
		    				);
		    //TP: because of the need to interpolate from multiple processes the whole stretch from 0 to AC_mlocal.y is done in a single task
		    if(shear_periodic && id.y != 0) continue;
                    auto task = std::make_shared<HaloExchangeTask>(op, i, start, dims, tag0, tag, grid_info, device, swap_offset,shear_periodic);
		    if(task->active)
		    {
		    	if(id.x != 0 && !acGridGetLocalMeshInfo()[AC_periodic_grid].x)
		    	{
		    	        fatal("Trying to apply periodic bc on (%d,%d,%d) even though X is not periodic!!\n",id.x,id.y,id.z);
		    	}
		    	if(id.y != 0 && !acGridGetLocalMeshInfo()[AC_periodic_grid].y)
		    	{
		    	        fatal("Trying to apply periodic bc on (%d,%d,%d) even though Y is not periodic!!\n",id.x,id.y,id.z);
		    	}
		    	if(id.z != 0 && !acGridGetLocalMeshInfo()[AC_periodic_grid].z)
		    	{
		    	        fatal("Trying to apply periodic bc on (%d,%d,%d) even though Z is not periodic!!\n",id.x,id.y,id.z);
		    	}
		    }
                    graph->halo_tasks.push_back(task);
                    graph->all_tasks.push_back(task);
                    acVerboseLogFromRootProc(rank,
                                             "Done creating periodic bc task with tag%d\n",
                                             tag);

                }
		else
		{
                	auto task = std::make_shared<BoundaryConditionTask>(op,
                	                                                       boundary_normal(tag,op.boundary),
                	                                                       i, tag, start, dims,
                	                                                       device,
                	                                                       swap_offset);
                	graph->all_tasks.push_back(task);
		}
            }
            grid.mpi_tag_space_count += (op.kernel_enum  == BOUNDCOND_PERIODIC);
            acVerboseLogFromRootProc(rank, "Boundcond tasks created\n");
            break;
        }
	case TASKTYPE_REDUCE:  {

	  if(kernel_reduces_only_profiles(op.kernel_enum,PROFILE_X))
	  {
		for(int id = -1; id <= 1; ++id)
		{
	  		auto task   = std::make_shared<ReduceTask>(op, i, Region::id_to_tag({id,0,0}), start,dims, device, swap_offset);
	  		graph->all_tasks.push_back(task);
		}
	  }
	  else if(kernel_reduces_only_profiles(op.kernel_enum,PROFILE_Y))
	  {
		for(int id = -1; id <= 1; ++id)
		{
	  		auto task   = std::make_shared<ReduceTask>(op, i, Region::id_to_tag({0,id,0}), start,dims, device, swap_offset);
	  		graph->all_tasks.push_back(task);
		}
	  }
	  else if(kernel_reduces_only_profiles(op.kernel_enum,PROFILE_Z))
	  {
		for(int id = -1; id <= 1; ++id)
		{
	  		auto task   = std::make_shared<ReduceTask>(op, i, Region::id_to_tag({0,0,id}), start,dims, device, swap_offset);
	  		graph->all_tasks.push_back(task);
		}
	  }
	  else
	  {
	  	auto task = std::make_shared<ReduceTask>(op, i, 0, start,dims, device, swap_offset);
	  	graph->all_tasks.push_back(task);
	  }
	  break;
	}
        }
    }
    acVerboseLogFromRootProc(rank, "acGridBuildTaskGraph: Done creating tasks\n");

    op_indices.push_back(graph->all_tasks.size());
    graph->device_swaps = swap_offset;

    graph->halo_tasks.shrink_to_fit();
    graph->all_tasks.shrink_to_fit();

    // In order to reduce redundant dependencies, we keep track of which tasks are connected
    acVerboseLogFromRootProc(rank, "acGridBuildTaskGraph: Calculating dependencies\n");

    const size_t n_tasks               = graph->all_tasks.size();
    std::vector<std::vector<size_t>> neighbours(n_tasks,std::vector<size_t>{});

    std::vector<int> visited(n_tasks,0);
    int generation = 0;
    //...and check if there is already a forward path that connects two tasks
    auto forward_search = [&neighbours, &visited, &generation
                  	  ](size_t preq, size_t dept) {

	++generation;
        std::queue<size_t> walk;
        walk.push(preq);

        while (!walk.empty()) {
            auto curr = walk.front();
            walk.pop();
            if (curr == dept) {
                return true;
            }
	    for(const size_t& neighbor : neighbours[curr])
	    {
                if (visited[neighbor] != generation) {
                        walk.push(neighbor);
                        visited[neighbor] = generation;
                 }
            }
        }
        return false;
    };

    const auto profile_overlap_in_regions = [&](const auto gem_overlaps, const auto profiles_1, const auto profiles_2)
    {
	if constexpr (NUM_PROFILES == 0) return false;
	bool profiles_overlap = false;
	for(auto profile_1 : profiles_1)
	{
		for(auto profile_2 : profiles_2)
		{
			if(profile_1 == profile_2)
			{
				profiles_overlap |= (prof_types[profile_1] == PROFILE_X && gem_overlaps.x);
				profiles_overlap |= (prof_types[profile_1] == PROFILE_Y && gem_overlaps.y);
				profiles_overlap |= (prof_types[profile_1] == PROFILE_Z && gem_overlaps.z);

				profiles_overlap |= ((prof_types[profile_1] == PROFILE_XY || prof_types[profile_1] == PROFILE_YX) && (gem_overlaps.x || gem_overlaps.y));
				profiles_overlap |= ((prof_types[profile_1] == PROFILE_XZ || prof_types[profile_1] == PROFILE_ZX) && (gem_overlaps.x || gem_overlaps.z));
				profiles_overlap |= ((prof_types[profile_1] == PROFILE_YZ || prof_types[profile_1] == PROFILE_ZY) && (gem_overlaps.y || gem_overlaps.z));
			}
		}
	}
	return profiles_overlap;
    };

    //TP: we do profile overlaps in this convoluted manner since we want to skip stencil dependencies on profile updates
    //whenever possible.
    //I.e. if kernel A writes profile P and B reads it B should not use the stencil dependency geometry but the pointwise (with profile corrections) dependencies
    //
    //TP: In theory doing the vertex buffer dependencies in this manner would be also more precise but there is not really a an actual use case where it would help
    //so postponed until it would actually help
    const auto profile_overlap = [&](const auto preq_task, const auto dept_task)
    {
	if(dept_task->isComputeTask() && preq_task->isComputeTask())
	{
		if(kernel_has_profile_stencil_ops(std::dynamic_pointer_cast<ComputeTask>(dept_task)->getKernel()))
		{
			if(profile_overlap_in_regions(
					preq_task->output_region.geometry_overlaps(&dept_task->input_regions[0]),
					preq_task->output_region.memory.profiles,
					dept_task->input_regions[0].memory.profiles
					))
				return true;
		}
		
		if(kernel_has_profile_stencil_ops(std::dynamic_pointer_cast<ComputeTask>(preq_task)->getKernel()))
		{
			if(profile_overlap_in_regions(
					preq_task->output_region.geometry_overlaps(&dept_task->output_region),
					preq_task->output_region.memory.profiles,
					dept_task->output_region.memory.profiles
					))
				return true;
		}

	}
	const AcBool3 gem_overlaps = preq_task->output_region.geometry_overlaps(&dept_task->output_region);
	return profile_overlap_in_regions(gem_overlaps,preq_task->output_region.memory.profiles,dept_task->input_regions[0].memory.profiles);
    };

    // We walk through all tasks, and compare tasks from pairs of operations at
    // a time. Pairs are considered in order of increasing distance between the
    // operations in the pair. The final set of pairs that are considered are
    // self-equal pairs, since the operations form a cycle when iterated over
    for (size_t op_offset = 0; op_offset < ops.size(); op_offset++) {
        for (size_t dept_op = 0; dept_op < ops.size(); dept_op++) {
            size_t preq_op = (ops.size()+ dept_op - op_offset - 1) % ops.size();
            for (auto i = op_indices[preq_op]; i != op_indices[preq_op + 1]; i++) {
                auto preq_task = graph->all_tasks[i];
                if (preq_task->active) {
                    for (auto j = op_indices[dept_op]; j != op_indices[dept_op + 1]; j++) {
                        auto dept_task = graph->all_tasks[j];
                        // Task A depends on task B if the input or output region of A overlaps with the
                        // output region of B.
			// Or for profiles if the A has profile P in and the output region of A overlaps
			// with the output region of B and B writes/reduces profile P
			bool preq_output_overlaps_with_input = false;
			for(const auto& input_region : dept_task->input_regions)
			{
				preq_output_overlaps_with_input |= preq_task->output_region.overlaps(&input_region);
			}
                        if (dept_task->active &&
                            (
			     preq_output_overlaps_with_input  ||
                             preq_task->output_region.overlaps(&dept_task->output_region) ||
			     profile_overlap(preq_task,dept_task)
			    )
			    ) {
                            // iteration offset of 1 -> dependency from preq_task in iteration k to
                            // dept_task in iteration k+1
			    //
                            if (!forward_search(i, j)) {
                                preq_task->registerDependent(dept_task, preq_op < dept_op ? 0 : 1);
				if(preq_op < dept_op) neighbours[i].push_back(j);
                            }
                        }
                    }
                }
            }
        }
    }
    acVerboseLogFromRootProc(rank, "acGridBuildTaskGraph: Done calculating dependencies\n");

    // Finally sort according to a priority. Larger volumes first and comm before comp
    auto sort_lambda = [](std::shared_ptr<Task> t1, std::shared_ptr<Task> t2) {
        auto comp1 = t1->task_type == TASKTYPE_COMPUTE;
        auto comp2 = t2->task_type == TASKTYPE_COMPUTE;

        auto vol1 = t1->output_region.volume;
        auto vol2 = t2->output_region.volume;

        auto order1 = t1->order;
        auto order2 = t2->order;

        auto tag1 = t1->output_region.tag;
        auto tag2 = t2->output_region.tag;

        auto dim1 = t1->output_region.dims;
        auto dim2 = t2->output_region.dims;

	if(vol1 > vol2) return true;
        if(vol1 == vol2 && ((!comp1 && comp2) || dim1.x < dim2.x || dim1.z > dim2.z)) return true;
	//TP: these are somewhat arbitrary but the sorting function requires a well-defined order: otherwise seg faults
	if(vol1 == vol2 && dim1 == dim2 && order1 < order2) return true;
	if(vol1 == vol2 && dim1 == dim2 && order1 == order2 && tag1 < tag2) return true;
	return false;

    };
    acVerboseLogFromRootProc(rank, "acGridBuildTaskGraph: Sorting tasks by priority\n");

    std::sort(graph->halo_tasks.begin(), graph->halo_tasks.end(), sort_lambda);
    std::sort(graph->all_tasks.begin(), graph->all_tasks.end(), sort_lambda);
    acVerboseLogFromRootProc(rank, "acGridBuildTaskGraph: Done sorting tasks by priority\n");


    return graph;
}

AcTaskGraph*
acGridBuildTaskGraph(const AcTaskDefinition ops_in[], const size_t n_ops)
{
	const Volume ghost = to_volume(grid.submesh.info[AC_nmin]);
	return acGridBuildTaskGraphWithBounds(ops_in,n_ops,ghost,grid.nn+ghost,false);
}


AcResult
acGridDestroyTaskGraph(AcTaskGraph* graph)
{
    graph->all_tasks.clear();
    graph->comp_tasks.clear();
    graph->halo_tasks.clear();
    delete graph;
    return AC_SUCCESS;
}


static std::vector<KernelReduceOutput>
get_reduce_outputs(const AcTaskGraph* graph)
{
    std::vector<KernelReduceOutput> reduce_outputs{};
    for (auto& task : graph->all_tasks) {
    	if(!task->isComputeTask()) continue;
	//TP: skip inner halo regions to not count outputs multiple times
	if(task->output_region.id != (int3){0,0,0}) continue;
    	auto compute_task = std::dynamic_pointer_cast<ComputeTask>(task); 
    	auto kernel       = compute_task -> getKernel();
    	for(size_t i = 0; i < grid.kernel_analysis_info[kernel].n_reduce_outputs; ++i)
    	    reduce_outputs.push_back(grid.kernel_analysis_info[kernel].reduce_outputs[i]);
    }
    return reduce_outputs;
}
static std::vector<Profile>
get_reduced_profiles(const AcTaskGraph* graph)
{
    std::vector<Profile> reduced_profiles{};
    for (auto& task : graph->all_tasks) {
    	if(!task->isComputeTask()) continue;
	//TP: skip inner halo regions to not count outputs multiple times
	if(task->output_region.id != (int3){0,0,0}) continue;
    	auto compute_task = std::dynamic_pointer_cast<ComputeTask>(task); 
    	auto kernel       = compute_task -> getKernel();
    	for(int i = 0; i < NUM_PROFILES; ++i)
	{
	    if(grid.kernel_analysis_info[kernel].reduced_profiles[i])
    	    	reduced_profiles.push_back(Profile(i));
	}
    }
    return reduced_profiles;
}

static void
preprocess_reduce_buffers(const AcTaskGraph* graph)
{
	const auto reduce_outputs = get_reduce_outputs(graph);
    	for(const auto& output: reduce_outputs)
		acDevicePreprocessScratchPad(grid.device,output.variable,output.type,output.op);
	const auto reduced_profiles = get_reduced_profiles(graph);
	for(const auto& profile: reduced_profiles)
		acDevicePreprocessScratchPad(grid.device,profile,AC_PROF_TYPE,REDUCE_SUM);
}

AcResult
acGridFinalizeReduceLocal(AcTaskGraph* graph)
{
    if constexpr(NUM_OUTPUTS == 0) return AC_SUCCESS;
    AcReal local_res_real[NUM_OUTPUTS]{};
    int    local_res_int[NUM_OUTPUTS]{};
#if AC_DOUBLE_PRECISION
    float  local_res_float[NUM_OUTPUTS]{};
#endif
    const auto reduce_outputs = get_reduce_outputs(graph);
    for(size_t i = 0; i < reduce_outputs.size(); ++i)
    {
	    const auto var    = reduce_outputs[i].variable;
	    const auto op     = reduce_outputs[i].op;
	    const auto kernel = reduce_outputs[i].kernel;
	    if(reduce_outputs[i].type == AC_REAL_TYPE)
	    	acDeviceFinishReduce(grid.device,(Stream)(int)i,&local_res_real[i],kernel,op,(AcRealOutputParam)var);
	    else if(reduce_outputs[i].type == AC_INT_TYPE)
	    	acDeviceFinishReduce(grid.device,(Stream)(int)i,&local_res_int[i],kernel,op,(AcIntOutputParam)var);
#if AC_DOUBLE_PRECISION
	    else if(reduce_outputs[i].type == AC_FLOAT_TYPE)
	    	acDeviceFinishReduce(grid.device,(Stream)(float)i,&local_res_float[i],kernel,op,(AcFloatOutputParam)var);
#endif
	    else if(reduce_outputs[i].type == AC_PROF_TYPE)
		;//acDeviceReduceAverages(grid.device, reduce_outputs[i].variable, (Profile)reduce_outputs[i].variable);
	    else
		fatal("Unknown reduce output type: %d,%d\n",i,reduce_outputs[i].type);
    }

    acDeviceSynchronizeStream(grid.device,STREAM_ALL);
    for(size_t i = 0; i < reduce_outputs.size(); ++i)
    {
		if(reduce_outputs[i].type == AC_REAL_TYPE)
			acDeviceSetOutput(grid.device,(AcRealOutputParam)reduce_outputs[i].variable,local_res_real[i]);
		else if(reduce_outputs[i].type == AC_INT_TYPE)
			acDeviceSetOutput(grid.device,(AcIntOutputParam)reduce_outputs[i].variable,local_res_int[i]);
#if AC_DOUBLE_PRECISION
		else if(reduce_outputs[i].type == AC_FLOAT_TYPE)
			acDeviceSetOutput(grid.device,(AcFloatOutputParam)reduce_outputs[i].variable,local_res_float[i]);
#endif
		else if(reduce_outputs[i].type == AC_PROF_TYPE)
			;
		else
			fatal("Unknown reduce output type: %d,%d\n",i,reduce_outputs[i].type);
    }
    return AC_SUCCESS;
}
static MPI_Op
to_mpi_op(const AcReduceOp op)
{
	switch(op)
	{
		case(REDUCE_SUM):
			return MPI_SUM;
		case(REDUCE_MIN):
			return MPI_MIN;
		case(REDUCE_MAX):
			return MPI_MAX;
		case(NO_REDUCE):
			fatal("%s","Should not call to_mpi_op for NO_REDUCE\n");
	}
	fatal("%s","No mapping from AcReduceOp to MPI_Op\n");
}

AcResult
acGridFinalizeReduce(AcTaskGraph* graph)
{
    if constexpr(NUM_OUTPUTS == 0) return AC_SUCCESS;
    acGridFinalizeReduceLocal(graph);
    const auto reduce_outputs = get_reduce_outputs(graph);
    for(int i = 0; i < NUM_OUTPUTS; ++i)
    {
    	if(reduce_outputs[i].variable >=0)
    	{
		AcReal local_res;
		if(reduce_outputs[i].type == AC_REAL_TYPE)
			local_res = acDeviceGetOutput(grid.device,(AcRealOutputParam)reduce_outputs[i].variable);
		else
			fatal("%s","Unknown reduce output type\n");
    	        AcReal mpi_res{};
    		MPI_Allreduce(&local_res, &mpi_res, 1, AC_REAL_MPI_TYPE, to_mpi_op(reduce_outputs[i].op), astaroth_comm);
		acDeviceSetOutput(grid.device,(AcRealOutputParam)reduce_outputs[i].variable,mpi_res);
    	}
    }
    return AC_SUCCESS;
}

static void
set_device_to_grid_device()
{
	acSetDevice(acDeviceGetId(grid.device));
}


AcResult
acGridExecuteTaskGraphBase(AcTaskGraph* graph, size_t n_iterations, const bool include_inactive)
{
    preprocess_reduce_buffers(graph);
    ERRCHK(grid.initialized);
    // acGridSynchronizeStream(stream);
    // acDeviceSynchronizeStream(grid.device, stream);
    set_device_to_grid_device();
    if (graph->trace_file.enabled) {
        timer_reset(&(graph->trace_file.timer));
    }

    for (auto& task : graph->all_tasks) {
        if (include_inactive || task->active) {
            task->syncVBA();
            task->setIterationParams(0, n_iterations);
        }
    }
    bool ready;
    do {
        ready = true;
        for (auto& task : graph->all_tasks) {
            if (include_inactive || task->active) {
                task->update(graph->device_swaps, &(graph->trace_file));
                ready &= task->isFinished();
            }
        }
    } while (!ready);

    if (n_iterations % 2 != 0) {
        for (size_t i = 0; i < NUM_VTXBUF_HANDLES; i++) {
            if (graph->device_swaps[i]) {
                acDeviceSwapBuffer(grid.device, (VertexBufferHandle)i);
            }
        }
        for (int i = 0; i < NUM_PROFILES; i++) {
            if (graph->device_swaps[i+NUM_VTXBUF_HANDLES]) {
                acDeviceSwapProfileBuffer(grid.device, (Profile)i);
            }
        }
    }
    return AC_SUCCESS;
}

AcResult
acGridExecuteTaskGraph(AcTaskGraph* graph, size_t n_iterations)
{
        return acGridExecuteTaskGraphBase(graph,n_iterations,false);
}


#ifdef AC_INTEGRATION_ENABLED
AcResult
acGridIntegrate(const Stream stream, const AcReal dt)
{
    if(!grid.submesh.info[AC_fully_periodic_grid])
    {
	    fatal("%s","acGridIntegrate assumes fully periodic grid!\n");
    }
    (void)stream;
    ERRCHK(grid.initialized);
    acDeviceSetInput(grid.device,AC_dt,dt);
    acDeviceSetInput(grid.device,AC_current_time,dt);
    return acGridExecuteTaskGraph(grid.default_tasks.get(), 3);
}
#endif // AC_INTEGRATION_ENABLED

AcResult
acGridHaloExchange()
{
    return acGridExecuteTaskGraph(grid.halo_exchange_tasks.get(), 1);
}
AcResult
acGridPeriodicBoundconds(const Stream stream)
{
    (void)stream;
    if(!grid.submesh.info[AC_fully_periodic_grid])
    {
	    fatal("%s","acGridPeriodicBoundconds assumes fully periodic grid!\n");
    }
    return acGridExecuteTaskGraph(grid.periodic_bc_tasks.get(), 1);
}
static size_t
get_n_global_points()
{
	return (grid.nn.x * grid.decomposition.x *
		grid.nn.y * grid.decomposition.y * 
		grid.nn.z * grid.decomposition.z);
}

static UNUSED AcReal
get_cell_volume()
{
	const AcReal3 spacings = get_spacings();
	const AcReal cell_volume   = spacings.x*spacings.y
				     *spacings.z
				     ;
	return cell_volume;
}

static AcResult
distributedScalarReduction(const AcReal local_result, const AcReduction reduction, AcReal* result)
{
    const MPI_Op op = to_mpi_op(reduction.reduce_op);

    int rank;
    MPI_Comm_rank(astaroth_comm, &rank);

    AcReal mpi_res;
    MPI_Allreduce(&local_result, &mpi_res, 1, AC_REAL_MPI_TYPE, op, astaroth_comm);

    //TP: If I am not mistaken this won't generalize to non-cartesian coordinates
    //TP: spacings should be included in the mapping function
    if (reduction.post_processing_op == AC_RMS) {
        mpi_res            = sqrt(mpi_res/get_n_global_points());
    }

#ifdef AC_INTEGRATION_ENABLED

    //TP: should this be done for other radial reductions?
    if (reduction.post_processing_op == AC_RADIAL_WINDOW_RMS) {
        // MV NOTE: This has to be calculated here separately, because does not
        //          know what GPU is doing. 
	const AcReal cell_volume = get_cell_volume(); 
	const AcReal window_radius = acGridGetLocalMeshInfo()[AC_window_radius];
        const AcReal sphere_volume = (AcReal)(4.0 / 3.0) * (AcReal)M_PI * window_radius*window_radius*window_radius;
        // only include whole cells
        const AcReal cell_number = AcReal(int(sphere_volume / cell_volume));

        mpi_res = sqrt(mpi_res / cell_number);
    }
#endif
    *result = mpi_res;
    return AC_SUCCESS;
}

AcResult
acGridReduceScal(const Stream stream, const AcReduction reduction,
                 const VertexBufferHandle vtxbuf_handle, AcReal* result)
{
    ERRCHK(grid.initialized);
    const Device device = grid.device;
    acGridSynchronizeStream(STREAM_ALL);

    AcReal local_result;
    if (acDeviceReduceScalNoPostProcessing(device, stream, reduction, vtxbuf_handle, &local_result) == AC_NOT_ALLOCATED) return AC_NOT_ALLOCATED;

    return distributedScalarReduction(local_result, reduction, result);
}

AcResult
acGridReduceVec(const Stream stream, const AcReduction reduction, const VertexBufferHandle vtxbuf0,
                const VertexBufferHandle vtxbuf1, const VertexBufferHandle vtxbuf2, AcReal* result)
{
    ERRCHK(grid.initialized);
    const Device device = grid.device;
    acGridSynchronizeStream(STREAM_ALL);

    AcReal local_result;
    if (acDeviceReduceVecNoPostProcessing(device, stream, reduction, vtxbuf0, vtxbuf1, vtxbuf2, &local_result) == AC_NOT_ALLOCATED) return AC_NOT_ALLOCATED;

    return distributedScalarReduction(local_result, reduction, result);
}

AcResult
acGridReduceVec(const Stream stream, const AcReduction reduction, const VertexBufferHandle* vtxbufs, AcReal* result)
{
    return acGridReduceVec(stream, reduction, vtxbufs[0], vtxbufs[1], vtxbufs[2], result);
}

AcResult
acGridReduceVecScal(const Stream stream, const AcReduction reduction,
                    const VertexBufferHandle vtxbuf0, const VertexBufferHandle vtxbuf1,
                    const VertexBufferHandle vtxbuf2, const VertexBufferHandle vtxbuf3,
                    AcReal* result)
{
    ERRCHK(grid.initialized);
    const Device device = grid.device;
    acGridSynchronizeStream(STREAM_ALL);

    AcReal local_result;
    if (acDeviceReduceVecScalNoPostProcessing(device, stream, reduction, vtxbuf0, vtxbuf1, vtxbuf2, vtxbuf3,
                                     &local_result) == AC_NOT_ALLOCATED) return AC_NOT_ALLOCATED;

    return distributedScalarReduction(local_result, reduction, result);
}

AcResult
acGridReduceXY(const Stream stream, const Field field, const Profile profile, const AcReduction reduction)
{
    if (profile >= 0 && profile < NUM_PROFILES) {
        ERRCHK(grid.initialized);
        const Device device = grid.device;
        acGridSynchronizeStream(STREAM_ALL);

        // Strategy:
        // 1) Reduce the local result to device->vba.profiles.in
        acDeviceReduceXY(device, stream, field, profile,reduction);

        // 2) Create communicator that encompasses the processes that are neighbors in the xy
        // direction
        int nprocs, pid;
        MPI_Comm_size(astaroth_comm, &nprocs);
        MPI_Comm_rank(astaroth_comm, &pid);

        const uint3_64 decomp = decompose(nprocs,ac_decomp_strategy());
        const int3 pid3d      = getPid3D(pid, decomp);
        MPI_Comm xy_neighbors;
        MPI_Comm_split(acGridMPIComm(), pid3d.z, pid, &xy_neighbors);

        // 3) Allreduce
	//
	const size3_t m = (size3_t){
		   		    static_cast<size_t>(grid.submesh.info[AC_mlocal].x),
				    static_cast<size_t>(grid.submesh.info[AC_mlocal].y),
				    static_cast<size_t>(grid.submesh.info[AC_mlocal].z)
			           };
        MPI_Allreduce(MPI_IN_PLACE, acDeviceGetProfileBuffer(device,profile), prof_size(profile,m),
                      AC_REAL_MPI_TYPE, MPI_SUM, xy_neighbors);

        // 4) Optional: Test
        // AcReal arr[device->vba.profiles.count];
        // acMemcpy(arr, device->vba.profiles.in[profile], device->vba.profiles.count,
        //            cudaMemcpyDeviceToHost);
        // for (size_t i = 0; i < device->vba.profiles.count; ++i)
        //     printf("%i: %g\n", i, arr[i]);

        return AC_SUCCESS;
    }
    else {
        return AC_FAILURE;
    }
}

AcResult
acGridReduceXYAverages(const Stream stream)
{
    if (NUM_PROFILES > 0) {
        ERRCHK(grid.initialized);
        const Device device = grid.device;
        acGridSynchronizeStream(STREAM_ALL);

        // Strategy:
        // 1) Reduce the local result to device->vba.profiles.in
        acDeviceReduceXYAverages(device, stream);

        // 2) Create communicator that encompasses the processes that are neighbors in the xy
        // direction
        int nprocs, pid;
        MPI_Comm_size(astaroth_comm, &nprocs);
        MPI_Comm_rank(astaroth_comm, &pid);

        const uint3_64 decomp = decompose(nprocs,ac_decomp_strategy());
        const int3 pid3d      = getPid3D(pid, decomp);
        MPI_Comm xy_neighbors;
        MPI_Comm_split(acGridMPIComm(), pid3d.z, pid, &xy_neighbors);

        // 3) Allreduce
	const size_t count = static_cast<size_t>(grid.submesh.info[AC_mlocal].x*grid.submesh.info[AC_mlocal].y);
        MPI_Allreduce(MPI_IN_PLACE, acDeviceGetStartOfProfiles(grid.device),
                      NUM_PROFILES * count, AC_REAL_MPI_TYPE, MPI_SUM,
                      xy_neighbors);

        // 4) Optional: Test
        // AcReal arr[device->vba.profiles.count];
        // acMemcpy(arr, device->vba.profiles.in[profile], device->vba.profiles.count,
        //            cudaMemcpyDeviceToHost);
        // for (size_t i = 0; i < device->vba.profiles.count; ++i)
        //     printf("%i: %g\n", i, arr[i]);

        return AC_SUCCESS;
    }
    else {
        return AC_FAILURE;
    }
}

/** */
AcResult
acGridLaunchKernel(const Stream stream, const AcKernel kernel, const Volume start, const Volume end)
{
    ERRCHK(grid.initialized);
    acGridSynchronizeStream(stream);
    return acDeviceLaunchKernel(grid.device, stream, kernel, start, end);
}

AcResult
acGridSwapBuffers(void)
{
    ERRCHK(grid.initialized);
    return acDeviceSwapBuffers(grid.device);
}

/** */
AcResult
acGridLoadStencil(const Stream stream, const Stencil stencil,
                  const AcReal data[STENCIL_DEPTH][STENCIL_HEIGHT][STENCIL_WIDTH])
{
    ERRCHK(grid.initialized);

    acGridSynchronizeStream(stream);
    return acDeviceLoadStencil(grid.device, stream, stencil, data);
}

/** */
AcResult
acGridStoreStencil(const Stream stream, const Stencil stencil,
                   AcReal data[STENCIL_DEPTH][STENCIL_HEIGHT][STENCIL_WIDTH])
{
    ERRCHK(grid.initialized);

    acGridSynchronizeStream(stream);
    return acDeviceStoreStencil(grid.device, stream, stencil, data);
}

/** */
AcResult
acGridLoadStencils(const Stream stream,
                   const AcReal data[NUM_STENCILS][STENCIL_DEPTH][STENCIL_HEIGHT][STENCIL_WIDTH])
{
    ERRCHK(grid.initialized);
    ERRCHK((int)AC_SUCCESS == 0);
    ERRCHK((int)AC_FAILURE == 1);
    acGridSynchronizeStream(stream);

    int retval = 0;
    for (size_t i = 0; i < NUM_STENCILS; ++i)
        retval |= acGridLoadStencil(stream, (Stencil)i, data[i]);

    return (AcResult)retval;
}

/** */
AcResult
acGridStoreStencils(const Stream stream,
                    AcReal data[NUM_STENCILS][STENCIL_DEPTH][STENCIL_HEIGHT][STENCIL_WIDTH])
{
    ERRCHK(grid.initialized);
    ERRCHK((int)AC_SUCCESS == 0);
    ERRCHK((int)AC_FAILURE == 1);
    acGridSynchronizeStream(stream);

    int retval = 0;
    for (size_t i = 0; i < NUM_STENCILS; ++i)
        retval |= acGridStoreStencil(stream, (Stencil)i, data[i]);

    return (AcResult)retval;
}

/*
static AcResult
volume_copy_to_from_host(const VertexBufferHandle vtxbuf, const AccessType type)
{
    ERRCHK(grid.initialized);

    acGridSynchronizeStream(STREAM_ALL); // Possibly unnecessary

    const Device device   = grid.device;
    const AcMeshInfo info = acDeviceGetLocalConfig(device);

    if (type == ACCESS_WRITE) {
        const AcReal* in      = device->vba.on_device.in[vtxbuf];
        const int3 in_offset  = acConstructInt3Param(AC_nx_min, AC_ny_min, AC_nz_min, info);
        const int3 in_volume  = acConstructInt3Param(AC_mx, AC_my, AC_mz, info);
        AcReal* out           = device->vba.on_device.out[vtxbuf];
        const int3 out_offset = (int3){0, 0, 0};
        const int3 out_volume = acConstructInt3Param(AC_nx, AC_ny, AC_nz, info);
        acDeviceVolumeCopy(device, STREAM_DEFAULT, in, in_offset, in_volume, out, out_offset,
                           out_volume);
        acDeviceSynchronizeStream(device, STREAM_DEFAULT);

        // ---------------------------------------
        // Buffer through CPU
        set_device_to_grid_device();
        const size_t count = acVertexBufferCompdomainSizeBytes(info);
        acMemcpy(grid.submesh.vertex_buffer[vtxbuf], out, count, cudaMemcpyDeviceToHost);
        // ----------------------------------------
    }

    if (type == ACCESS_READ) {
        AcReal* in           = device->vba.on_device.out[vtxbuf];
        const int3 in_offset = (int3){0, 0, 0};
        const int3 in_volume = acConstructInt3Param(AC_nx, AC_ny, AC_nz, info);

        AcReal* out           = device->vba.on_device.in[vtxbuf];
        const int3 out_offset = acConstructInt3Param(AC_nx_min, AC_ny_min, AC_nz_min, info);
        const int3 out_volume = acConstructInt3Param(AC_mx, AC_my, AC_mz, info);

        // ---------------------------------------
        // Buffer through CPU
        set_device_to_grid_device();
        const size_t count = acVertexBufferCompdomainSizeBytes(info);
        acMemcpy(in, grid.submesh.vertex_buffer[vtxbuf], count, cudaMemcpyHostToDevice);
        // ----------------------------------------

        acDeviceVolumeCopy(device, STREAM_DEFAULT, in, in_offset, in_volume, out, out_offset,
                           out_volume);

        // Apply boundconds and sync
        acGridPeriodicBoundconds(STREAM_DEFAULT);
        acDeviceSynchronizeStream(device, STREAM_DEFAULT);
    }
    acGridSynchronizeStream(STREAM_ALL); // Possibly unnecessary
    return AC_SUCCESS;
}

static AcResult
access_vtxbuf_on_disk(const VertexBufferHandle vtxbuf, const char* path, const AccessType type)
{
    const Device device   = grid.device;
    const AcMeshInfo info = acDeviceGetLocalConfig(device);
    const int3 nn         = acConstructInt3Param(AC_nxgrid, AC_nygrid, AC_nzgrid);
    const int3 nn_sub     = acConstructInt3Param(AC_nx, AC_ny, AC_nz, info);
    const int3 offset     = info[AC_multigpu_offset]; // Without halo

    MPI_Datatype subarray;
    const int arr_nn[]     = {nn.z, nn.y, nn.x};
    const int arr_nn_sub[] = {nn_sub.z, nn_sub.y, nn_sub.x};
    const int arr_offset[] = {offset.z, offset.y, offset.x};
    MPI_Type_create_subarray(3, arr_nn, arr_nn_sub, arr_offset, MPI_ORDER_C, AC_REAL_MPI_TYPE,
                             &subarray);
    MPI_Type_commit(&subarray);

    MPI_File file;

    int flags = 0;
    if (type == ACCESS_READ)
        flags = MPI_MODE_RDONLY;
    else
        flags = MPI_MODE_CREATE | MPI_MODE_WRONLY;

    ERRCHK_ALWAYS(MPI_File_open(astaroth_comm, path, flags, MPI_INFO_NULL, &file) == MPI_SUCCESS);

    ERRCHK_ALWAYS(MPI_File_set_view(file, 0, AC_REAL_MPI_TYPE, subarray, "native", MPI_INFO_NULL) ==
                  MPI_SUCCESS);

    MPI_Status status;

    // ---------------------------------------
    // Buffer through CPU
    AcReal* arr = grid.submesh.vertex_buffer[vtxbuf];
    // ----------------------------------------

    const size_t nelems = nn_sub.x * nn_sub.y * nn_sub.z;
    if (type == ACCESS_READ) {
        ERRCHK_ALWAYS(MPI_File_read_all(file, arr, nelems, AC_REAL_MPI_TYPE, &status) ==
                      MPI_SUCCESS);
    }
    else {
        ERRCHK_ALWAYS(MPI_File_write_all(file, arr, nelems, AC_REAL_MPI_TYPE, &status) ==
                      MPI_SUCCESS);
    }

    ERRCHK_ALWAYS(MPI_File_close(&file) == MPI_SUCCESS);

    MPI_Type_free(&subarray);
    return AC_SUCCESS;
}
*/

/*

    write:
        sync transfer to host
        async write to disk
    read:
        async read from disk
        sync transfer to device

    sync:
        complete write or read locally  (future.get() and status complete)
        complete write or read globally (MPI_Barrier)

    static:
        future
        status

    static std::future<void> future;
    static AccessType access_type;
    static bool complete = true;
*/

/*
#include <chrono>
#include <future>

static std::future<void> future;
static AccessType access_type = ACCESS_WRITE;
static bool complete          = true;

AcResult
acGridDiskAccessSyncOld(void)
{
    ERRCHK(grid.initialized);

    // Sync and mark as completed
    if (future.valid())
        future.get();

    if (access_type == ACCESS_READ)
        for (size_t i = 0; i < NUM_VTXBUF_HANDLES; ++i)
            volume_copy_to_from_host((VertexBufferHandle)i, ACCESS_READ);

    acGridSynchronizeStream(STREAM_ALL);
    access_type = ACCESS_WRITE;
    complete    = true;
    return AC_SUCCESS;
}

static void
write_async(void)
{
    for (size_t i = 0; i < NUM_VTXBUF_HANDLES; ++i) {
        char file[4096] = "";
        sprintf(file, "field-%lu.out", i); // Note: could use vtxbuf_names[i]
        access_vtxbuf_on_disk((VertexBufferHandle)i, file, ACCESS_WRITE);
    }
}

static void
read_async(void)
{
    for (size_t i = 0; i < NUM_VTXBUF_HANDLES; ++i) {
        char file[4096] = "";
        sprintf(file, "field-%lu.out", i); // Note: could use vtxbuf_names[i]
        access_vtxbuf_on_disk((VertexBufferHandle)i, file, ACCESS_READ);
    }
}

AcResult
acGridDiskAccessLaunch(const AccessType type)
{
    ERRCHK_ALWAYS(grid.initialized);
    WARNING("\n------------------------\n"
            "acGridDiskAccessLaunch does not work concurrently with acGridIntegrate due to an\n"
            "unknown issue (invalid CUDA context, double free, or invalid memory access). Suspect\n"
            "some complex interaction with the underlying MPI library and the asynchronous task\n"
            "system. `acGridAccessMeshOnDiskSynchronous` has been tested to work on multiple\n"
            "processes. It is recommended to use that instead in production."
            "\n------------------------\n");

    acGridDiskAccessSync();
    ERRCHK_ALWAYS(!future.valid());

    ERRCHK_ALWAYS(complete);
    complete    = false;
    access_type = type;

    if (type == ACCESS_WRITE) {
        for (size_t i = 0; i < NUM_VTXBUF_HANDLES; ++i)
            volume_copy_to_from_host((VertexBufferHandle)i, ACCESS_WRITE);

        future = std::async(std::launch::async, write_async);
    }
    else if (type == ACCESS_READ) {
        future = std::async(std::launch::async, read_async);
    }
    else {
        ERROR("Unknown access type in acGridDiskAccessLaunch");
        return AC_FAILURE;
    }
    return AC_SUCCESS;
}*/

#define USE_CPP_THREADS (1)
#if USE_CPP_THREADS
#include <thread>
#include <vector>

static std::vector<std::thread> threads;
static bool running = false;

AcResult
acGridDiskAccessSync(void)
{
    ERRCHK(grid.initialized);

    for (auto& thread : threads)
        if (thread.joinable())
            thread.join();

    threads.clear();

    acGridSynchronizeStream(STREAM_ALL);
    running = false;
    return AC_SUCCESS;
}

/*
AcResult
acGridDiskAccessLaunch(const AccessType type)
{
    ERRCHK(grid.initialized);
    ERRCHK_ALWAYS(type == ACCESS_WRITE);
    ERRCHK_ALWAYS(!running)
    running = true;

    for (int i = 0; i < NUM_VTXBUF_HANDLES; ++i) {

        const Device device = grid.device;
        acDeviceSynchronizeStream(device, STREAM_ALL);
        const AcMeshInfo info = acDeviceGetLocalConfig(device);
        // const int3 nn         = acConstructInt3Param(AC_nxgrid, AC_nygrid, AC_nzgrid);
        // const int3 nn_sub     = acConstructInt3Param(AC_nx, AC_ny, AC_nz, info);
        // const int3 offset     = info[AC_multigpu_offset]; // Without halo
        AcReal* host_buffer = grid.submesh.vertex_buffer[i];

        const AcReal* in      = device->vba.on_device.in[i];
        const int3 in_offset  = acConstructInt3Param(AC_nx_min, AC_ny_min, AC_nz_min, info);
        const int3 in_volume  = acConstructInt3Param(AC_mx, AC_my, AC_mz, info);
        AcReal* out           = device->vba.on_device.out[i];
        const int3 out_offset = (int3){0, 0, 0};
        const int3 out_volume = acConstructInt3Param(AC_nx, AC_ny, AC_nz, info);
        acDeviceVolumeCopy(device, STREAM_DEFAULT, in, in_offset, in_volume, out, out_offset,
                           out_volume);
        acDeviceSynchronizeStream(device, STREAM_DEFAULT);

        const size_t bytes = acVertexBufferCompdomainSizeBytes(info);
        acMemcpy(host_buffer, out, bytes, cudaMemcpyDeviceToHost);

        const auto write_async = [](const int device_id, const int i, const AcMeshInfo info,
                                    const AcReal* host_buffer) {
#if USE_PERFSTUBS
            PERFSTUBS_REGISTER_THREAD();
            PERFSTUBS_TIMER_START(_write_timer, "acGridDiskAccessLaunch::write_async");
#endif
            set_device_to_grid_device();

            char path[4096] = "";
            sprintf(path, "%s.out", vtxbuf_names[i]);

            const int3 offset = info[AC_multigpu_offset]; // Without halo
#if USE_DISTRIBUTED_IO

#if USE_POSIX_IO
            char outfile[4096] = "";
            snprintf(outfile, 4096, "segment-%d_%d_%d-%s", offset.x, offset.y, offset.z, path);

            FILE* fp = fopen(outfile, "w");
            ERRCHK_ALWAYS(fp);

            const size_t count         = acVertexBufferCompdomainSize(info);
            const size_t count_written = fwrite(host_buffer, sizeof(AcReal), count, fp);
            ERRCHK_ALWAYS(count_written == count);

            fclose(fp);
#else // Use MPI IO
            MPI_File file;
            int mode           = MPI_MODE_CREATE | MPI_MODE_WRONLY;
            char outfile[4096] = "";
            snprintf(outfile, 4096, "segment-%d_%d_%d-%s", offset.x, offset.y, offset.z, path);
#if AC_VERBOSE
            fprintf(stderr, "Writing %s\n", outfile);
#endif
            int retval = MPI_File_open(MPI_COMM_SELF, outfile, mode, MPI_INFO_NULL, &file);
            ERRCHK_ALWAYS(retval == MPI_SUCCESS);

            MPI_Status status;
            const size_t count = acVertexBufferCompdomainSize(info);
            retval = MPI_File_write(file, host_buffer, count, AC_REAL_MPI_TYPE, &status);
            ERRCHK_ALWAYS(retval == MPI_SUCCESS);

            retval = MPI_File_close(&file);
            ERRCHK_ALWAYS(retval == MPI_SUCCESS);
#endif
#else
            MPI_Datatype subarray;
            const int3 nn          = acConstructInt3Param(AC_nxgrid, AC_nygrid, AC_nzgrid);
            const int3 nn_sub      = acConstructInt3Param(AC_nx, AC_ny, AC_nz, info);
            const int arr_nn[]     = {nn.z, nn.y, nn.x};
            const int arr_nn_sub[] = {nn_sub.z, nn_sub.y, nn_sub.x};
            const int arr_offset[] = {offset.z, offset.y, offset.x};

            // printf(" nn.z     %3i, nn.y     %3i, nn.x     %3i, \n nn_sub.z %3i, nn_sub.y %3i,
            // nn_sub.x %3i, \n offset.z %3i, offset.y %3i, offset.x %3i  \n",
            //         nn.z, nn.y, nn.x, nn_sub.z, nn_sub.y, nn_sub.x, offset.z, offset.y,
            //         offset.x);

            MPI_Type_create_subarray(3, arr_nn, arr_nn_sub, arr_offset, MPI_ORDER_C,
                                     AC_REAL_MPI_TYPE, &subarray);
            MPI_Type_commit(&subarray);

            MPI_File file;

#if AC_VERBOSE
            fprintf(stderr, "Writing %s\n", path);
#endif

            int flags = MPI_MODE_CREATE | MPI_MODE_WRONLY;
            ERRCHK_ALWAYS(MPI_File_open(astaroth_comm, path, flags, MPI_INFO_NULL, &file) ==
                          MPI_SUCCESS); // ISSUE TODO: fails with multiple threads

            ERRCHK_ALWAYS(MPI_File_set_view(file, 0, AC_REAL_MPI_TYPE, subarray, "native",
                                            MPI_INFO_NULL) == MPI_SUCCESS);

            MPI_Status status;

            const size_t nelems = nn_sub.x * nn_sub.y * nn_sub.z;
            ERRCHK_ALWAYS(MPI_File_write_all(file, host_buffer, nelems, AC_REAL_MPI_TYPE,
                                             &status) == MPI_SUCCESS);

            ERRCHK_ALWAYS(MPI_File_close(&file) == MPI_SUCCESS);

            MPI_Type_free(&subarray);
#endif
#if USE_PERFSTUBS
            PERFSTUBS_TIMER_STOP(_write_timer);
#endif
        };

        threads.push_back(std::thread(write_async, device->id, i, info, host_buffer));
        // write_async();
    }

    return AC_SUCCESS;
}
*/

static void
ac_write_slice_metadata(const int step, const AcReal simulation_time)
{
   if(ac_pid() == 0)
   {

         FILE* header_file = fopen("ac_slices_info.csv", step == 0 ? "w" : "a");

         // Header only at the step zero
         if (step == 0) {
             fprintf(header_file,
                     "step,t\n");
         }

         fprintf(header_file, "%d,%g\n",step,simulation_time);
         fclose(header_file);
    }
}

static UNUSED long 
measure_file_size(const char* filepath)
{
    
    FILE* fp_in                   = fopen(filepath, "r");
    ERRCHK_ALWAYS(fp_in);
    fseek(fp_in, 0L, SEEK_END);
    const long measured_size = ftell(fp_in);
    fclose(fp_in);
    return measured_size;
};

static void
ac_make_dir(const std::string dir)
{
    static std::unordered_map<std::string,bool> created_dirs{};
    if(created_dirs.find(dir) != created_dirs.end()) return;
    if (!ac_pid()) {
	const std::string cmd = "mkdir -p " + dir;
        system(cmd.c_str());
    }
    created_dirs[dir] = true;
    MPI_Barrier(astaroth_comm);
}

AcResult
acGridWriteMeshToDiskLaunch(const char* dir, const char* label)
{
    ERRCHK(grid.initialized);
    ERRCHK_ALWAYS(!running)
    ac_make_dir(std::string(dir));
    running = true;

    int non_auxiliary_vtxbuf = -1;
    for(int i = 0; i < NUM_VTXBUF_HANDLES; ++i) if(!vtxbuf_is_auxiliary[i]) non_auxiliary_vtxbuf = i;
    if(non_auxiliary_vtxbuf == -1)
    {
    	fatal("%s", "Can not read snapshot if all Fields are auxiliary!\n");
    }
    for (int i = 0; i < NUM_VTXBUF_HANDLES; ++i) {
        if(!vtxbuf_is_alive[i] || vtxbuf_is_device_only[i]) return AC_NOT_ALLOCATED;
        const Device device = grid.device;
        acDeviceSynchronizeStream(device, STREAM_ALL);
        const AcMeshInfo info = acDeviceGetLocalConfig(device);
        // const int3 nn         = acConstructInt3Param(AC_nxgrid, AC_nygrid, AC_nzgrid);
        // const int3 nn_sub     = acConstructInt3Param(AC_nx, AC_ny, AC_nz, info);
        // const int3 offset     = info[AC_multigpu_offset]; // Without halo
        AcReal* host_buffer = grid.submesh.vertex_buffer[i];

	auto vba = acDeviceGetVBA(device);
        const AcReal* in      = vba.on_device.in[i];
        const Volume in_offset  = acGetMinNN(acGridGetLocalMeshInfo());
	const Volume in_volume = acGetLocalMM(acGridGetLocalMeshInfo());
	

	//TP: this can be done since everything is blocking until the memcpy to host has been finished
        AcReal* out = vba.on_device.out[non_auxiliary_vtxbuf];
	const Volume out_offset = {0, 0, 0,};
	const Volume out_volume = acGetLocalNN(acGridGetLocalMeshInfo());
        acDeviceVolumeCopy(device, STREAM_DEFAULT, in, in_offset, in_volume, out, out_offset,
                           out_volume);
        acDeviceSynchronizeStream(device, STREAM_DEFAULT);

        const size_t bytes = acVertexBufferCompdomainSizeBytes(info,VertexBufferHandle(i));
        acMemcpy(host_buffer, out, bytes, cudaMemcpyDeviceToHost);

        const int3 offset = info[AC_multigpu_offset]; // Without halo
        char filepath[4096];
#if USE_DISTRIBUTED_IO
        sprintf(filepath, "%s/%s-segment-%d-%d-%d-%s.mesh", dir, vtxbuf_names[i], offset.x,
                offset.y, offset.z, label);
#else
        sprintf(filepath, "%s/%s-%s.mesh", dir, vtxbuf_names[i], label);
#endif

        const auto write_async = [filepath, offset, i](const AcMeshInfo info_in,
                                                    const AcReal* host_buffer_in) {

#if USE_PERFSTUBS
            PERFSTUBS_REGISTER_THREAD();
            PERFSTUBS_TIMER_START(_write_timer, "acGridWriteMeshToDiskLaunch::write_async");
#endif

#if USE_DISTRIBUTED_IO
            (void)offset; // Unused
#if USE_POSIX_IO
            FILE* fp = fopen(filepath, "wb");
            ERRCHK_ALWAYS(fp);
            const size_t count         = acVertexBufferCompdomainSize(info_in,VertexBufferHandle(i));
            const size_t count_written = fwrite(host_buffer_in, sizeof(AcReal), count, fp);
            ERRCHK_ALWAYS(count_written == count);

            fclose(fp);
            //const long expected_size = acVertexBufferCompdomainSizeBytes(grid.submesh.info);
	    //const long measured_size = measure_file_size(filepath);
	    //if(expected_size != measured_size)
	    //{
	    //        fprintf(stderr,"Expected output file to be of size (%zu) but it was of size (%zu)!\n",expected_size,measured_size);
	    //        fflush(stderr);
	    //}
	    //ERRCHK_ALWAYS(expected_size == measured_size);
#else // Use MPI IO
            MPI_File file;
            int mode = MPI_MODE_CREATE | MPI_MODE_WRONLY;
            // fprintf(stderr, "Writing %s\n", filepath);
            int retval = MPI_File_open(MPI_COMM_SELF, filepath, mode, MPI_INFO_NULL, &file);
            ERRCHK_ALWAYS(retval == MPI_SUCCESS);

            MPI_Status status;
            const size_t count = acVertexBufferCompdomainSize(info_in,VertexBufferHandle(i));
            retval = MPI_File_write(file, host_buffer_in, count, AC_REAL_MPI_TYPE, &status);
            ERRCHK_ALWAYS(retval == MPI_SUCCESS);

            retval = MPI_File_close(&file);
            ERRCHK_ALWAYS(retval == MPI_SUCCESS);
#endif
#undef USE_POSIX_IO
#else
            WARNING("Collective mesh writing not working with async IO");
            MPI_Datatype subarray;
            const int3 nn          = acConstructInt3Param(AC_nxgrid, AC_nygrid, AC_nzgrid, info_in);
            const int3 nn_sub      = acConstructInt3Param(AC_nx, AC_ny, AC_nz, info_in);
            const int arr_nn[]     = {nn.z, nn.y, nn.x};
            const int arr_nn_sub[] = {nn_sub.z, nn_sub.y, nn_sub.x};
            const int arr_offset[] = {offset.z, offset.y, offset.x};

            // printf(" nn.z     %3i, nn.y     %3i, nn.x     %3i, \n nn_sub.z %3i, nn_sub.y %3i,
            // nn_sub.x %3i, \n offset.z %3i, offset.y %3i, offset.x %3i  \n",
            //         nn.z, nn.y, nn.x, nn_sub.z, nn_sub.y, nn_sub.x, offset.z, offset.y,
            //         offset.x);

            MPI_Type_create_subarray(3, arr_nn, arr_nn_sub, arr_offset, MPI_ORDER_C,
                                     AC_REAL_MPI_TYPE, &subarray);
            MPI_Type_commit(&subarray);

            MPI_File file;
            // fprintf(stderr, "Writing %s\n", filepath);

            int flags = MPI_MODE_CREATE | MPI_MODE_WRONLY;
            ERRCHK_ALWAYS(MPI_File_open(astaroth_comm, filepath, flags, MPI_INFO_NULL, &file) ==
                          MPI_SUCCESS); // ISSUE TODO: fails with multiple threads

            ERRCHK_ALWAYS(MPI_File_set_view(file, 0, AC_REAL_MPI_TYPE, subarray, "native",
                                            MPI_INFO_NULL) == MPI_SUCCESS);

            MPI_Status status;

            const size_t nelems = nn_sub.x * nn_sub.y * nn_sub.z;
            ERRCHK_ALWAYS(MPI_File_write_all(file, host_buffer_in, nelems, AC_REAL_MPI_TYPE,
                                             &status) == MPI_SUCCESS);

            ERRCHK_ALWAYS(MPI_File_close(&file) == MPI_SUCCESS);

            MPI_Type_free(&subarray);
#endif
#if USE_PERFSTUBS
            PERFSTUBS_TIMER_STOP(_write_timer);
#endif
        };

        // write_async(info, host_buffer); // Synchronous, non-threaded
        threads.push_back(std::thread(write_async, info, host_buffer)); // Async, threaded
    }

    return AC_SUCCESS;
}

static AcResult
acGridWriteSlicesToDiskLaunchBase(const char* dir,const int step_number, const AcReal simulation_time, const bool async)
{
    ERRCHK(grid.initialized);
    ERRCHK_ALWAYS(!running);
    running = async;
    ac_make_dir(std::string(dir));
    ac_write_slice_metadata(step_number,simulation_time);
    constexpr size_t label_size = 20000;
    char label[label_size];
    sprintf(label,"step_%012d",step_number);

    const Device device       = grid.device;
    const AcMeshInfo info     = acDeviceGetLocalConfig(device);
    const Volume local_nn = acGetLocalNN(acGridGetLocalMeshInfo());
    const Volume global_nn = get_global_nn();
    const Volume global_offset  = to_volume(info[AC_multigpu_offset]);
    const Volume global_pos_min = global_offset;
    // const int3 global_pos_max = global_pos_min + local_nn;

    //Does work also for 2D
    const int global_z = global_nn.z / 2;
    const int local_z  = global_z - global_pos_min.z;
    const int color    = local_z >= 0 && local_z < (int)local_nn.z ? 0 : MPI_UNDEFINED;

    for (int field = 0; field < NUM_FIELDS; ++field) {


        acDeviceSynchronizeStream(device, STREAM_ALL);

	const Volume slice_volume = (Volume){(size_t)info[AC_nlocal].x, (size_t)info[AC_nlocal].y, 1};
        const int3 slice_offset = (int3){0, 0, local_z};

	auto vba = acDeviceGetVBA(device);
        const AcReal* in     = vba.on_device.in[field];

	const int3 in_offset = acGetMinNN(acGridGetLocalMeshInfo()) + slice_offset;
	const int3 in_volume = to_int3(acGetLocalMM(acGridGetLocalMeshInfo()));
        AcReal* out           = vba.on_device.out[field];
        const Volume out_offset = {0, 0, 0};
        const Volume out_volume = slice_volume;

        if (color != MPI_UNDEFINED)
            acDeviceVolumeCopy(device, STREAM_DEFAULT, in, to_volume(in_offset), to_volume(in_volume), out, out_offset,
                               out_volume);
        acDeviceSynchronizeStream(device, STREAM_DEFAULT);

        AcReal* host_buffer = grid.submesh.vertex_buffer[field];
        const size_t count  = slice_volume.x * slice_volume.y * slice_volume.z;
        const size_t bytes  = sizeof(host_buffer[0]) * count;
        if (color != MPI_UNDEFINED)
            acMemcpy(host_buffer, out, bytes, cudaMemcpyDeviceToHost);

        char filepath[2*label_size];
#if USE_DISTRIBUTED_IO
        sprintf(filepath, "%s/%s-segment-at_%ld_%ld_%d-dims_%ld_%ld-%s.slice", dir, vtxbuf_names[field],
                global_pos_min.x, global_pos_min.y, global_z, local_nn.x, local_nn.y, label);
#else
        sprintf(filepath, "%s/%s-dims_%d_%d-%s.slice", dir, vtxbuf_names[field], global_nn.x,
                global_nn.y, label);
#endif

        // if (color != MPI_UNDEFINED)
        //     fprintf(stderr, "Writing field %d, proc %d, to %s\n", field, pid, filepath);

        acGridSynchronizeStream(STREAM_ALL);
        const auto write_async = [filepath, global_nn, global_pos_min, slice_volume,
                                  color](const AcReal* host_buffer_in, const size_t count_in
                                        ) { 
#if USE_PERFSTUBS
            PERFSTUBS_REGISTER_THREAD();
            PERFSTUBS_TIMER_START(_write_timer, "acGridWriteMeshToDiskLaunch::write_async");
#endif

            set_device_to_grid_device();
            // Write to file

#if USE_DISTRIBUTED_IO
            (void)global_nn;      // Unused
            (void)global_pos_min; // Unused
            (void)slice_volume;   // Unused
#if USE_POSIX_IO
            if (color != MPI_UNDEFINED) {
                FILE* fp = fopen(filepath, "w");
                ERRCHK_ALWAYS(fp);

                const size_t count_written = fwrite(host_buffer_in, sizeof(AcReal), count, fp);
                ERRCHK_ALWAYS(count_written == count_in);

                fclose(fp);
            }
#else // Use MPI IO
            if (color != MPI_UNDEFINED) {
                MPI_File file;
                int mode = MPI_MODE_CREATE | MPI_MODE_WRONLY;
                // fprintf(stderr, "Writing %s\n", filepath);
                int retval = MPI_File_open(MPI_COMM_SELF, filepath, mode, MPI_INFO_NULL, &file);
                ERRCHK_ALWAYS(retval == MPI_SUCCESS);

                MPI_Status status;
                retval = MPI_File_write(file, host_buffer_in, count_in, AC_REAL_MPI_TYPE, &status);
                ERRCHK_ALWAYS(retval == MPI_SUCCESS);

                retval = MPI_File_close(&file);
                ERRCHK_ALWAYS(retval == MPI_SUCCESS);
            }
#endif
#undef USE_POSIX_IO
#else
            ERROR("Collective slice writing not working with async IO");
            // Possible MPI bug: need to cudaSetDevice or otherwise invalid context
            // But also causes a deadlock for some reason
            MPI_Comm slice_communicator;
            MPI_Comm_split(astaroth_comm, color, 0, &slice_communicator);
            if (color != MPI_UNDEFINED) {
                const int3 nn     = (int3){global_nn.x, global_nn.y, 1};
                const int3 nn_sub = slice_volume;

                const int nn_[]     = {nn.z, nn.y, nn.x};
                const int nn_sub_[] = {nn_sub.z, nn_sub.y, nn_sub.x};
                const int offset_[] = {
                    0,
                    global_pos_min.y,
                    global_pos_min.x,
                };
                MPI_Datatype subdomain;
                MPI_Type_create_subarray(3, nn_, nn_sub_, offset_, MPI_ORDER_C, AC_REAL_MPI_TYPE,
                                         &subdomain);
                MPI_Type_commit(&subdomain);

                // printf("Writing %s\n", filepath);

                MPI_File fp;
                int retval = MPI_File_open(slice_communicator, filepath,
                                           MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fp);
                ERRCHK_ALWAYS(retval == MPI_SUCCESS);

                retval = MPI_File_set_view(fp, 0, AC_REAL_MPI_TYPE, subdomain, "native",
                                           MPI_INFO_NULL);
                ERRCHK_ALWAYS(retval == MPI_SUCCESS);

                MPI_Status status;
                retval = MPI_File_write_all(fp, host_buffer_in, count_in, AC_REAL_MPI_TYPE, &status);
                ERRCHK_ALWAYS(retval == MPI_SUCCESS);

                retval = MPI_File_close(&fp);
                ERRCHK_ALWAYS(retval == MPI_SUCCESS);

                MPI_Type_free(&subdomain);

                MPI_Comm_free(&slice_communicator);
            }
#endif

#if USE_PERFSTUBS
            PERFSTUBS_TIMER_STOP(_write_timer);
#endif
        };

	if(async)
	{
        	threads.push_back(
        	    std::thread(write_async, host_buffer, count)); // Async, threaded
	}
	else
	{
        	write_async(host_buffer, count); // Synchronous, non-threaded
	}
    }
    return AC_SUCCESS;
}

AcResult
acGridWriteSlicesToDiskLaunch(const char* dir,const int step_number, const AcReal simulation_time)
{
	return acGridWriteSlicesToDiskLaunchBase(dir,step_number,simulation_time,true);
}

AcResult
acGridWriteSlicesToDiskSynchronous(const char* dir,const int step_number, const AcReal simulation_time)
{
	return acGridWriteSlicesToDiskLaunchBase(dir,step_number,simulation_time,false);
}

AcResult
acGridWriteSlicesToDiskCollectiveSynchronous(const char* dir, const int step_number, const AcReal simulation_time)
{
    ERRCHK(grid.initialized);
    ERRCHK_ALWAYS(!running);
    ac_make_dir(std::string(dir));
    ac_write_slice_metadata(step_number,simulation_time);

    constexpr size_t label_size = 20000;
    char label[label_size];
    sprintf(label,"step_%012d",step_number);

    const Device device       = grid.device;
    const AcMeshInfo info     = acDeviceGetLocalConfig(device);
    const Volume local_nn = acGetLocalNN(acGridGetLocalMeshInfo());
    const Volume global_nn = get_global_nn();
    const Volume global_offset  = to_volume(info[AC_multigpu_offset]);
    const Volume global_pos_min = global_offset;
    // const int3 global_pos_max = global_pos_min + local_nn;

    const int global_z = global_nn.z / 2;
    const int local_z  = global_z - global_pos_min.z;
    const int color    = local_z >= 0 && local_z < (int)local_nn.z ? 0 : MPI_UNDEFINED;

    for (int field = 0; field < NUM_FIELDS; ++field) {

        acDeviceSynchronizeStream(device, STREAM_ALL);

	const Volume slice_volume = {(size_t)info[AC_nlocal].x, (size_t)info[AC_nlocal].y, 1};
        const int3 slice_offset = (int3){0, 0, local_z};

	auto vba = acDeviceGetVBA(device);
	const AcReal* in = vba.on_device.in[field];
	const int3 in_offset = acGetMinNN(acGridGetLocalMeshInfo()) + slice_offset;
	const Volume in_volume = acGetLocalMM(acGridGetLocalMeshInfo());

        AcReal* out           = vba.on_device.out[field];
        const Volume out_offset = {0, 0, 0};
        const Volume out_volume = slice_volume;

        if (color != MPI_UNDEFINED)
            acDeviceVolumeCopy(device, STREAM_DEFAULT, in, to_volume(in_offset), in_volume, out, out_offset,
                               out_volume);
        acDeviceSynchronizeStream(device, STREAM_DEFAULT);

        AcReal* host_buffer = grid.submesh.vertex_buffer[field];
        const size_t count  = slice_volume.x * slice_volume.y * slice_volume.z;
        const size_t bytes  = sizeof(host_buffer[0]) * count;
        if (color != MPI_UNDEFINED)
            acMemcpy(host_buffer, out, bytes, cudaMemcpyDeviceToHost);

        char filepath[2*label_size];
        sprintf(filepath, "%s/%s-dims_%ld_%ld-%s.slice", dir, vtxbuf_names[field], global_nn.x,
                global_nn.y, label);

        // if (color != MPI_UNDEFINED)
        //     fprintf(stderr, "Writing field %d, proc %d, to %s\n", field, pid, filepath);

        acGridSynchronizeStream(STREAM_ALL);
        const auto write_sync = [filepath, global_nn, global_pos_min, slice_volume,
                                 color](const AcReal* host_buffer_in, const size_t count_in
                                       ) {
            set_device_to_grid_device();
            // Write to file

            // Possible MPI bug: need to cudaSetDevice or otherwise invalid context
            // But also causes a deadlock for some reason
            MPI_Comm slice_communicator;
            MPI_Comm_split(astaroth_comm, color, 0, &slice_communicator);
            if (color != MPI_UNDEFINED) {
                const Volume nn     = (Volume){global_nn.x, global_nn.y, 1};
                const Volume nn_sub = slice_volume;

                const int nn_[]     = {(int)nn.z, (int)nn.y, (int)nn.x};
                const int nn_sub_[] = {(int)nn_sub.z, (int)nn_sub.y, (int)nn_sub.x};
                const int offset_[] = {
                    0,
                    (int)global_pos_min.y,
                    (int)global_pos_min.x,
                };
                MPI_Datatype subdomain;
                MPI_Type_create_subarray(3, nn_, nn_sub_, offset_, MPI_ORDER_C, AC_REAL_MPI_TYPE,
                                         &subdomain);
                MPI_Type_commit(&subdomain);

                // printf("Writing %s\n", filepath);

                MPI_File fp;
                int retval = MPI_File_open(slice_communicator, filepath,
                                           MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fp);
                ERRCHK_ALWAYS(retval == MPI_SUCCESS);

                retval = MPI_File_set_view(fp, 0, AC_REAL_MPI_TYPE, subdomain, "native",
                                           MPI_INFO_NULL);
                ERRCHK_ALWAYS(retval == MPI_SUCCESS);

                MPI_Status status;
                retval = MPI_File_write_all(fp, host_buffer_in, count_in, AC_REAL_MPI_TYPE, &status);
                ERRCHK_ALWAYS(retval == MPI_SUCCESS);

                retval = MPI_File_close(&fp);
                ERRCHK_ALWAYS(retval == MPI_SUCCESS);

                MPI_Type_free(&subdomain);

                MPI_Comm_free(&slice_communicator);
            }
        };

        write_sync(host_buffer, count); // Synchronous, non-threaded
                                                    // threads.push_back(std::move(std::thread(write_sync,
                                                    // host_buffer, count, device->id)));
                                                    // // Async, threaded
    }
    return AC_SUCCESS;
}
#else

static MPI_File files[NUM_VTXBUF_HANDLES];
static MPI_Request reqs[NUM_VTXBUF_HANDLES];
static bool req_running[NUM_VTXBUF_HANDLES];

AcResult
acGridDiskAccessSync(void)
{
    ERRCHK(grid.initialized);

    for (size_t i = 0; i < NUM_VTXBUF_HANDLES; ++i) {
        if (req_running[i]) {
            MPI_Wait(&reqs[i], MPI_STATUS_IGNORE);
            const int retval = MPI_File_close(&files[i]);
            ERRCHK_ALWAYS(retval == MPI_SUCCESS);
            req_running[i] = false;
        }
    }
    MPI_Barrier(astaroth_comm);
    return AC_SUCCESS;
}

AcResult
acGridDiskAccessLaunch(const AccessType type)
{
    ERRCHK(grid.initialized);
    ERRCHK_ALWAYS(type == ACCESS_WRITE);

    for (int i = 0; i < NUM_VTXBUF_HANDLES; ++i) {

        ERRCHK_ALWAYS(!reqs[i]);

        const Device device = grid.device;
        set_device_to_grid_device();
        acDeviceSynchronize();

        const AcMeshInfo info = acDeviceGetLocalConfig(device);
        AcReal* host_buffer   = grid.submesh.vertex_buffer[i];

	auto vba = acDeviceGetVBA(device);
        const AcReal* in      = vba.on_device.in[i];
        const int3 in_offset  = acConstructInt3Param(AC_nx_min, AC_ny_min, AC_nz_min, info);
        const int3 in_volume  = acConstructInt3Param(AC_mx, AC_my, AC_mz, info);
        AcReal* out           = vba.on_device.out[i];
        const int3 out_offset = (int3){0, 0, 0};
        const int3 out_volume = acConstructInt3Param(AC_nx, AC_ny, AC_nz, info);
        acDeviceVolumeCopy(device, STREAM_DEFAULT, in, in_offset, in_volume, out, out_offset,
                           out_volume);
        acDeviceSynchronizeStream(device, STREAM_DEFAULT);

        const size_t bytes = acVertexBufferCompdomainSizeBytes(info,VertexBufferHandle(i));
        acMemcpy(host_buffer, out, bytes, cudaMemcpyDeviceToHost);

        char path[4096] = "";
        sprintf(path, "%s.out", vtxbuf_names[i]);

        const int3 offset = info[AC_multigpu_offset]; // Without halo
#if USE_DISTRIBUTED_IO
        int mode           = MPI_MODE_CREATE | MPI_MODE_WRONLY;
        char outfile[4096] = "";
        snprintf(outfile, 4096, "segment-%d_%d_%d-%s", offset.x, offset.y, offset.z, path);

#if AC_VERBOSE
        fprintf(stderr, "Writing %s\n", outfile);
#endif

        int retval = MPI_File_open(MPI_COMM_SELF, outfile, mode, MPI_INFO_NULL, &files[i]);
        ERRCHK_ALWAYS(retval == MPI_SUCCESS);

        const size_t count = acVertexBufferCompdomainSize(info,VertexBufferHandle(i));
        retval = MPI_File_iwrite(files[i], host_buffer, count, AC_REAL_MPI_TYPE, &reqs[i]);
        ERRCHK_ALWAYS(retval == MPI_SUCCESS);

        req_running[i] = true;
#else
        MPI_Datatype subarray;
        const int3 nn     = get_global_nn();
        const int3 nn_sub = acGetLocalNN(acGridGetLocalMeshInfo());
        const int arr_nn[]     = {nn.z, nn.y, nn.x};
        const int arr_nn_sub[] = {nn_sub.z, nn_sub.y, nn_sub.x};
        const int arr_offset[] = {offset.z, offset.y, offset.x};

        MPI_Type_create_subarray(3, arr_nn, arr_nn_sub, arr_offset, MPI_ORDER_C, AC_REAL_MPI_TYPE,
                                 &subarray);
        MPI_Type_commit(&subarray);

#if AC_VERBOSE
        fprintf(stderr, "Writing %s\n", path);
#endif

        int flags  = MPI_MODE_CREATE | MPI_MODE_WRONLY;
        int retval = MPI_File_open(astaroth_comm, path, flags, MPI_INFO_NULL, &files[i]);
        ERRCHK_ALWAYS(retval == MPI_SUCCESS);

        retval = MPI_File_set_view(files[i], 0, AC_REAL_MPI_TYPE, subarray, "native",
                                   MPI_INFO_NULL);
        ERRCHK_ALWAYS(retval == MPI_SUCCESS);

        const size_t nelems = nn_sub.x * nn_sub.y * nn_sub.z;
// Does not work
#if 0   
        retval = MPI_File_iwrite_all(files[i], host_buffer, nelems, AC_REAL_MPI_TYPE, &reqs[i]);
        ERRCHK_ALWAYS(retval == MPI_SUCCESS);
        MPI_Type_free(&subarray);
        req_running[i] = true;
// Does not work either, even though otherwise identical to the blocking version below
#elif 0 
        // (except iwrite + wait)
        ERRCHK_ALWAYS(&files[i]);
        ERRCHK_ALWAYS(&reqs[i]);
        ERRCHK_ALWAYS(host_buffer);
        ERRCHK_ALWAYS(subarray);
        retval = MPI_File_iwrite_all(files[i], host_buffer, nelems, AC_REAL_MPI_TYPE, &reqs[i]);
        ERRCHK_ALWAYS(retval == MPI_SUCCESS);

        retval = MPI_Wait(&reqs[i], MPI_STATUS_IGNORE);
        ERRCHK_ALWAYS(retval == MPI_SUCCESS);

        retval = MPI_File_close(&files[i]);
        ERRCHK_ALWAYS(retval == MPI_SUCCESS);

        MPI_Type_free(&subarray);
        req_running[i] = false;
#else   // Blocking, this works
        WARNING("Called collective non-blocking MPI_File_write_all, but currently blocks\n");
        MPI_Status status;
        retval = MPI_File_write_all(files[i], host_buffer, nelems, AC_REAL_MPI_TYPE, &status);
        ERRCHK_ALWAYS(retval == MPI_SUCCESS);

        retval = MPI_File_close(&files[i]);
        ERRCHK_ALWAYS(retval == MPI_SUCCESS);

        MPI_Type_free(&subarray);
        req_running[i] = false;
#endif
#endif
    }

    return AC_SUCCESS;
}
#endif

static void
check_file_size(const char* filepath)
{
        const long expected_size = acVertexBufferCompdomainSizeBytes(grid.submesh.info);
	const long measured_size = measure_file_size(filepath);
        if (expected_size != measured_size) {
            fprintf(stderr,
                    "Expected size of (%s) did not match measured size (%lu vs %lu), factor of %g "
                    "difference\n",
                    filepath,expected_size, measured_size, (double)expected_size / measured_size);
            fprintf(stderr, "Note that old data files must be removed when switching to a smaller "
                            "mesh size, otherwise the file on disk will be too large (the above "
                            "factor < 1)\n");
            ERRCHK_ALWAYS(expected_size == measured_size);
        }
}

AcResult
acGridAccessMeshOnDiskSynchronous(const VertexBufferHandle vtxbuf, const char* dir,
                                  const char* label, const AccessType type)
{

    if(!vtxbuf_is_alive[vtxbuf] || vtxbuf_is_device_only[vtxbuf]) return AC_NOT_ALLOCATED;
#define BUFFER_DISK_WRITE_THROUGH_CPU (1)

    ERRCHK(grid.initialized);
    acGridSynchronizeStream(STREAM_ALL);
    // acGridDiskAccessSync();

    const Device device   = grid.device;
    const AcMeshInfo info = acDeviceGetLocalConfig(device);
    // const int3 nn         = acConstructInt3Param(AC_nxgrid, AC_nygrid, AC_nzgrid);
    const Volume nn_sub = acGetLocalNN(acGridGetLocalMeshInfo());
    const Volume offset = to_volume(info[AC_multigpu_offset]); // Without halo

    const size_t buflen = 4096;
    char filepath[buflen];
#if USE_DISTRIBUTED_IO
    sprintf(filepath, "%s/%s-segment-%ld-%ld-%ld-%s.mesh", dir, vtxbuf_names[vtxbuf], offset.x,
            offset.y, offset.z, label);
#else
    sprintf(filepath, "%s/%s-%s.mesh", dir, vtxbuf_names[vtxbuf], label);
#endif

#if AC_VERBOSE
    fprintf(stderr, "%s %s\n", type == ACCESS_WRITE ? "Writing" : "Reading", filepath);
#endif

    if (type == ACCESS_WRITE) {
	auto vba = acDeviceGetVBA(device);
        const AcReal* in      = vba.on_device.in[vtxbuf];
	const Volume in_offset = acGetMinNN(acGridGetLocalMeshInfo());
	const Volume in_volume = acGetLocalMM(acGridGetLocalMeshInfo());
        AcReal* out           = vba.on_device.out[vtxbuf];
        const Volume out_offset = (Volume){0, 0, 0};
	const Volume out_volume = acGetLocalNN(acGridGetLocalMeshInfo());
        acDeviceVolumeCopy(device, STREAM_DEFAULT, in, in_offset, in_volume, out, out_offset,
                           out_volume);
        acDeviceSynchronizeStream(device, STREAM_DEFAULT);

// ---------------------------------------
// Buffer through CPU
#if BUFFER_DISK_WRITE_THROUGH_CPU
        const size_t count = acVertexBufferCompdomainSizeBytes(info,VertexBufferHandle(vtxbuf));
        acMemcpy(grid.submesh.vertex_buffer[vtxbuf], out, count, cudaMemcpyDeviceToHost);
#endif
        // ----------------------------------------
    }

#ifndef NDEBUG
    if (type == ACCESS_READ) {
        check_file_size(filepath);
    }
#endif // NDEBUG

#if BUFFER_DISK_WRITE_THROUGH_CPU
    // ---------------------------------------
    // Buffer through CPU
    AcReal* arr = grid.submesh.vertex_buffer[vtxbuf];
    // ----------------------------------------
#else
    auto vba = acDeviceGetVBA(device);
    AcReal* arr = vba.on_device.out[vtxbuf];
#endif

#if USE_DISTRIBUTED_IO
    const size_t nelems = nn_sub.x * nn_sub.y * nn_sub.z;

    FILE* fp;
    if (type == ACCESS_READ)
    {
        fp = fopen(filepath, "rb");
	if(!fp)
	{
		fatal("Could not open file %s\n",filepath);
	}
    }
    else
        fp = fopen(filepath, "wb");

    ERRCHK_ALWAYS(fp);

    if (type == ACCESS_READ)
    {
        fread(arr, sizeof(AcReal), nelems, fp);
    }
    else
    {
        fwrite(arr, sizeof(AcReal), nelems, fp);
    }
    fclose(fp);


#else // Collective IO
    MPI_Datatype subarray;
    const int3 nn = get_global_nn(); // TODO recheck whether this is correct
    const int arr_nn[] = {nn.z, nn.y, nn.x};
    const int arr_nn_sub[] = {nn_sub.z, nn_sub.y, nn_sub.x};
    const int arr_offset[] = {offset.z, offset.y, offset.x};

    // printf(" nn.z     %3i, nn.y     %3i, nn.x     %3i, \n nn_sub.z %3i, nn_sub.y %3i, nn_sub.x
    // %3i, \n offset.z %3i, offset.y %3i, offset.x %3i  \n",
    //         nn.z, nn.y, nn.x, nn_sub.z, nn_sub.y, nn_sub.x, offset.z, offset.y, offset.x);

    MPI_Type_create_subarray(3, arr_nn, arr_nn_sub, arr_offset, MPI_ORDER_C, AC_REAL_MPI_TYPE,
                             &subarray);
    MPI_Type_commit(&subarray);

    MPI_File file;

    int flags = 0;
    if (type == ACCESS_READ)
        flags = MPI_MODE_RDONLY;
    else
        flags = MPI_MODE_CREATE | MPI_MODE_WRONLY;

    ERRCHK_ALWAYS(MPI_File_open(astaroth_comm, filepath, flags, MPI_INFO_NULL, &file) ==
                  MPI_SUCCESS);

    ERRCHK_ALWAYS(MPI_File_set_view(file, 0, AC_REAL_MPI_TYPE, subarray, "native", MPI_INFO_NULL) ==
                  MPI_SUCCESS);

    MPI_Status status;

    const size_t nelems = nn_sub.x * nn_sub.y * nn_sub.z;
    if (type == ACCESS_READ) {
        ERRCHK_ALWAYS(MPI_File_read_all(file, arr, nelems, AC_REAL_MPI_TYPE, &status) ==
                      MPI_SUCCESS);
    }
    else {
        ERRCHK_ALWAYS(MPI_File_write_all(file, arr, nelems, AC_REAL_MPI_TYPE, &status) ==
                      MPI_SUCCESS);
    }

    ERRCHK_ALWAYS(MPI_File_close(&file) == MPI_SUCCESS);

    MPI_Type_free(&subarray);
#endif

    check_file_size(filepath);
    if (type == ACCESS_READ) {
	auto vba = acDeviceGetVBA(device);
	int non_auxiliary_vtxbuf = -1;
	for(int i = 0; i < NUM_VTXBUF_HANDLES; ++i) if(!vtxbuf_is_auxiliary[i]) non_auxiliary_vtxbuf = i;
	if(non_auxiliary_vtxbuf == -1)
	{
		fatal("%s", "Can not read snapshot if all Fields are auxiliary!\n");
	}
	//TP: it is safe to borrow other fields output since this function is blocking
        AcReal* in           = vba.on_device.out[non_auxiliary_vtxbuf];
        const Volume in_offset = {0, 0, 0};
        const Volume in_volume = acGetLocalNN(acGridGetLocalMeshInfo());

        AcReal* out           = vba.on_device.in[vtxbuf];

	const Volume out_offset = acGetMinNN(acGridGetLocalMeshInfo());
	const Volume out_volume = acGetLocalMM(acGridGetLocalMeshInfo());

#if BUFFER_DISK_WRITE_THROUGH_CPU
        // ---------------------------------------
        // Buffer through CPU
        const size_t count = acVertexBufferCompdomainSizeBytes(info,VertexBufferHandle(vtxbuf));
        acMemcpy(in, arr, count, cudaMemcpyHostToDevice);
        //  ----------------------------------------
#endif

        // DEBUG hotfix START
        // TODO better solution (need to recheck all acDevice functions)
        acDeviceSynchronize();             // This sync *is* needed
        acGridSynchronizeStream(STREAM_ALL); // This sync may not be needed
        // DEBUG hotfix END

        acDeviceVolumeCopy(device, STREAM_DEFAULT, in, in_offset, in_volume, out, out_offset,
                           out_volume);
        // Apply boundconds and sync
        if(grid.submesh.info[AC_fully_periodic_grid]) acGridPeriodicBoundconds(STREAM_DEFAULT);
        // acDeviceSynchronizeStream(device, STREAM_ALL);

        // DEBUG hotfix START
        acGridSynchronizeStream(STREAM_ALL); // This sync may not be needed
                                             // DEBUG hotfix END
    }
    return AC_SUCCESS;
}

AcResult
acGridAccessMeshOnDiskSynchronousDistributed(const VertexBufferHandle vtxbuf, const char* dir,
                                             const char* label, const AccessType type)
{
    if(!vtxbuf_is_alive[vtxbuf] || vtxbuf_is_device_only[vtxbuf]) return AC_NOT_ALLOCATED;
#define BUFFER_DISK_WRITE_THROUGH_CPU (1)

    ERRCHK(grid.initialized);
    acGridSynchronizeStream(STREAM_ALL);
    // acGridDiskAccessSync();

    const Device device   = grid.device;
    const AcMeshInfo info = acDeviceGetLocalConfig(device);
    // const int3 nn         = acConstructInt3Param(AC_nxgrid, AC_nygrid, AC_nzgrid);
    const Volume nn_sub = acGetLocalNN(acGridGetLocalMeshInfo());
    const Volume offset = to_volume(info[AC_multigpu_offset]); // Without halo

    const size_t buflen = 4096;
    char filepath[buflen];
    sprintf(filepath, "%s/%s-segment-%ld-%ld-%ld-%s.mesh", dir, vtxbuf_names[vtxbuf], offset.x,
            offset.y, offset.z, label);
#if AC_VERBOSE
    fprintf(stderr, "%s %s\n", type == ACCESS_WRITE ? "Writing" : "Reading", filepath);
#endif

    if (type == ACCESS_WRITE) {
	auto vba = acDeviceGetVBA(device);
        const AcReal* in      = vba.on_device.in[vtxbuf];
	const Volume in_offset = acGetMinNN(acGridGetLocalMeshInfo());
	const Volume in_volume = acGetLocalMM(acGridGetLocalMeshInfo());
        AcReal* out           = vba.on_device.out[vtxbuf];
        const Volume out_offset = (Volume){0, 0, 0};
	const Volume out_volume = acGetLocalNN(acGridGetLocalMeshInfo());
        acDeviceVolumeCopy(device, STREAM_DEFAULT, in, in_offset, in_volume, out, out_offset,
                           out_volume);
        acDeviceSynchronizeStream(device, STREAM_DEFAULT);

// ---------------------------------------
// Buffer through CPU
#if BUFFER_DISK_WRITE_THROUGH_CPU
        const size_t count = acVertexBufferCompdomainSizeBytes(info,VertexBufferHandle(vtxbuf));
        acMemcpy(grid.submesh.vertex_buffer[vtxbuf], out, count, cudaMemcpyDeviceToHost);
#endif
        // ----------------------------------------
    }

#if BUFFER_DISK_WRITE_THROUGH_CPU
    // ---------------------------------------
    // Buffer through CPU
    AcReal* arr = grid.submesh.vertex_buffer[vtxbuf];
    // ----------------------------------------
#else
    auto vba = acDeviceGetVBA(device);
    AcReal* arr = vba.on_device.out[vtxbuf];
#endif

    const size_t nelems = nn_sub.x * nn_sub.y * nn_sub.z;

    FILE* fp;
    if (type == ACCESS_READ)
        fp = fopen(filepath, "r");
    else
        fp = fopen(filepath, "w");
    ERRCHK_ALWAYS(fp);

    if (type == ACCESS_READ)
        fread(arr, sizeof(AcReal), nelems, fp);
    else
        fwrite(arr, sizeof(AcReal), nelems, fp);
    fclose(fp);

    if (type == ACCESS_READ) {
	auto vba = acDeviceGetVBA(device);
        AcReal* in           = vba.on_device.out[vtxbuf];
        const Volume in_offset = (Volume){0, 0, 0};
	const Volume in_volume = acGetLocalNN(acGridGetLocalMeshInfo());

        AcReal* out           = vba.on_device.in[vtxbuf];
	const Volume out_offset = acGetMinNN(acGridGetLocalMeshInfo());
	const Volume out_volume = acGetLocalMM(acGridGetLocalMeshInfo());

#if BUFFER_DISK_WRITE_THROUGH_CPU
        // ---------------------------------------
        // Buffer through CPU
        const size_t count = acVertexBufferCompdomainSizeBytes(info,VertexBufferHandle(vtxbuf));
        acMemcpy(in, arr, count, cudaMemcpyHostToDevice);
        //  ----------------------------------------
#endif

        // DEBUG hotfix START
        // TODO better solution (need to recheck all acDevice functions)
        acDeviceSynchronize();             // This sync *is* needed
        acGridSynchronizeStream(STREAM_ALL); // This sync may not be needed
        // DEBUG hotfix END

        acDeviceVolumeCopy(device, STREAM_DEFAULT, in, in_offset, in_volume, out, out_offset,
                           out_volume);
        // Apply boundconds and sync
        if(grid.submesh.info[AC_fully_periodic_grid]) acGridPeriodicBoundconds(STREAM_DEFAULT);
        // acDeviceSynchronizeStream(device, STREAM_ALL);

        // DEBUG hotfix START
        acGridSynchronizeStream(STREAM_ALL); // This sync may not be needed
                                             // DEBUG hotfix END
    }
    return AC_SUCCESS;
}

AcResult
acGridAccessMeshOnDiskSynchronousCollective(const VertexBufferHandle vtxbuf, const char* dir,
                                            const char* label, const AccessType type)
{
    if(!vtxbuf_is_alive[vtxbuf] || vtxbuf_is_device_only[vtxbuf]) return AC_NOT_ALLOCATED;
#define BUFFER_DISK_WRITE_THROUGH_CPU (1)

    ERRCHK(grid.initialized);
    acGridSynchronizeStream(STREAM_ALL);
    // acGridDiskAccessSync();

    const Device device   = grid.device;
    const AcMeshInfo info = acDeviceGetLocalConfig(device);
    const Volume nn_sub = acGetLocalNN(acGridGetLocalMeshInfo());
    const Volume nn     = get_global_nn();
    const Volume offset     = to_volume(info[AC_multigpu_offset]); // Without halo

    const size_t buflen = 4096;
    char filepath[buflen];
    sprintf(filepath, "%s/%s-%s.mesh", dir, vtxbuf_names[vtxbuf], label);
#if AC_VERBOSE
    fprintf(stderr, "%s %s\n", type == ACCESS_WRITE ? "Writing" : "Reading", filepath);
#endif

    if (type == ACCESS_WRITE) {
	auto vba = acDeviceGetVBA(device);
        const AcReal* in      = vba.on_device.in[vtxbuf];
	const Volume in_offset = acGetMinNN(acGridGetLocalMeshInfo());
	const Volume in_volume = acGetLocalMM(acGridGetLocalMeshInfo());
        AcReal* out           = vba.on_device.out[vtxbuf];
        const Volume out_offset = {0, 0, 0};
        const Volume out_volume = nn_sub;
        acDeviceVolumeCopy(device, STREAM_DEFAULT, in, in_offset, in_volume, out, out_offset,
                           out_volume);
        acDeviceSynchronizeStream(device, STREAM_DEFAULT);

// ---------------------------------------
// Buffer through CPU
#if BUFFER_DISK_WRITE_THROUGH_CPU
        const size_t count = acVertexBufferCompdomainSizeBytes(info,VertexBufferHandle(vtxbuf));
        acMemcpy(grid.submesh.vertex_buffer[vtxbuf], out, count, cudaMemcpyDeviceToHost);
#endif
        // ----------------------------------------
    }

#if BUFFER_DISK_WRITE_THROUGH_CPU
    // ---------------------------------------
    // Buffer through CPU
    AcReal* arr = grid.submesh.vertex_buffer[vtxbuf];
    // ----------------------------------------
#else
    auto vba = acDeviceGetVBA(device);
    AcReal* arr = vba.on_device.out[vtxbuf];
#endif

    MPI_Datatype subarray;
    const int arr_nn[]     = {(int)nn.z, (int)nn.y, (int)nn.x};
    const int arr_nn_sub[] = {(int)nn_sub.z, (int)nn_sub.y, (int)nn_sub.x};
    const int arr_offset[] = {(int)offset.z, (int)offset.y, (int)offset.x};

    // printf(" nn.z     %3i, nn.y     %3i, nn.x     %3i, \n nn_sub.z %3i, nn_sub.y %3i, nn_sub.x
    // %3i, \n offset.z %3i, offset.y %3i, offset.x %3i  \n",
    //         nn.z, nn.y, nn.x, nn_sub.z, nn_sub.y, nn_sub.x, offset.z, offset.y, offset.x);

    MPI_Type_create_subarray(3, arr_nn, arr_nn_sub, arr_offset, MPI_ORDER_C, AC_REAL_MPI_TYPE,
                             &subarray);
    MPI_Type_commit(&subarray);

    MPI_File file;

    int flags = 0;
    if (type == ACCESS_READ)
        flags = MPI_MODE_RDONLY;
    else
        flags = MPI_MODE_CREATE | MPI_MODE_WRONLY;

    ERRCHK_ALWAYS(MPI_File_open(astaroth_comm, filepath, flags, MPI_INFO_NULL, &file) ==
                  MPI_SUCCESS);

    ERRCHK_ALWAYS(MPI_File_set_view(file, 0, AC_REAL_MPI_TYPE, subarray, "native", MPI_INFO_NULL) ==
                  MPI_SUCCESS);

    MPI_Status status;

    const size_t nelems = nn_sub.x * nn_sub.y * nn_sub.z;
    if (type == ACCESS_READ) {
        ERRCHK_ALWAYS(MPI_File_read_all(file, arr, nelems, AC_REAL_MPI_TYPE, &status) ==
                      MPI_SUCCESS);
    }
    else {
        ERRCHK_ALWAYS(MPI_File_write_all(file, arr, nelems, AC_REAL_MPI_TYPE, &status) ==
                      MPI_SUCCESS);
    }

    ERRCHK_ALWAYS(MPI_File_close(&file) == MPI_SUCCESS);

    MPI_Type_free(&subarray);

    if (type == ACCESS_READ) {
	auto vba = acDeviceGetVBA(device);
        AcReal* in           = vba.on_device.out[vtxbuf];
        const Volume in_offset = {0, 0, 0};
        const Volume in_volume = nn_sub;

        AcReal* out           = vba.on_device.in[vtxbuf];
	const Volume out_offset = acGetMinNN(acGridGetLocalMeshInfo());
	const Volume out_volume = acGetLocalMM(acGridGetLocalMeshInfo());

#if BUFFER_DISK_WRITE_THROUGH_CPU
        // ---------------------------------------
        // Buffer through CPU
        const size_t count = acVertexBufferCompdomainSizeBytes(info,VertexBufferHandle(vtxbuf));
        acMemcpy(in, arr, count, cudaMemcpyHostToDevice);
        //  ----------------------------------------
#endif

        // DEBUG hotfix START
        // TODO better solution (need to recheck all acDevice functions)
        acDeviceSynchronize();             // This sync *is* needed
        acGridSynchronizeStream(STREAM_ALL); // This sync may not be needed
        // DEBUG hotfix END

        acDeviceVolumeCopy(device, STREAM_DEFAULT, in, in_offset, in_volume, out, out_offset,
                           out_volume);
        // Apply boundconds and sync
        if(grid.submesh.info[AC_fully_periodic_grid]) acGridPeriodicBoundconds(STREAM_DEFAULT);
        // acDeviceSynchronizeStream(device, STREAM_ALL);

        // DEBUG hotfix START
        acGridSynchronizeStream(STREAM_ALL); // This sync may not be needed
                                             // DEBUG hotfix END
    }
    return AC_SUCCESS;
}



AcResult
acGridReadVarfileToMesh(const char* file, const Field fields[], const size_t num_fields,
                        const int3 nn, const int3 rr)
{
    // Ensure the library state is ready
    ERRCHK_ALWAYS(grid.initialized);
    acGridSynchronizeStream(STREAM_ALL);

    // Derive the input mesh dimensions
    const int3 mm = (int3){
        nn.x + 2 * rr.x,
        nn.y + 2 * rr.y,
        nn.z + 2 * rr.z,
    };
    const size_t field_offset = (size_t)mm.x * (size_t)mm.y * (size_t)mm.z;

    // Set the helper variables
    const Device device         = grid.device;
    const AcMeshInfo info       = acDeviceGetLocalConfig(device);
    const Volume subdomain_nn = acGetLocalNN(acGridGetLocalMeshInfo());
    const Volume subdomain_offset = to_volume(info[AC_multigpu_offset]); // Without halo
    int retval;

    // Load the fields to host memory
    MPI_Datatype subdomain;
    const int domain_mm_[]        = {mm.z, mm.y, mm.x};
    const int subdomain_nn_[]     = {(int)subdomain_nn.z, (int)subdomain_nn.y, (int)subdomain_nn.x};
    const int subdomain_offset_[] = {
        (int)(rr.z + subdomain_offset.z),
        (int)(rr.y + subdomain_offset.y),
        (int)(rr.x + subdomain_offset.x),
    }; // Offset the ghost zone

    MPI_Type_create_subarray(3, domain_mm_, subdomain_nn_, subdomain_offset_, MPI_ORDER_C,
                             AC_REAL_MPI_TYPE, &subdomain);
    MPI_Type_commit(&subdomain);

    MPI_File fp;
    retval = MPI_File_open(astaroth_comm, file, MPI_MODE_RDONLY, MPI_INFO_NULL, &fp);
    ERRCHK_ALWAYS(retval == MPI_SUCCESS);

    for (size_t i = 0; i < num_fields; ++i) {
    	if(!vtxbuf_is_alive[i] || vtxbuf_is_device_only[i]) continue;
        const Field field = fields[i];

        // Load from file to host memory
        AcReal* host_buffer       = grid.submesh.vertex_buffer[field];
        const size_t displacement = i * field_offset * sizeof(AcReal); // Bytes

        retval = MPI_File_set_view(fp, displacement, AC_REAL_MPI_TYPE, subdomain, "native",
                                   MPI_INFO_NULL);
        ERRCHK_ALWAYS(retval == MPI_SUCCESS);

        MPI_Status status;
        const size_t count = acVertexBufferCompdomainSize(info,VertexBufferHandle(i));
        // retval             = MPI_File_read_all(fp, host_buffer, count, AC_REAL_MPI_TYPE,
        // &status);
        retval = MPI_File_read(fp, host_buffer, count, AC_REAL_MPI_TYPE, &status); // workaround
        ERRCHK_ALWAYS(retval == MPI_SUCCESS);

        /*
        for (size_t kk = 0; kk < subdomain_nn.z; ++kk) {
            for (size_t jj = 0; jj < subdomain_nn.y; ++jj) {
                for (size_t ii = 0; ii < subdomain_nn.x; ++ii) {
                    const size_t idx = ii + jj * subdomain_nn.x + kk * subdomain_nn.x *
        subdomain_nn.y; host_buffer[idx] = (ii+subdomain_offset.x) + (jj+subdomain_offset.y);
                }
            }
        }
        */

        // Load from host memory to device memory
	auto vba = acDeviceGetVBA(device);
        AcReal* in           = vba.on_device.out[field];
        const Volume in_offset = {0, 0, 0};
        const Volume in_volume = subdomain_nn;

        AcReal* out           = vba.on_device.in[field];
	const Volume out_offset = acGetMinNN(acGridGetLocalMeshInfo());
	const Volume out_volume = acGetLocalMM(acGridGetLocalMeshInfo());
        const size_t bytes = acVertexBufferCompdomainSizeBytes(info,VertexBufferHandle(field));
        acMemcpy(in, host_buffer, bytes, cudaMemcpyHostToDevice);
        retval = acDeviceVolumeCopy(device, (Stream)field, in, in_offset, in_volume, out, out_offset,
                                    out_volume);
        ERRCHK_ALWAYS(retval == AC_SUCCESS);
    }
    acGridSynchronizeStream(STREAM_ALL);
    if(grid.submesh.info[AC_fully_periodic_grid]) acGridPeriodicBoundconds(STREAM_DEFAULT);
    acGridSynchronizeStream(STREAM_ALL);

    return AC_SUCCESS;
}
bool
acGridTaskGraphHasPeriodicBoundcondsX(AcTaskGraph* graph)
{
    return (graph->periodic_boundaries & BOUNDARY_X) != 0;
}

bool
acGridTaskGraphHasPeriodicBoundcondsY(AcTaskGraph* graph)
{
    return (graph->periodic_boundaries & BOUNDARY_Y) != 0;
}

bool
acGridTaskGraphHasPeriodicBoundcondsZ(AcTaskGraph* graph)
{
    return (graph->periodic_boundaries & BOUNDARY_Z) != 0;
}

VertexBufferArray
acGridGetVBA(void)
{
    return acDeviceGetVBA(grid.device);
}
/*
AcResult
acGridLoadFieldFromFile(const char* path, const VertexBufferHandle vtxbuf)
{
    ERRCHK(grid.initialized);

    acGridDiskAccessSync();

    const Device device   = grid.device;
    const AcMeshInfo info = acDeviceGetLocalConfig(device);
    const int3 global_nn  = acConstructInt3Param(AC_nxgrid, AC_nygrid, AC_nzgrid);
    const int3 global_mm  = (int3){
        2 * STENCIL_ORDER + global_nn.x,
        2 * STENCIL_ORDER + global_nn.y,
        2 * STENCIL_ORDER + global_nn.z,
    };
    const int3 local_nn         = acConstructInt3Param(AC_nx, AC_ny, AC_nz, info);
    const int3 global_nn_offset = info[AC_multigpu_offset];

    MPI_Datatype subarray;
    const int mm[]     = {global_mm.z, global_mm.y, global_mm.x};
    const int nn_sub[] = {local_nn.z, local_nn.y, local_nn.x};
    const int offset[] = {
        STENCIL_ORDER + global_nn_offset.z,
        STENCIL_ORDER + global_nn_offset.y,
        STENCIL_ORDER + global_nn_offset.x,
    };
    MPI_Type_create_subarray(3, mm, nn_sub, offset, MPI_ORDER_C, AC_REAL_MPI_TYPE, &subarray);
    MPI_Type_commit(&subarray);

    MPI_File file;
    const int flags = MPI_MODE_RDONLY;
    ERRCHK_ALWAYS(MPI_File_open(astaroth_comm, path, flags, MPI_INFO_NULL, &file) == MPI_SUCCESS);
    ERRCHK_ALWAYS(MPI_File_set_view(file, 0, AC_REAL_MPI_TYPE, subarray, "native", MPI_INFO_NULL) ==
                  MPI_SUCCESS);

    MPI_Status status;

    AcReal* arr        = grid.submesh.vertex_buffer[vtxbuf];
    const size_t count = nn_sub[0] * nn_sub[1] * nn_sub[2];
    ERRCHK_ALWAYS(MPI_File_read_all(file, arr, count, AC_REAL_MPI_TYPE, &status) == MPI_SUCCESS);
    ERRCHK_ALWAYS(MPI_File_close(&file) == MPI_SUCCESS);

    MPI_Type_free(&subarray);

    AcReal* in         = device->vba.on_device.out[vtxbuf]; // Note swapped order (vba.on_device.out)
    const size_t bytes = sizeof(in[0]) * count;
    acMemcpy(in, arr, bytes, cudaMemcpyHostToDevice);

    const int3 in_offset = (int3){0, 0, 0};
    const int3 in_volume = acConstructInt3Param(AC_nx, AC_ny, AC_nz, info);

    AcReal* out           = device->vba.on_device.in[vtxbuf]; // Note swapped order (vba.on_device.in)
    const int3 out_offset = acConstructInt3Param(AC_nx_min, AC_ny_min, AC_nz_min, info);
    const int3 out_volume = acConstructInt3Param(AC_mx, AC_my, AC_mz, info);
    acDeviceVolumeCopy(device, STREAM_DEFAULT, in, in_offset, in_volume, out, out_offset,
                       out_volume);

    // Update halos
    acGridPeriodicBoundconds(STREAM_DEFAULT);
    acDeviceSynchronizeStream(device, STREAM_DEFAULT);

    return AC_SUCCESS;
}

AcResult
acGridStoreFieldToFile(const char* path, const VertexBufferHandle vtxbuf)
{
    ERRCHK(grid.initialized);

    const Device device   = grid.device;
    const AcMeshInfo info = acDeviceGetLocalConfig(device);

    AcReal* in           = device->vba.on_device.in[vtxbuf];
    const int3 in_offset = acConstructInt3Param(AC_nx_min, AC_ny_min, AC_nz_min, info);
    const int3 in_volume = acConstructInt3Param(AC_mx, AC_my, AC_mz, info);

    AcReal* out           = device->vba.on_device.out[vtxbuf];
    const int3 out_offset = (int3){0, 0, 0};
    const int3 out_volume = acConstructInt3Param(AC_nx, AC_ny, AC_nz, info);

    acDeviceVolumeCopy(device, STREAM_DEFAULT, in, in_offset, in_volume, out, out_offset,
                       out_volume);

    AcReal* arr        = grid.submesh.vertex_buffer[vtxbuf];
    const size_t bytes = sizeof(in[0]) * out_volume.x * out_volume.y * out_volume.z;
    acMemcpy(in, arr, bytes, cudaMemcpyHostToDevice);

    acGridDiskAccessSync();

    const int3 global_nn = acConstructInt3Param(AC_nxgrid, AC_nygrid, AC_nzgrid);
    const int3 global_mm = (int3){
        2 * STENCIL_ORDER + global_nn.x,
        2 * STENCIL_ORDER + global_nn.y,
        2 * STENCIL_ORDER + global_nn.z,
    };
    const int3 local_nn         = acConstructInt3Param(AC_nx, AC_ny, AC_nz, info);
    const int3 global_nn_offset = info[AC_multigpu_offset];

    MPI_Datatype subarray;
    const int mm[]     = {global_mm.z, global_mm.y, global_mm.x};
    const int nn_sub[] = {local_nn.z, local_nn.y, local_nn.x};
    const int offset[] = {
        STENCIL_ORDER + global_nn_offset.z,
        STENCIL_ORDER + global_nn_offset.y,
        STENCIL_ORDER + global_nn_offset.x,
    };
    MPI_Type_create_subarray(3, mm, nn_sub, offset, MPI_ORDER_C, AC_REAL_MPI_TYPE, &subarray);
    MPI_Type_commit(&subarray);

    MPI_File file;
    const int flags = MPI_MODE_CREATE | MPI_MODE_WRONLY;
    ERRCHK_ALWAYS(MPI_File_open(astaroth_comm, path, flags, MPI_INFO_NULL, &file) == MPI_SUCCESS);
    ERRCHK_ALWAYS(MPI_File_set_view(file, 0, AC_REAL_MPI_TYPE, subarray, "native", MPI_INFO_NULL) ==
                  MPI_SUCCESS);

    MPI_Status status;

    const size_t count = nn_sub[0] * nn_sub[1] * nn_sub[2];
    ERRCHK_ALWAYS(MPI_File_write_all(file, arr, count, AC_REAL_MPI_TYPE, &status) == MPI_SUCCESS);
    ERRCHK_ALWAYS(MPI_File_close(&file) == MPI_SUCCESS);

    MPI_Type_free(&subarray);
    return AC_SUCCESS;
}
*/

/*   MV: Commented out for a while, but save for the future when standalone_MPI
         works with periodic boundary conditions.
AcResult
acGridGeneralBoundconds(const Device device, const Stream stream)
{
    // Non-periodic Boundary conditions
    // Check the position in MPI frame
    int nprocs, pid;
    MPI_Comm_size(astaroth_comm, &nprocs);
    MPI_Comm_rank(astaroth_comm, &pid);
    const uint3_64 decomposition = decompose(nprocs);
    const int3 pid3d             = getPid3D(pid, decomposition);

    // Set outer boudaries after substep computation.
    const int3 m1 = (int3){0, 0, 0};
    const int3 m2 = grid.nn;
    const int3 pid3d = getPid3D(pid, decomposition);
    // If we are are a boundary element
    int3 bindex = (int3){0, 0, 0};

    // Check if there are active boundary condition edges.
    // 0 is no boundary, 1 both edges, 2 is top edge, 3 bottom edge
    if      ((pid3d.x == 0) && (pid3d.x == decomposition.x - 1)) { bindex.x = 1; }
    else if  (pid3d.x == 0)                                      { bindex.x = 2; }
    else if                    (pid3d.x == decomposition.x - 1)  { bindex.x = 3; }

    if      ((pid3d.y == 0) && (pid3d.y == decomposition.y - 1)) { bindex.y = 1; }
    else if  (pid3d.y == 0)                                      { bindex.y = 2; }
    else if                    (pid3d.y == decomposition.y - 1)  { bindex.y = 3; }

    if      ((pid3d.z == 0) && (pid3d.z == decomposition.z - 1)) { bindex.z = 1; }
    else if  (pid3d.z == 0)                                      { bindex.z = 2; }
    else if                    (pid3d.z == decomposition.z - 1)  { bindex.z = 3; }


    if (bindex.x != 1) && (bindex.y != 1) && (bindex.z != 1) {
        acDeviceGeneralBoundconds(device, stream, m1, m2, bindex);
    }
    acGridSynchronizeStream(stream);

    return AC_SUCCESS;
}
*/

/*   MV: Commented out for a while, but save for the future when standalone_MPI
         works with periodic boundary conditions.
AcResult
acGridIntegrateNonperiodic(const Stream stream, const AcReal dt)
{
    ERRCHK(grid.initialized);
    acGridSynchronizeStream(stream);

    const Device device = grid.device;
    const int3 nn       = grid.nn;
#if MPI_INCL_CORNERS
    CommData corner_data = grid.corner_data; // Do not rm: required for corners
#endif                                       // MPI_INCL_CORNERS
    CommData edgex_data  = grid.edgex_data;
    CommData edgey_data  = grid.edgey_data;
    CommData edgez_data  = grid.edgez_data;
    CommData sidexy_data = grid.sidexy_data;
    CommData sidexz_data = grid.sidexz_data;
    CommData sideyz_data = grid.sideyz_data;

    acGridLoadScalarUniform(stream, AC_dt, dt);
    acDeviceSynchronizeStream(device, stream);


// Corners
#if MPI_INCL_CORNERS
    // Do not rm: required for corners
    const int3 corner_b0s[] = {
        (int3){0, 0, 0},
        (int3){NGHOST + nn.x, 0, 0},
        (int3){0, NGHOST + nn.y, 0},
        (int3){0, 0, NGHOST + nn.z},

        (int3){NGHOST + nn.x, NGHOST + nn.y, 0},
        (int3){NGHOST + nn.x, 0, NGHOST + nn.z},
        (int3){0, NGHOST + nn.y, NGHOST + nn.z},
        (int3){NGHOST + nn.x, NGHOST + nn.y, NGHOST + nn.z},
    };
#endif // MPI_INCL_CORNERS

    // Edges X
    const int3 edgex_b0s[] = {
        (int3){NGHOST, 0, 0},
        (int3){NGHOST, NGHOST + nn.y, 0},

        (int3){NGHOST, 0, NGHOST + nn.z},
        (int3){NGHOST, NGHOST + nn.y, NGHOST + nn.z},
    };

    // Edges Y
    const int3 edgey_b0s[] = {
        (int3){0, NGHOST, 0},
        (int3){NGHOST + nn.x, NGHOST, 0},

        (int3){0, NGHOST, NGHOST + nn.z},
        (int3){NGHOST + nn.x, NGHOST, NGHOST + nn.z},
    };

    // Edges Z
    const int3 edgez_b0s[] = {
        (int3){0, 0, NGHOST},
        (int3){NGHOST + nn.x, 0, NGHOST},

        (int3){0, NGHOST + nn.y, NGHOST},
        (int3){NGHOST + nn.x, NGHOST + nn.y, NGHOST},
    };

    // Sides XY
    const int3 sidexy_b0s[] = {
        (int3){NGHOST, NGHOST, 0},             //
        (int3){NGHOST, NGHOST, NGHOST + nn.z}, //
    };

    // Sides XZ
    const int3 sidexz_b0s[] = {
        (int3){NGHOST, 0, NGHOST},             //
        (int3){NGHOST, NGHOST + nn.y, NGHOST}, //
    };

    // Sides YZ
    const int3 sideyz_b0s[] = {
        (int3){0, NGHOST, NGHOST},             //
        (int3){NGHOST + nn.x, NGHOST, NGHOST}, //
    };

    for (int isubstep = 0; isubstep < 3; ++isubstep) {

#if MPI_COMM_ENABLED
#if MPI_INCL_CORNERS
        acPackCommData(device, corner_b0s, &corner_data); // Do not rm: required for corners
#endif                                                    // MPI_INCL_CORNERS
        acPackCommData(device, edgex_b0s, &edgex_data);
        acPackCommData(device, edgey_b0s, &edgey_data);
        acPackCommData(device, edgez_b0s, &edgez_data);
        acPackCommData(device, sidexy_b0s, &sidexy_data);
        acPackCommData(device, sidexz_b0s, &sidexz_data);
        acPackCommData(device, sideyz_b0s, &sideyz_data);
#endif

#if MPI_COMM_ENABLED
        MPI_Barrier(astaroth_comm);

#if MPI_GPUDIRECT_DISABLED
#if MPI_INCL_CORNERS
        acTransferCommDataToHost(device, &corner_data); // Do not rm: required for corners
#endif                                                  // MPI_INCL_CORNERS
        acTransferCommDataToHost(device, &edgex_data);
        acTransferCommDataToHost(device, &edgey_data);
        acTransferCommDataToHost(device, &edgez_data);
        acTransferCommDataToHost(device, &sidexy_data);
        acTransferCommDataToHost(device, &sidexz_data);
        acTransferCommDataToHost(device, &sideyz_data);
#endif
#if MPI_INCL_CORNERS
        acTransferCommData(device, corner_b0s, &corner_data); // Do not rm: required for corners
#endif                                                        // MPI_INCL_CORNERS
        acTransferCommData(device, edgex_b0s, &edgex_data);
        acTransferCommData(device, edgey_b0s, &edgey_data);
        acTransferCommData(device, edgez_b0s, &edgez_data);
        acTransferCommData(device, sidexy_b0s, &sidexy_data);
        acTransferCommData(device, sidexz_b0s, &sidexz_data);
        acTransferCommData(device, sideyz_b0s, &sideyz_data);
#endif // MPI_COMM_ENABLED

#if MPI_COMPUTE_ENABLED
        //////////// INNER INTEGRATION //////////////
        {
            const int3 m1 = (int3){2 * NGHOST, 2 * NGHOST, 2 * NGHOST};
            const int3 m2 = nn;
            acKernelIntegrateSubstep(device->streams[STREAM_16], isubstep, m1, m2, device->vba);
        }
////////////////////////////////////////////
#endif // MPI_COMPUTE_ENABLED

#if MPI_COMM_ENABLED
#if MPI_INCL_CORNERS
        acTransferCommDataWait(corner_data); // Do not rm: required for corners
#endif                                       // MPI_INCL_CORNERS
        acTransferCommDataWait(edgex_data);
        acTransferCommDataWait(edgey_data);
        acTransferCommDataWait(edgez_data);
        acTransferCommDataWait(sidexy_data);
        acTransferCommDataWait(sidexz_data);
        acTransferCommDataWait(sideyz_data);

#if MPI_INCL_CORNERS
        acUnpinCommData(device, &corner_data); // Do not rm: required for corners
#endif                                         // MPI_INCL_CORNERS
        acUnpinCommData(device, &edgex_data);
        acUnpinCommData(device, &edgey_data);
        acUnpinCommData(device, &edgez_data);
        acUnpinCommData(device, &sidexy_data);
        acUnpinCommData(device, &sidexz_data);
        acUnpinCommData(device, &sideyz_data);

#if MPI_INCL_CORNERS
        acUnpackCommData(device, corner_b0s, &corner_data);
#endif // MPI_INCL_CORNERS
        acUnpackCommData(device, edgex_b0s, &edgex_data);
        acUnpackCommData(device, edgey_b0s, &edgey_data);
        acUnpackCommData(device, edgez_b0s, &edgez_data);
        acUnpackCommData(device, sidexy_b0s, &sidexy_data);
        acUnpackCommData(device, sidexz_b0s, &sidexz_data);
        acUnpackCommData(device, sideyz_b0s, &sideyz_data);
//////////// OUTER INTEGRATION //////////////

// Wait for unpacking
#if MPI_INCL_CORNERS
        acSyncCommData(corner_data); // Do not rm: required for corners
#endif                               // MPI_INCL_CORNERS
        acSyncCommData(edgex_data);
        acSyncCommData(edgey_data);
        acSyncCommData(edgez_data);
        acSyncCommData(sidexy_data);
        acSyncCommData(sidexz_data);
        acSyncCommData(sideyz_data);
#endif // MPI_COMM_ENABLED

        // Invoke outer edge boundary conditions.
        acGridGeneralBoundconds(device, stream)

#if MPI_COMPUTE_ENABLED
        { // Front
            const int3 m1 = (int3){NGHOST, NGHOST, NGHOST};
            const int3 m2 = m1 + (int3){nn.x, nn.y, NGHOST};
            acKernelIntegrateSubstep(device->streams[STREAM_0], isubstep, m1, m2, device->vba);
        }
        { // Back
            const int3 m1 = (int3){NGHOST, NGHOST, nn.z};
            const int3 m2 = m1 + (int3){nn.x, nn.y, NGHOST};
            acKernelIntegrateSubstep(device->streams[STREAM_1], isubstep, m1, m2, device->vba);
        }
        { // Bottom
            const int3 m1 = (int3){NGHOST, NGHOST, 2 * NGHOST};
            const int3 m2 = m1 + (int3){nn.x, NGHOST, nn.z - 2 * NGHOST};
            acKernelIntegrateSubstep(device->streams[STREAM_2], isubstep, m1, m2, device->vba);
        }
        { // Top
            const int3 m1 = (int3){NGHOST, nn.y, 2 * NGHOST};
            const int3 m2 = m1 + (int3){nn.x, NGHOST, nn.z - 2 * NGHOST};
            acKernelIntegrateSubstep(device->streams[STREAM_3], isubstep, m1, m2, device->vba);
        }
        { // Left
            const int3 m1 = (int3){NGHOST, 2 * NGHOST, 2 * NGHOST};
            const int3 m2 = m1 + (int3){NGHOST, nn.y - 2 * NGHOST, nn.z - 2 * NGHOST};
            acKernelIntegrateSubstep(device->streams[STREAM_4], isubstep, m1, m2, device->vba);
        }
        { // Right
            const int3 m1 = (int3){nn.x, 2 * NGHOST, 2 * NGHOST};
            const int3 m2 = m1 + (int3){NGHOST, nn.y - 2 * NGHOST, nn.z - 2 * NGHOST};
            acKernelIntegrateSubstep(device->streams[STREAM_5], isubstep, m1, m2, device->vba);
        }
#endif // MPI_COMPUTE_ENABLED
        acDeviceSwapBuffers(device);
        acDeviceSynchronizeStream(device, STREAM_ALL); // Wait until inner and outer done
        ////////////////////////////////////////////

    }

    return AC_SUCCESS;
}
*/
#endif // AC_MPI_ENABLED
