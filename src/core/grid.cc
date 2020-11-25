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
 * struct Grid contains information about the local GPU device, decomposition, the
 *             total mesh dimensions and CommDatas
 
 * Basic steps:
 *   1) Distribute the mesh among ranks
 *   2) Integrate & communicate
 *     - start inner integration and at the same time, pack halo data and send it to neighbors
 *     - as halo data is received and unpacked, integrate segments whose dependencies are ready
 *     - sync and start again
 *   3) Gather the mesh to rank 0 for postprocessing
 *
 * This file contains the grid interface, with algorithms and high level functionality
 * The nitty gritty of the MPI communication is defined in task.cc
 */


#include "astaroth.h"
#include "task.h"

#include <mpi.h>
#include <vector>
#include <utility>
#include <cstring>

#include "decomposition.h" //getPid3D, morton3D
#include "errchk.h"
#include "math_utils.h"

#define MPI_COMPUTE_ENABLED (1)
#define MPI_COMM_ENABLED (1)

static uint3_64
decompose(const uint64_t target)
{
    // This is just so beautifully elegant. Complex and efficient decomposition
    // in just one line of code.
    uint3_64 p = morton3D(target - 1) + (uint3_64){1, 1, 1};

    ERRCHK_ALWAYS(p.x * p.y * p.z == target);
    return p;
}

/* Internal interface to grid (a global variable)  */
typedef struct Grid{
    Device device;
    AcMesh submesh;
    uint3_64 decomposition;
    bool initialized;
    int3 nn;

    std::vector<HaloExchangeTask> halo_exchange_tasks;

    std::vector<HaloExchangeTask> face_exchange_tasks;
    std::vector<HaloExchangeTask> edge_exchange_tasks;
    std::vector<HaloExchangeTask> corner_exchange_tasks;

    std::vector<ComputationTask> computation_tasks;
    ComputationTask* inner_integration_task;

    MPI_Request* recv_reqs;
    MPI_Request* send_reqs;

    MPI_Request* curr_recv_reqs;
    MPI_Request* back_recv_reqs;

    MPI_Request* curr_send_reqs;
    MPI_Request* back_send_reqs;

} Grid;

static Grid grid = {};

static void
gridSwapRequestBuffers(){
    //Assumption SWAP_CHAIN_LENGTH = 2 in these swaps
    std::swap(grid.curr_recv_reqs, grid.back_recv_reqs);
    std::swap(grid.curr_send_reqs, grid.back_send_reqs);
}

AcResult
acGridSynchronizeStream(const Stream stream)
{
    ERRCHK(grid.initialized);
    acDeviceSynchronizeStream(grid.device, stream);
    MPI_Barrier(MPI_COMM_WORLD);
    return AC_SUCCESS;
}

AcResult
acGridRandomize(void)
{
    ERRCHK(grid.initialized);

    AcMesh host;
    acMeshCreate(grid.submesh.info, &host);
    acMeshRandomize(&host);
    acDeviceLoadMesh(grid.device, STREAM_DEFAULT, host);
    acMeshDestroy(&host);

    return AC_SUCCESS;
}

AcResult
acGridInit(const AcMeshInfo info)
{
    ERRCHK(!grid.initialized);

    // Check that MPI is initialized
    int nprocs, pid;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);

    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);

    // Decompose
    AcMeshInfo submesh_info      = info;
    const uint3_64 decomposition = decompose(nprocs);
    const int3 pid3d             = getPid3D(pid, decomposition);

    MPI_Barrier(MPI_COMM_WORLD);
    printf("Processor %s. Process %d of %d: (%d, %d, %d)\n", processor_name, pid, nprocs, pid3d.x,
           pid3d.y, pid3d.z);
    printf("Decomposition: %lu, %lu, %lu\n", decomposition.x, decomposition.y, decomposition.z);
    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);

    ERRCHK_ALWAYS(info.int_params[AC_nx] % decomposition.x == 0);
    ERRCHK_ALWAYS(info.int_params[AC_ny] % decomposition.y == 0);
    ERRCHK_ALWAYS(info.int_params[AC_nz] % decomposition.z == 0);

    const int submesh_nx                       = info.int_params[AC_nx] / decomposition.x;
    const int submesh_ny                       = info.int_params[AC_ny] / decomposition.y;
    const int submesh_nz                       = info.int_params[AC_nz] / decomposition.z;
    submesh_info.int_params[AC_nx]             = submesh_nx;
    submesh_info.int_params[AC_ny]             = submesh_ny;
    submesh_info.int_params[AC_nz]             = submesh_nz;
    submesh_info.int3_params[AC_global_grid_n] = (int3){
        info.int_params[AC_nx],
        info.int_params[AC_ny],
        info.int_params[AC_nz],
    };
    submesh_info.int3_params[AC_multigpu_offset] = pid3d *
                                                   (int3){submesh_nx, submesh_ny, submesh_nz};
    acUpdateBuiltinParams(&submesh_info);

    // GPU alloc
    int devices_per_node = -1;
    cudaGetDeviceCount(&devices_per_node);

    Device device;
    acDeviceCreate(pid % devices_per_node, submesh_info, &device);
 
    // CPU alloc
    AcMesh submesh;
    acMeshCreate(submesh_info, &submesh);

    // Setup the global grid structure
    grid.device = device;
    grid.submesh = submesh;
    grid.decomposition = decomposition;

    // Configure
    const int3 grid_dimensions = (int3){
        device->local_config.int_params[AC_nx],
        device->local_config.int_params[AC_ny],
        device->local_config.int_params[AC_nz],
    };

    grid.nn = grid_dimensions;

    grid.computation_tasks.clear();
    grid.computation_tasks.reserve(NUM_SEGMENTS);

    for (int idx = 0; idx < NUM_SEGMENTS; idx++){
        const int3 segment_id = index_to_segment_id(idx);
        Stream stream = (Stream)(idx + STREAM_DEFAULT);
        grid.computation_tasks.emplace_back(segment_id, grid_dimensions, device, stream);
    }
 
    grid.inner_integration_task = new ComputationTask((int3){0,0,0}, grid_dimensions, device, STREAM_26);

    grid.halo_exchange_tasks.clear();
    grid.halo_exchange_tasks.reserve(NUM_SEGMENTS);
 
    grid.recv_reqs = new MPI_Request[NUM_SEGMENTS*SWAP_CHAIN_LENGTH];
    grid.send_reqs = new MPI_Request[NUM_SEGMENTS*SWAP_CHAIN_LENGTH];

    //This below assumes SWAP_CHAIN_LENGTH == 2
    grid.curr_recv_reqs = grid.recv_reqs;
    grid.back_recv_reqs = &grid.recv_reqs[NUM_SEGMENTS];
    grid.curr_send_reqs = grid.send_reqs;
    grid.back_send_reqs = &grid.send_reqs[NUM_SEGMENTS];

    for (int i = 0; i < NUM_SEGMENTS; i++){
        const int3 seg_id = index_to_segment_id(i);
        grid.halo_exchange_tasks.emplace_back(device, pid, seg_id,
                                              decomposition, grid_dimensions,
                                              grid.recv_reqs, grid.send_reqs);
 
        for (int j = 0; j < NUM_SEGMENTS; j++){
            const int3 compute_seg_id = index_to_segment_id(j);
            if (  ((seg_id.x == 0) || (seg_id.x == compute_seg_id.x))
                &&((seg_id.y == 0) || (seg_id.y == compute_seg_id.y))
                &&((seg_id.z == 0) || (seg_id.z == compute_seg_id.z))){

                grid.halo_exchange_tasks[i].register_dependent(&grid.computation_tasks[j]);
            }
        }
    }
 
    grid.initialized = true;

    acGridSynchronizeStream(STREAM_ALL);
    return AC_SUCCESS;
}

AcResult
acGridQuit(void)
{
    ERRCHK(grid.initialized);
    acGridSynchronizeStream(STREAM_ALL);

    grid.halo_exchange_tasks.clear();

    for (int i = 0; i < NUM_SEGMENTS*SWAP_CHAIN_LENGTH; i++){
        MPI_Request* req = &(grid.recv_reqs[i]);
        if (*req != MPI_REQUEST_NULL){
            MPI_Cancel(req);
            MPI_Request_free(req);
        }
    }

    for (int i = 0; i < NUM_SEGMENTS*SWAP_CHAIN_LENGTH; i++){
        MPI_Request* req = &(grid.send_reqs[i]);
        if (*req != MPI_REQUEST_NULL){
            MPI_Wait(req,MPI_STATUS_IGNORE);
            MPI_Request_free(req);
        }
    }

    delete[] grid.recv_reqs;
    delete[] grid.send_reqs;

    grid.computation_tasks.clear();
    delete grid.inner_integration_task;

    grid.initialized   = false;
    grid.decomposition = (uint3_64){0, 0, 0};
    acMeshDestroy(&grid.submesh);
    acDeviceDestroy(grid.device);

    acGridSynchronizeStream(STREAM_ALL);
    return AC_SUCCESS;
}

AcResult
acGridLoadScalarUniform(const Stream stream, const AcRealParam param, const AcReal value)
{
    ERRCHK(grid.initialized);
    acGridSynchronizeStream(stream);

    const int root_proc = 0;
    AcReal buffer       = value;
    MPI_Bcast(&buffer, 1, AC_MPI_TYPE, root_proc, MPI_COMM_WORLD);

    acDeviceLoadScalarUniform(grid.device, stream, param, buffer);
    return AC_SUCCESS;
}

AcResult
acGridLoadVectorUniform(const Stream stream, const AcReal3Param param, const AcReal3 value)
{
    ERRCHK(grid.initialized);
    acGridSynchronizeStream(stream);

    const int root_proc = 0;
    AcReal3 buffer      = value;
    MPI_Bcast(&buffer, 3, AC_MPI_TYPE, root_proc, MPI_COMM_WORLD);

    acDeviceLoadVectorUniform(grid.device, stream, param, buffer);
    return AC_SUCCESS;
}

// TODO: do with packed data
AcResult
acGridLoadMesh(const Stream stream, const AcMesh host_mesh)
{
    ERRCHK(grid.initialized);
    acGridSynchronizeStream(stream);

#if AC_VERBOSE
    printf("Distributing mesh...\n");
    fflush(stdout);
#endif

    int pid, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    ERRCHK_ALWAYS(&grid.submesh);

    // Submesh nn
    const int3 nn = (int3){
        grid.submesh.info.int_params[AC_nx],
        grid.submesh.info.int_params[AC_ny],
        grid.submesh.info.int_params[AC_nz],
    };

    // Send to self
    if (pid == 0) {
        for (int vtxbuf = 0; vtxbuf < NUM_VTXBUF_HANDLES; ++vtxbuf) {
            // For pencils
            for (int k = NGHOST; k < NGHOST + nn.z; ++k) {
                for (int j = NGHOST; j < NGHOST + nn.y; ++j) {
                    const int i       = NGHOST;
                    const int count   = nn.x;
                    const int src_idx = acVertexBufferIdx(i, j, k, host_mesh.info);
                    const int dst_idx = acVertexBufferIdx(i, j, k, grid.submesh.info);
                    memcpy(&grid.submesh.vertex_buffer[vtxbuf][dst_idx], //
                           &host_mesh.vertex_buffer[vtxbuf][src_idx],  //
                           count * sizeof(host_mesh.vertex_buffer[i][0]));
                }
            }
        }
    }

    for (int vtxbuf = 0; vtxbuf < NUM_VTXBUF_HANDLES; ++vtxbuf) {
        // For pencils
        for (int k = NGHOST; k < NGHOST + nn.z; ++k) {
            for (int j = NGHOST; j < NGHOST + nn.y; ++j) {
                const int i     = NGHOST;
                const int count = nn.x;

                if (pid != 0) {
                    const int dst_idx = acVertexBufferIdx(i, j, k, grid.submesh.info);
                    // Recv
                    MPI_Status status;
                    MPI_Recv(&grid.submesh.vertex_buffer[vtxbuf][dst_idx], count, AC_MPI_TYPE, 0, 0,
                             MPI_COMM_WORLD, &status);
                }
                else {
                    for (int tgt_pid = 1; tgt_pid < nprocs; ++tgt_pid) {
                        const int3 tgt_pid3d = getPid3D(tgt_pid, grid.decomposition);
                        const int src_idx    = acVertexBufferIdx(i + tgt_pid3d.x * nn.x, //
                                                              j + tgt_pid3d.y * nn.y, //
                                                              k + tgt_pid3d.z * nn.z, //
                                                              host_mesh.info);

                        // Send
                        MPI_Send(&host_mesh.vertex_buffer[vtxbuf][src_idx], count, AC_MPI_TYPE, tgt_pid, 0,
                                 MPI_COMM_WORLD);
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
acGridStoreMesh(const Stream stream, AcMesh* host_mesh)
{
    ERRCHK(grid.initialized);
    acGridSynchronizeStream(stream);

    acDeviceStoreMesh(grid.device, stream, &grid.submesh);
    acGridSynchronizeStream(stream);

#if AC_VERBOSE
    printf("Gathering mesh...\n");
    fflush(stdout);
#endif

    int pid, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    if (pid == 0)
        ERRCHK_ALWAYS(host_mesh);

    // Submesh nn
    const int3 nn = (int3){
        grid.submesh.info.int_params[AC_nx],
        grid.submesh.info.int_params[AC_ny],
        grid.submesh.info.int_params[AC_nz],
    };

    // Submesh mm
    const int3 mm = (int3){
        grid.submesh.info.int_params[AC_mx],
        grid.submesh.info.int_params[AC_my],
        grid.submesh.info.int_params[AC_mz],
    };

    // Send to self
    if (pid == 0) {
        for (int vtxbuf = 0; vtxbuf < NUM_VTXBUF_HANDLES; ++vtxbuf) {
            // For pencils
            for (int k = 0; k < mm.z; ++k) {
                for (int j = 0; j < mm.y; ++j) {
                    const int i       = 0;
                    const int count   = mm.x;
                    const int src_idx = acVertexBufferIdx(i, j, k, grid.submesh.info);
                    const int dst_idx = acVertexBufferIdx(i, j, k, host_mesh->info);
                    memcpy(&host_mesh->vertex_buffer[vtxbuf][dst_idx], //
                           &grid.submesh.vertex_buffer[vtxbuf][src_idx],  //
                           count * sizeof(grid.submesh.vertex_buffer[i][0]));
                }
            }
        }
    }

    for (int vtxbuf = 0; vtxbuf < NUM_VTXBUF_HANDLES; ++vtxbuf) {
        // For pencils
        for (int k = 0; k < mm.z; ++k) {
            for (int j = 0; j < mm.y; ++j) {
                const int i     = 0;
                const int count = mm.x;

                if (pid != 0) {
                    // Send
                    const int src_idx = acVertexBufferIdx(i, j, k, grid.submesh.info);
                    MPI_Send(&grid.submesh.vertex_buffer[vtxbuf][src_idx], count, AC_MPI_TYPE, 0, 0,
                             MPI_COMM_WORLD);
                }
                else {
                    for (int tgt_pid = 1; tgt_pid < nprocs; ++tgt_pid) {
                        const int3 tgt_pid3d = getPid3D(tgt_pid, grid.decomposition);
                        const int dst_idx    = acVertexBufferIdx(i + tgt_pid3d.x * nn.x, //
                                                              j + tgt_pid3d.y * nn.y, //
                                                              k + tgt_pid3d.z * nn.z, //
                                                              host_mesh->info);

                        // Recv
                        MPI_Status status;
                        MPI_Recv(&host_mesh->vertex_buffer[vtxbuf][dst_idx], count, AC_MPI_TYPE, tgt_pid, 0,
                                 MPI_COMM_WORLD, &status);
                    }
                }
            }
        }
    }

    return AC_SUCCESS;
}

AcResult
acGridIntegrate(const Stream stream, const AcReal dt)
{
    ERRCHK(grid.initialized);
    acGridSynchronizeStream(stream);
    const Device device = grid.device;
    acDeviceSynchronizeStream(device, stream);
    cudaSetDevice(device->id);
 
    MPI_Barrier(MPI_COMM_WORLD);

    for (auto &halo_task : grid.halo_exchange_tasks){
        if (halo_task.active)
            halo_task.set_trigger_limit(3);
    }

    for (auto &compute_task : grid.computation_tasks){
        compute_task.set_trigger_limit(3);
    }

    for (int isubstep = 0; isubstep < 3; ++isubstep) {
 
#if MPI_COMM_ENABLED
        for (auto &halo_task : grid.halo_exchange_tasks){
            if (halo_task.active){
                halo_task.pack();
            }
        }
 
        for (auto &halo_task : grid.halo_exchange_tasks){
            if (halo_task.active){
                halo_task.send();
            }
        }
#endif //MPI_COMM_ENABLED

#if MPI_COMPUTE_ENABLED
        grid.inner_integration_task->execute(isubstep,dt);
#endif
 
#if MPI_COMM_ENABLED
        //Handle messages as they arrive in a fused loop pipeline
        int idx, prev_idx;

        for (int n = 0; n < NUM_ACTIVE_SEGMENTS+1; n++){
            prev_idx = idx;
            if (n < NUM_ACTIVE_SEGMENTS){
                MPI_Waitany(NUM_SEGMENTS, grid.curr_recv_reqs, &idx, MPI_STATUS_IGNORE);
                ERRCHK(idx >= 0 && idx < NUM_SEGMENTS && grid.halo_exchange_tasks[idx].active);
                grid.halo_exchange_tasks[idx].unpack();
            }
            if(n > 0){
                if ( grid.halo_exchange_tasks[prev_idx].active && prev_idx >= 0 && prev_idx < NUM_SEGMENTS){
                    grid.halo_exchange_tasks[prev_idx].sync();
                    grid.halo_exchange_tasks[prev_idx].receive();
#if MPI_COMPUTE_ENABLED
                    grid.halo_exchange_tasks[prev_idx].notify_dependents(isubstep, dt);
#endif
                }
            }
        }
#else //if no comms, just compute all segments
        for (auto &comp_task : grid.computation_tasks){
            comp_task.execute(isubstep,dt);
        }
#endif //MPI_COMM_ENABLED

        gridSwapRequestBuffers();
        acDeviceSwapBuffers(device);
        acDeviceSynchronizeStream(device, STREAM_ALL); // Wait until inner and outer done
    }
    MPI_Waitall(NUM_SEGMENTS*SWAP_CHAIN_LENGTH, grid.send_reqs, MPI_STATUSES_IGNORE);
    return AC_SUCCESS;
}

AcResult
acGridPeriodicBoundconds(const Stream stream)
{
    ERRCHK(grid.initialized);
    acGridSynchronizeStream(stream);

    //Active halo exchange tasks
    for (auto &halo_task : grid.halo_exchange_tasks){
        if (halo_task.active){
            halo_task.pack();
            halo_task.send();
        }
    }

    MPI_Waitall(NUM_SEGMENTS, grid.curr_recv_reqs, MPI_STATUSES_IGNORE);
    for (auto &halo_task : grid.halo_exchange_tasks){
        if (halo_task.active){
            halo_task.unpack();
            halo_task.sync();
            halo_task.receive();
        }
    }

    //Inactive halo exchange tasks (i.e. possibly corners)
    for (auto &halo_task : grid.halo_exchange_tasks){
        if( !halo_task.active){
            halo_task.pack();
            halo_task.exchange();
        }
    }
    for (auto &halo_task : grid.halo_exchange_tasks){
        if( !halo_task.active){
            halo_task.wait_recv();
            halo_task.unpack();
            halo_task.sync();
        }
    }

    MPI_Waitall(NUM_SEGMENTS*SWAP_CHAIN_LENGTH, grid.send_reqs, MPI_STATUSES_IGNORE);
    gridSwapRequestBuffers();
    return AC_SUCCESS;
}

static AcResult
reduceScal(const AcReal local_result, const ReductionType rtype, AcReal* result)
{

    MPI_Op op;
    if (rtype == RTYPE_MAX) {
        op = MPI_MAX;
    }
    else if (rtype == RTYPE_MIN) {
        op = MPI_MIN;
    }
    else if (rtype == RTYPE_RMS || rtype == RTYPE_RMS_EXP || rtype == RTYPE_SUM) {
        op = MPI_SUM;
    }
    else {
        ERROR("Unrecognised rtype");
    }

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    AcReal mpi_res;
    MPI_Reduce(&local_result, &mpi_res, 1, AC_MPI_TYPE, op, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        if (rtype == RTYPE_RMS || rtype == RTYPE_RMS_EXP) {
            const AcReal inv_n = AcReal(1.) /
                                 (grid.nn.x * grid.decomposition.x * grid.nn.y *
                                  grid.decomposition.y * grid.nn.z * grid.decomposition.z);
            mpi_res = sqrt(inv_n * mpi_res);
        }
        *result = mpi_res;
    }
    return AC_SUCCESS;
}

AcResult
acGridReduceScal(const Stream stream, const ReductionType rtype,
                 const VertexBufferHandle vtxbuf_handle, AcReal* result)
{
    ERRCHK(grid.initialized);
    const Device device = grid.device;
    acGridSynchronizeStream(STREAM_ALL);

    AcReal local_result;
    acDeviceReduceScal(device, stream, rtype, vtxbuf_handle, &local_result);

    return reduceScal(local_result, rtype, result);
}

AcResult
acGridReduceVec(const Stream stream, const ReductionType rtype, const VertexBufferHandle vtxbuf0,
                const VertexBufferHandle vtxbuf1, const VertexBufferHandle vtxbuf2, AcReal* result)
{
    ERRCHK(grid.initialized);
    const Device device = grid.device;
    acGridSynchronizeStream(STREAM_ALL);

    AcReal local_result;
    acDeviceReduceVec(device, stream, rtype, vtxbuf0, vtxbuf1, vtxbuf2, &local_result);

    return reduceScal(local_result, rtype, result);
}
//MV: for MPI we will need acGridReduceVecScal() to get Alfven speeds etc. TODO 
#endif // AC_MPI_ENABLED
