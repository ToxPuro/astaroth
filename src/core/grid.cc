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
#include "task.h"

#include <algorithm>
#include <cstring> //memcpy
#include <mpi.h>
#include <queue>
#include <vector>

#include "decomposition.h" //getPid3D, morton3D
#include "errchk.h"
#include "math_utils.h"
#include "timer_hires.h"

/* Internal interface to grid (a global variable)  */
typedef struct Grid {
    Device device;
    AcMesh submesh;
    uint3_64 decomposition;
    bool initialized;
    int3 nn;
    std::shared_ptr<AcTaskGraph> default_tasks;
    size_t mpi_tag_space_count;
} Grid;

static Grid grid = {};

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
    acHostMeshCreate(grid.submesh.info, &host);
    acHostMeshRandomize(&host);
    acDeviceLoadMesh(grid.device, STREAM_DEFAULT, host);
    acHostMeshDestroy(&host);

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
    AcMeshInfo submesh_info = info;
    const uint3_64 decomp   = decompose(nprocs);
    const int3 pid3d        = getPid3D(pid, decomp);

    MPI_Barrier(MPI_COMM_WORLD);
    printf("Processor %s. Process %d of %d: (%d, %d, %d)\n", processor_name, pid, nprocs, pid3d.x,
           pid3d.y, pid3d.z);
    printf("Decomposition: %lu, %lu, %lu\n", decomp.x, decomp.y, decomp.z);
    printf("Mesh size: %d, %d, %d\n", info.int_params[AC_nx], info.int_params[AC_ny],
           info.int_params[AC_nz]);
    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);

    ERRCHK_ALWAYS(info.int_params[AC_nx] % decomp.x == 0);
    ERRCHK_ALWAYS(info.int_params[AC_ny] % decomp.y == 0);
    ERRCHK_ALWAYS(info.int_params[AC_nz] % decomp.z == 0);

    // Check that mixed precision is correctly configured, AcRealPacked == AC_MPI_TYPE
    // CAN BE REMOVED IF MIXED PRECISION IS SUPPORTED AS A PREPROCESSOR FLAG
    int mpi_type_size;
    MPI_Type_size(AC_MPI_TYPE, &mpi_type_size);
    ERRCHK_ALWAYS(sizeof(AcRealPacked) == mpi_type_size);

    const int submesh_nx                       = info.int_params[AC_nx] / decomp.x;
    const int submesh_ny                       = info.int_params[AC_ny] / decomp.y;
    const int submesh_nz                       = info.int_params[AC_nz] / decomp.z;
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
    acHostUpdateBuiltinParams(&submesh_info);

    // GPU alloc
    int devices_per_node = -1;
    cudaGetDeviceCount(&devices_per_node);

    Device device;
    acDeviceCreate(pid % devices_per_node, submesh_info, &device);

    // CPU alloc
    AcMesh submesh;
    acHostMeshCreate(submesh_info, &submesh);

    // Setup the global grid structure
    grid.device        = device;
    grid.submesh       = submesh;
    grid.decomposition = decomp;

    // Configure
    grid.nn = (int3){
        device->local_config.int_params[AC_nx],
        device->local_config.int_params[AC_ny],
        device->local_config.int_params[AC_nz],
    };

    grid.mpi_tag_space_count = 0;

    Field all_fields[NUM_VTXBUF_HANDLES];
    for (int i = 0; i < NUM_VTXBUF_HANDLES; i++) {
        all_fields[i] = (Field)i;
    }

    AcTaskDefinition default_ops[] = {acHaloExchange(all_fields),
                                      acBoundaryCondition(BOUNDARY_XYZ, BOUNDCOND_PERIODIC,
                                                          all_fields),
#ifdef AC_INTEGRATION_ENABLED
                                      acCompute(KERNEL_solve, all_fields)
#endif // AC_INTEGRATION_ENABLED
    };

    grid.initialized   = true;
    grid.default_tasks = std::shared_ptr<AcTaskGraph>(acGridBuildTaskGraph(default_ops));

    acGridSynchronizeStream(STREAM_ALL);
    return AC_SUCCESS;
}

AcResult
acGridQuit(void)
{
    ERRCHK(grid.initialized);
    acGridSynchronizeStream(STREAM_ALL);

    grid.default_tasks = nullptr;

    grid.initialized   = false;
    grid.decomposition = (uint3_64){0, 0, 0};
    acHostMeshDestroy(&grid.submesh);
    acDeviceDestroy(grid.device);

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
                           &host_mesh.vertex_buffer[vtxbuf][src_idx],    //
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
                        MPI_Send(&host_mesh.vertex_buffer[vtxbuf][src_idx], count, AC_MPI_TYPE,
                                 tgt_pid, 0, MPI_COMM_WORLD);
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
                    memcpy(&host_mesh->vertex_buffer[vtxbuf][dst_idx],   //
                           &grid.submesh.vertex_buffer[vtxbuf][src_idx], //
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
                        MPI_Recv(&host_mesh->vertex_buffer[vtxbuf][dst_idx], count, AC_MPI_TYPE,
                                 tgt_pid, 0, MPI_COMM_WORLD, &status);
                    }
                }
            }
        }
    }

    return AC_SUCCESS;
}

AcTaskGraph*
acGridGetDefaultTaskGraph()
{
    ERRCHK(grid.initialized);
    return grid.default_tasks.get();
}

static void
check_ops(const AcTaskDefinition ops[], const size_t n_ops)
{
    if (n_ops == 0) {
        ERROR("\nIncorrect task graph {}:\n - Task graph is empty.\n")
    }

    bool found_halo_exchange        = false;
    unsigned int boundaries_defined = 0x00;
    // bool found_compute              = false;

    bool boundary_condition_before_halo_exchange = false;
    bool compute_before_halo_exchange            = false;
    bool compute_before_boundary_condition       = false;

    bool error   = false;
    bool warning = false;

    std::string task_graph_repr = "{";

    for (size_t i = 0; i < n_ops; i++) {
        AcTaskDefinition op = ops[i];
        switch (op.task_type) {
        case TASKTYPE_HALOEXCHANGE:
            found_halo_exchange = true;
            task_graph_repr += "HaloExchange,";
            break;
        case TASKTYPE_BOUNDCOND:
        case TASKTYPE_SPECIAL_MHD_BOUNDCOND:
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
            if (boundaries_defined != BOUNDARY_XYZ) {
                compute_before_boundary_condition = true;
                warning                           = true;
            }
            // found_compute = true;
            task_graph_repr += "Compute,";
            break;
        }
    }

    task_graph_repr += "}";

    std::string msg = "";

    if (!found_halo_exchange) {
        msg += " - No halo exchange defined in task graph.\n";
        error = true;
    }

    if (boundaries_defined != BOUNDARY_XYZ) {
        error = true;
    }
    if ((boundaries_defined & BOUNDARY_X_TOP) != BOUNDARY_X_TOP) {
        msg += " - Boundary conditions not defined for top X boundary.\n";
    }
    if ((boundaries_defined & BOUNDARY_X_BOT) != BOUNDARY_X_BOT) {
        msg += " - Boundary conditions not defined for bottom X boundary.\n";
    }
    if ((boundaries_defined & BOUNDARY_Y_TOP) != BOUNDARY_Y_TOP) {
        msg += " - Boundary conditions not defined for top Y boundary.\n";
    }
    if ((boundaries_defined & BOUNDARY_Y_BOT) != BOUNDARY_Y_BOT) {
        msg += " - Boundary conditions not defined for bottom Y boundary.\n";
    }
    if ((boundaries_defined & BOUNDARY_Z_TOP) != BOUNDARY_Z_TOP) {
        msg += " - Boundary conditions not defined for top Z boundary.\n";
    }
    if ((boundaries_defined & BOUNDARY_Z_BOT) != BOUNDARY_Z_BOT) {
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

    if (error) {
        ERROR(("\nIncorrect task graph " + task_graph_repr + ":\n" + msg).c_str())
    }
    if (warning) {
        WARNING(("\nUnusual task graph " + task_graph_repr + ":\n" + msg).c_str())
    }
}

AcTaskGraph*
acGridBuildTaskGraph(const AcTaskDefinition ops[], const size_t n_ops)
{
    // ERRCHK(grid.initialized);
    check_ops(ops, n_ops);

    AcTaskGraph* graph = new AcTaskGraph();
    graph->halo_tasks.reserve(n_ops * Region::n_halo_regions);
    graph->all_tasks.reserve(n_ops * max(Region::n_halo_regions, Region::n_comp_regions));

    // Create tasks for each operation & store indices to ranges of tasks belonging to operations
    std::vector<size_t> op_indices;
    op_indices.reserve(n_ops);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int3 nn         = grid.nn;
    uint3_64 decomp = grid.decomposition;
    int3 pid3d      = getPid3D(rank, grid.decomposition);
    Device device   = grid.device;

    auto region_at_boundary = [&decomp, &pid3d](int tag, AcBoundary boundary) -> bool {
        int3 neighbor = pid3d + Region::tag_to_id(tag);

        return (((boundary & BOUNDARY_X_BOT) != 0) && neighbor.x == -1) ||
               (((boundary & BOUNDARY_X_TOP) != 0) && neighbor.x == (int)decomp.x) ||
               (((boundary & BOUNDARY_Y_BOT) != 0) && neighbor.y == -1) ||
               (((boundary & BOUNDARY_Y_TOP) != 0) && neighbor.y == (int)decomp.y) ||
               (((boundary & BOUNDARY_Z_BOT) != 0) && neighbor.z == -1) ||
               (((boundary & BOUNDARY_Z_TOP) != 0) && neighbor.z == (int)decomp.z);
    };
    auto boundary_normal = [&decomp, &pid3d](int tag) -> int3 {
        int3 neighbor = pid3d + Region::tag_to_id(tag);
        if (neighbor.z == -1) {
            return int3{0, 0, -1};
        }
        else if (neighbor.z == (int)decomp.z) {
            return int3{0, 0, 1};
        }
        else if (neighbor.y == -1) {
            return int3{0, -1, 0};
        }
        else if (neighbor.y == (int)decomp.y) {
            return int3{0, 1, 0};
        }
        else if (neighbor.x == -1) {
            return int3{-1, 0, 0};
        }
        else if (neighbor.x == (int)decomp.x) {
            return int3{1, 0, 0};
        }
        else {
            // Something went wrong, this tag does not identify a boundary region.
            return int3{0, 0, 0};
        }
        // return int3{(neighbor.x == -1) ? -1 : (neighbor.x == (int)decomp.x ? 1 : 0),
        //             (neighbor.y == -1) ? -1 : (neighbor.y == (int)decomp.y ? 1 : 0),
        //             (neighbor.z == -1) ? -1 : (neighbor.z == (int)decomp.z ? 1 : 0)};
    };

    // The tasks start at different offsets from the beginning of the iteration
    // this array of bools keep track of that state
    std::array<bool, NUM_VTXBUF_HANDLES> swap_offset{};

    for (size_t i = 0; i < n_ops; i++) {
        auto op = ops[i];
        op_indices.push_back(graph->all_tasks.size());

        switch (op.task_type) {

        case TASKTYPE_COMPUTE: {
            // Kernel kernel = kernel_lookup[(int)op.kernel];
            for (int tag = Region::min_comp_tag; tag < Region::max_comp_tag; tag++) {
                auto task = std::make_shared<ComputeTask>(op, i, tag, nn, device, swap_offset);
                graph->all_tasks.push_back(task);
            }
            for (size_t buf = 0; buf < op.num_fields_out; buf++) {
                swap_offset[op.fields_out[buf]] = !swap_offset[op.fields_out[buf]];
            }
            break;
        }

        case TASKTYPE_HALOEXCHANGE: {
            int tag0 = grid.mpi_tag_space_count * Region::max_halo_tag;
            for (int tag = Region::min_halo_tag; tag < Region::max_halo_tag; tag++) {
                if (!region_at_boundary(tag, BOUNDARY_XYZ)) {
                    auto task = std::make_shared<HaloExchangeTask>(op, i, tag0, tag, nn, decomp,
                                                                   device, swap_offset);
                    graph->halo_tasks.push_back(task);
                    graph->all_tasks.push_back(task);
                }
            }
            grid.mpi_tag_space_count++;
            break;
        }

        case TASKTYPE_BOUNDCOND: {
            AcBoundcond bc = op.bound_cond;
            int tag0       = grid.mpi_tag_space_count * Region::max_halo_tag;
            for (int tag = Region::min_halo_tag; tag < Region::max_halo_tag; tag++) {
                if (region_at_boundary(tag, op.boundary)) {
                    if (bc == BOUNDCOND_PERIODIC) {
                        auto task = std::make_shared<HaloExchangeTask>(op, i, tag0, tag, nn, decomp,
                                                                       device, swap_offset);
                        graph->halo_tasks.push_back(task);
                        graph->all_tasks.push_back(task);
                    }
                    else {
                        auto task = std::make_shared<BoundaryConditionTask>(op,
                                                                            boundary_normal(tag), i,
                                                                            tag, nn, device,
                                                                            swap_offset);
                        graph->all_tasks.push_back(task);
                    }
                }
            }
            grid.mpi_tag_space_count++;
            break;
        }

        case TASKTYPE_SPECIAL_MHD_BOUNDCOND: {
#ifdef AC_INTEGRATION_ENABLED
            for (int tag = Region::min_halo_tag; tag < Region::max_halo_tag; tag++) {
                if (region_at_boundary(tag, op.boundary)) {
                    auto task = std::make_shared<SpecialMHDBoundaryConditionTask>(op,
                                                                                  boundary_normal(
                                                                                      tag),
                                                                                  i, tag, nn,
                                                                                  device,
                                                                                  swap_offset);
                    graph->all_tasks.push_back(task);
                }
            }
#endif
            break;
        }
        }
    }
    op_indices.push_back(graph->all_tasks.size());
    graph->vtxbuf_swaps = swap_offset;

    graph->halo_tasks.shrink_to_fit();
    graph->all_tasks.shrink_to_fit();

    // In order to reduce redundant dependencies, we keep track of which tasks are connected
    size_t n_tasks               = graph->all_tasks.size();
    size_t adjacancy_matrix_size = n_tasks * n_tasks;
    bool adjacent[adjacancy_matrix_size]{};

    //...and check if there is already a forward path that connects two tasks
    auto forward_search = [&adjacent, &op_indices, n_tasks,
                           n_ops](size_t preq, size_t dept, size_t preq_op, size_t path_len) {
        bool visited[n_tasks]{};
        size_t start_op = (preq_op + 1) % n_ops;

        struct walk_node {
            size_t node;
            size_t op_offset;
        };
        std::queue<walk_node> walk;
        walk.push({preq, 0});

        while (!walk.empty()) {
            auto curr = walk.front();
            walk.pop();
            if (adjacent[curr.node * n_tasks + dept]) {
                return true;
            }
            for (size_t op_offset = curr.op_offset; op_offset < path_len; op_offset++) {
                size_t op = (start_op + op_offset) % n_ops;
                for (size_t neighbor = op_indices[op]; neighbor != op_indices[op + 1]; neighbor++) {
                    if (!visited[neighbor] && adjacent[curr.node * n_tasks + neighbor]) {
                        walk.push({neighbor, op_offset});
                        visited[neighbor] = true;
                    }
                }
            }
        }
        return false;
    };

    // We walk through all tasks, and compare tasks from pairs of operations at
    // a time. Pairs are considered in order of increasing distance between the
    // operations in the pair. The final set of pairs that are considered are
    // self-equal pairs, since the operations form a cycle when iterated over
    for (size_t op_offset = 0; op_offset < n_ops; op_offset++) {
        for (size_t dept_op = 0; dept_op < n_ops; dept_op++) {
            size_t preq_op = (n_ops + dept_op - op_offset - 1) % n_ops;
            for (auto i = op_indices[preq_op]; i != op_indices[preq_op + 1]; i++) {
                auto preq_task = graph->all_tasks[i];
                if (preq_task->active) {
                    for (auto j = op_indices[dept_op]; j != op_indices[dept_op + 1]; j++) {
                        auto dept_task = graph->all_tasks[j];
                        // Task A depends on task B if the output region of A overlaps with the
                        // input region of B.
                        if (dept_task->active &&
                            preq_task->output_region.overlaps(&(dept_task->input_region))) {
                            // iteration offset of 0 -> dependency in the same iteration
                            // iteration offset of 1 -> dependency from preq_task in iteration k to
                            // dept_task in iteration k+1
                            if (!forward_search(i, j, preq_op, op_offset)) {
                                preq_task->registerDependent(dept_task, preq_op < dept_op ? 0 : 1);
                                adjacent[i * n_tasks + j] = true;
                            }
                        }
                    }
                }
            }
        }
    }

    // Finally sort according to a priority. Larger volumes first and comm before comp
    auto sort_lambda = [](std::shared_ptr<Task> t1, std::shared_ptr<Task> t2) {
        auto comp1 = t1->task_type == TASKTYPE_COMPUTE;
        auto comp2 = t2->task_type == TASKTYPE_COMPUTE;

        auto vol1 = t1->output_region.volume;
        auto vol2 = t2->output_region.volume;
        auto dim1 = t1->output_region.dims;
        auto dim2 = t2->output_region.dims;

        return vol1 > vol2 ||
               (vol1 == vol2 && ((!comp1 && comp2) || dim1.x < dim2.x || dim1.z > dim2.z));
    };

    std::sort(graph->halo_tasks.begin(), graph->halo_tasks.end(), sort_lambda);
    std::sort(graph->all_tasks.begin(), graph->all_tasks.end(), sort_lambda);
    return graph;
}

AcResult
acGridDestroyTaskGraph(AcTaskGraph* graph)
{
    graph->all_tasks.clear();
    graph->halo_tasks.clear();
    delete graph;
    return AC_SUCCESS;
}

AcResult
acGridExecuteTaskGraph(AcTaskGraph* graph, size_t n_iterations)
{
    ERRCHK(grid.initialized);
    // acGridSynchronizeStream(stream);
    // acDeviceSynchronizeStream(grid.device, stream);
    cudaSetDevice(grid.device->id);

    if (graph->trace_file.enabled) {
        timer_reset(&(graph->trace_file.timer));
    }

    for (auto& task : graph->all_tasks) {
        if (task->active) {
            task->syncVBA();
            task->setIterationParams(0, n_iterations);
        }
    }

    bool ready;
    do {
        ready = true;
        for (auto& task : graph->all_tasks) {
            if (task->active) {
                task->update(graph->vtxbuf_swaps, &(graph->trace_file));
                ready &= task->isFinished();
            }
        }
    } while (!ready);

    if (n_iterations % 2 != 0) {
        for (size_t i = 0; i < NUM_VTXBUF_HANDLES; i++) {
            if (graph->vtxbuf_swaps[i]) {
                acDeviceSwapBuffer(grid.device, (VertexBufferHandle)i);
            }
        }
    }
    return AC_SUCCESS;
}

#ifdef AC_INTEGRATION_ENABLED
AcResult
acGridIntegrate(const Stream stream, const AcReal dt)
{
    ERRCHK(grid.initialized);
    acGridLoadScalarUniform(stream, AC_dt, dt);
    acDeviceSynchronizeStream(grid.device, stream);
    return acGridExecuteTaskGraph(grid.default_tasks.get(), 3);
}
#endif // AC_INTEGRATION_ENABLED

AcResult
acGridPeriodicBoundconds(const Stream stream)
{
    ERRCHK(grid.initialized);
    acGridSynchronizeStream(stream);

    // Active halo exchange tasks use send() instead of exchange() because there is an active eager
    // receive that needs to be used. A new eager receive is posted after the exchange.
    for (auto& halo_task : grid.default_tasks->halo_tasks) {
        halo_task->syncVBA();
        halo_task->pack();
        if (halo_task->active) {
            halo_task->send();
        }
        else {
            halo_task->exchange();
        }
    }

    for (auto& halo_task : grid.default_tasks->halo_tasks) {
        halo_task->wait_recv();
        halo_task->unpack();
        halo_task->sync();
        if (halo_task->active) {
            halo_task->receive();
        }
    }

    for (auto& halo_task : grid.default_tasks->halo_tasks) {
        halo_task->wait_send();
    }
    return AC_SUCCESS;
}

static AcResult
distributedScalarReduction(const AcReal local_result, const ReductionType rtype, AcReal* result)
{
    MPI_Op op;
    if (rtype == RTYPE_MAX || rtype == RTYPE_ALFVEN_MAX) {
        op = MPI_MAX;
    }
    else if (rtype == RTYPE_MIN || rtype == RTYPE_ALFVEN_MIN) {
        op = MPI_MIN;
    }
    else if (rtype == RTYPE_RMS || rtype == RTYPE_RMS_EXP || rtype == RTYPE_SUM ||
             rtype == RTYPE_ALFVEN_RMS) {
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
        if (rtype == RTYPE_RMS || rtype == RTYPE_RMS_EXP || rtype == RTYPE_ALFVEN_RMS) {
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

    return distributedScalarReduction(local_result, rtype, result);
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

    return distributedScalarReduction(local_result, rtype, result);
}

AcResult
acGridReduceVecScal(const Stream stream, const ReductionType rtype,
                    const VertexBufferHandle vtxbuf0, const VertexBufferHandle vtxbuf1,
                    const VertexBufferHandle vtxbuf2, const VertexBufferHandle vtxbuf3,
                    AcReal* result)
{
    ERRCHK(grid.initialized);
    const Device device = grid.device;
    acGridSynchronizeStream(STREAM_ALL);

    AcReal local_result;
    acDeviceReduceVecScal(device, stream, rtype, vtxbuf0, vtxbuf1, vtxbuf2, vtxbuf3, &local_result);

    return distributedScalarReduction(local_result, rtype, result);
}

/*   MV: Commented out for a while, but save for the future when standalone_MPI
         works with periodic boundary conditions.
AcResult
acGridGeneralBoundconds(const Device device, const Stream stream)
{
    // Non-periodic Boundary conditions
    // Check the position in MPI frame
    int nprocs, pid;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
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
        MPI_Barrier(MPI_COMM_WORLD);

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
