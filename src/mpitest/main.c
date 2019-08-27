/*
    Copyright (C) 2014-2019, Johannes Pekkilae, Miikka Vaeisalae.

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
/**
    Running: mpirun -np <num processes> <executable>
*/
#undef NDEBUG // Assert always
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "astaroth.h"
#include "autotest.h"
#include "renderer.h"

#include <mpi.h>

// From Astaroth Standalone
#include "config_loader.h"
#include "model/host_memory.h"
#include "model/model_boundconds.h"
#include "model/model_rk3.h"

static void
distribute_mesh(const AcMesh* src, AcMesh* dst)
{
    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Datatype datatype = MPI_FLOAT;
    if (sizeof(AcReal) == 8)
        datatype = MPI_DOUBLE;

    int pid, num_processes;
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);

    const size_t count = acVertexBufferSize(dst->info);
    for (int i = 0; i < NUM_VTXBUF_HANDLES; ++i) {

        if (pid == 0) {
            // Communicate to self
            assert(src);
            assert(dst);
            memcpy(&dst->vertex_buffer[i][0], //
                   &src->vertex_buffer[i][0], //
                   count * sizeof(src->vertex_buffer[i][0]));

            // Communicate to others
            for (int j = 1; j < num_processes; ++j) {
                assert(src);

                const size_t src_idx = acVertexBufferIdx(
                    0, 0, j * src->info.int_params[AC_nz] / num_processes, src->info);

                MPI_Send(&src->vertex_buffer[i][src_idx], count, datatype, j, 0, MPI_COMM_WORLD);
            }
        }
        else {
            assert(dst);

            // Recv
            const size_t dst_idx = 0;
            MPI_Status status;
            MPI_Recv(&dst->vertex_buffer[i][dst_idx], count, datatype, 0, 0, MPI_COMM_WORLD,
                     &status);
        }
    }
}

static void
gather_mesh(const AcMesh* src, AcMesh* dst)
{
    MPI_Barrier(MPI_COMM_WORLD);
    printf("Gathering mesh...\n");
    MPI_Datatype datatype = MPI_FLOAT;
    if (sizeof(AcReal) == 8)
        datatype = MPI_DOUBLE;

    int pid, num_processes;
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);

    size_t count = acVertexBufferSize(src->info);

    for (int i = 0; i < NUM_VTXBUF_HANDLES; ++i) {
        // Communicate to self
        if (pid == 0) {
            assert(src);
            assert(dst);
            memcpy(&dst->vertex_buffer[i][0], //
                   &src->vertex_buffer[i][0], //
                   count * sizeof(src->vertex_buffer[i][0]));

            for (int j = 1; j < num_processes; ++j) {
                // Recv
                const size_t dst_idx = acVertexBufferIdx(
                    0, 0, j * dst->info.int_params[AC_nz] / num_processes, dst->info);

                assert(dst_idx + count <= acVertexBufferSize(dst->info));
                MPI_Status status;
                MPI_Recv(&dst->vertex_buffer[i][dst_idx], count, datatype, j, 0, MPI_COMM_WORLD,
                         &status);
            }
        }
        else {
            // Send
            const size_t src_idx = 0;

            assert(src_idx + count <= acVertexBufferSize(src->info));
            MPI_Send(&src->vertex_buffer[i][src_idx], count, datatype, 0, 0, MPI_COMM_WORLD);
        }
    }
}

static void
communicate_halos(AcMesh* submesh)
{
    MPI_Barrier(MPI_COMM_WORLD);
    printf("Communicating bounds...\n");
    MPI_Datatype datatype = MPI_FLOAT;
    if (sizeof(AcReal) == 8)
        datatype = MPI_DOUBLE;

    int pid, num_processes;
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);

    const size_t count = submesh->info.int_params[AC_mx] * submesh->info.int_params[AC_my] * NGHOST;

    for (int i = 0; i < NUM_VTXBUF_HANDLES; ++i) {
        { // Front
            // ...|ooooxxx|... -> xxx|ooooooo|...
            const size_t src_idx = acVertexBufferIdx(0, 0, submesh->info.int_params[AC_nz],
                                                     submesh->info);
            const size_t dst_idx = acVertexBufferIdx(0, 0, 0, submesh->info);
            const int send_pid   = (pid + 1) % num_processes;
            const int recv_pid   = (pid + num_processes - 1) % num_processes;

            MPI_Request request;
            MPI_Isend(&submesh->vertex_buffer[i][src_idx], count, datatype, send_pid, i,
                      MPI_COMM_WORLD, &request);
            fflush(stdout);

            MPI_Status status;
            MPI_Recv(&submesh->vertex_buffer[i][dst_idx], count, datatype, recv_pid, i,
                     MPI_COMM_WORLD, &status);

            MPI_Wait(&request, &status);
        }
        { // Back
            // ...|ooooooo|xxx <- ...|xxxoooo|...
            const size_t src_idx = acVertexBufferIdx(0, 0, NGHOST, submesh->info);
            const size_t dst_idx = acVertexBufferIdx(0, 0, NGHOST + submesh->info.int_params[AC_nz],
                                                     submesh->info);
            const int send_pid   = (pid + num_processes - 1) % num_processes;
            const int recv_pid   = (pid + 1) % num_processes;

            MPI_Request request;
            MPI_Isend(&submesh->vertex_buffer[i][src_idx], count, datatype, send_pid,
                      NUM_VTXBUF_HANDLES + i, MPI_COMM_WORLD, &request);

            MPI_Status status;
            MPI_Recv(&submesh->vertex_buffer[i][dst_idx], count, datatype, recv_pid,
                     NUM_VTXBUF_HANDLES + i, MPI_COMM_WORLD, &status);

            MPI_Wait(&request, &status);
        }
    }
}

int
main(void)
{
    MPI_Init(NULL, NULL);

//// Borrowing start (from OpenMPI examples)
#if defined(MPIX_CUDA_AWARE_SUPPORT) && MPIX_CUDA_AWARE_SUPPORT
    printf("This MPI library has CUDA-aware support.\n", MPIX_CUDA_AWARE_SUPPORT);
#elif defined(MPIX_CUDA_AWARE_SUPPORT) && !MPIX_CUDA_AWARE_SUPPORT
    printf("This MPI library does not have CUDA-aware support.\n");
#else
    printf("This MPI library cannot determine if there is CUDA-aware support.\n");
#endif /* MPIX_CUDA_AWARE_SUPPORT */

    printf("Run time check:\n");
#if defined(MPIX_CUDA_AWARE_SUPPORT)
    if (1 == MPIX_Query_cuda_support()) {
        printf("This MPI library has CUDA-aware support.\n");
    }
    else {
        printf("This MPI library does not have CUDA-aware support.\n");
    }
#else  /* !defined(MPIX_CUDA_AWARE_SUPPORT) */
    printf("This MPI library cannot determine if there is CUDA-aware support.\n");
#endif /* MPIX_CUDA_AWARE_SUPPORT */
       //////// Borrowing end

    int num_processes, pid;
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);

    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);
    printf("Processor %s. Process %d of %d.\n", processor_name, pid, num_processes);

    AcMeshInfo mesh_info;
    load_config(&mesh_info);
    update_config(&mesh_info);
    assert(mesh_info.int_params[AC_nz] % num_processes == 0); // MUST BE DIVISIBLE!

    AcMesh* main_mesh     = NULL;
    ModelMesh* model_mesh = NULL;
    if (pid == 0) {
        main_mesh = acmesh_create(mesh_info);
        acmesh_init_to(INIT_TYPE_RANDOM, main_mesh);
        model_mesh = modelmesh_create(mesh_info);
        acmesh_to_modelmesh(*main_mesh, model_mesh);
        boundconds(mesh_info, model_mesh);
        modelmesh_to_acmesh(*model_mesh, main_mesh);
    }

    AcMeshInfo submesh_info = mesh_info;
    submesh_info.int_params[AC_nz] /= num_processes;
    update_config(&submesh_info);

    AcMesh* submesh = acmesh_create(submesh_info);

    /*
    ///////////////////// Working basic CPU
    distribute_mesh(main_mesh, submesh);
    communicate_halos(submesh);
    gather_mesh(submesh, main_mesh);
    /////////////////////////
    */

    ///// DISTRIBUTE
    distribute_mesh(main_mesh, submesh);
    MPI_Barrier(MPI_COMM_WORLD);

    //////////////////// GPU ONLY STUFF
    // Node node;
    // acNodeCreate(0, submesh_info, &node);
    Device device;
    // NOTE: assumes that every node has the same number of devices
    const int device_id = pid % acGetNumDevicesPerNode();
    acDeviceCreate(device_id, submesh_info, &device);
    const AcReal dt = FLT_EPSILON; // TODO multi-node reduction calls and proper timestepping

    for (int isubstep = 0; isubstep < 3; ++isubstep) {
        acDeviceSynchronizeStream(device, STREAM_ALL);
        acDeviceLoadMesh(device, STREAM_DEFAULT, *submesh);
        {
            const int3 start = (int3){NGHOST, NGHOST, NGHOST};
            const int3 end   = (int3){submesh_info.int_params[AC_nx_max],
                                    submesh_info.int_params[AC_ny_max],
                                    submesh_info.int_params[AC_nz_max]};
            acDeviceIntegrateSubstep(device, STREAM_DEFAULT, isubstep, start, end, dt);
        }
        acDeviceSwapBuffers(device);
        {
            const int3 start = (int3){0, 0, NGHOST};
            const int3 end = (int3){submesh_info.int_params[AC_mx], submesh_info.int_params[AC_my],
                                    submesh_info.int_params[AC_nz_max]};
            acDevicePeriodicBoundconds(device, STREAM_DEFAULT, start, end);
        }
        acDeviceStoreMesh(device, STREAM_DEFAULT, submesh);
        MPI_Barrier(MPI_COMM_WORLD);
        // communicate_halos(submesh);
        acDeviceCommunicateHalosMPI(device);
        MPI_Barrier(MPI_COMM_WORLD);
        /*
        acNodeSynchronizeStream(node, STREAM_ALL);
        acNodeLoadMesh(node, STREAM_DEFAULT, *submesh);
        const int3 start = (int3){NGHOST, NGHOST, NGHOST};
        const int3 end   = (int3){submesh_info.int_params[AC_nx_max],
                                submesh_info.int_params[AC_ny_max],
                                submesh_info.int_params[AC_nz_max]};

        acNodeIntegrateSubstep(node, STREAM_DEFAULT, isubstep, start, end, dt);
        acNodeSwapBuffers(node);
        acNodePeriodicBoundconds(node, STREAM_DEFAULT);
        acNodeStoreMesh(node, STREAM_DEFAULT, submesh);
        acNodeSynchronizeStream(node, STREAM_DEFAULT);
        MPI_Barrier(MPI_COMM_WORLD);
        communicate_halos(submesh);
        // acDeviceCommunicateHalosMPI();
        MPI_Barrier(MPI_COMM_WORLD);
        */
    }
    //////////////////// GPU ONLY STUFF
    //// GATHER
    gather_mesh(submesh, main_mesh);
    /////////////

    acmesh_destroy(submesh);

    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);
    acDeviceDestroy(device);
    // acNodeDestroy(node);

    //////////// RENDER
    /*
    if (pid == 0) {
        acInit(main_mesh->info);
        acLoad(*main_mesh);
        assert(main_mesh);
        renderer_init(main_mesh->info.int_params[AC_mx], main_mesh->info.int_params[AC_my]);
        while (1) {
            check_input(1.f / 60.f);
            renderer_draw(*main_mesh);
        }
        renderer_quit();
        acQuit();
    }
    */
    /////////////

    if (pid == 0) {
        assert(main_mesh);
        assert(model_mesh);

        // Integrate step
        model_rk3(dt, model_mesh);
        boundconds(mesh_info, model_mesh);
        ///

        bool is_acceptable = verify_meshes(*model_mesh, *main_mesh);
        printf("%s\n", is_acceptable ? "OK!" : "FAIL!");
        modelmesh_destroy(model_mesh);
        acmesh_destroy(main_mesh);
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}
