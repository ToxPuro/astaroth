/*
    Copyright (C) 2014-2022, Johannes Pekkila, Miikka Vaisala.

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

    May need to allocate >= 2 cores per task to get tryly parallel compute and disk IO
    SRUN="srun --account=project_2000403 --gres=gpu:v100:2 --mem=24000 -t 00:14:59 -p gputest
   --ntasks-per-socket=1 -n 2 -N 1 --cpus-per-task=2"
*/
#include "astaroth.h"
#include "astaroth_utils.h"
#include "errchk.h"

#if !AC_MPI_ENABLED
int
main(void)
{
    printf("The library was built without MPI support, cannot run mpitest. Rebuild Astaroth with "
           "cmake -DMPI_ENABLED=ON .. to enable.\n");
    return EXIT_FAILURE;
}
#else

#include <mpi.h>

#include <chrono>
#include <future>

#include <unistd.h>

static std::future<void> future;

void
write_async(void)
{
    for (size_t i = 0; i < NUM_VTXBUF_HANDLES; ++i) {
        char file[4096] = "";
        sprintf(file, "field-%lu.out", i);
        // acGridAccessMeshOnDisk((VertexBufferHandle)i, file, ACCESS_WRITE);
        acGridAccessMeshOnDiskAsync((VertexBufferHandle)i, file, ACCESS_WRITE);
    }
}

void
launch_disk_io(void)
{
    if (!future.valid()) { // Complete
        for (size_t i = 0; i < NUM_VTXBUF_HANDLES; ++i)
            acGridVolumeCopy((VertexBufferHandle)i, ACCESS_WRITE);

        future = std::async(std::launch::async, write_async);
    }
    else { // Not complete
        const auto status = future.wait_for(std::chrono::milliseconds(0));
        if (status == std::future_status::ready)
            future.get(); // Mark as completed
    }
}

int
main(int argc, char** argv)
{
    // MPI_Init(NULL, NULL);
    int thread_support_level;
    MPI_Init_thread(NULL, NULL, MPI_THREAD_MULTIPLE, &thread_support_level);
    if (thread_support_level < MPI_THREAD_MULTIPLE) {
        fprintf(stderr, "MPI_THREAD_MULTIPLE not supported by the MPI implementation\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        return EXIT_FAILURE;
    }

    int pid, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    AcMeshInfo info;
    acLoadConfig(AC_DEFAULT_CONFIG, &info);
    // Set mesh dimensions
    if (argc != 4) {
        fprintf(stderr, "Usage: ./mpi-io <nx> <ny> <nz>\n");
        return EXIT_FAILURE;
    }
    else {
        info.int_params[AC_nx] = atoi(argv[1]);
        info.int_params[AC_ny] = atoi(argv[2]);
        info.int_params[AC_nz] = atoi(argv[3]);
        acHostUpdateBuiltinParams(&info);
    }

    // Alloc
    AcMesh model, candidate;
    acHostMeshCreate(info, &model);
    acHostMeshCreate(info, &candidate);

    // Init
    acHostMeshRandomize(&model);
    acHostMeshRandomize(&candidate);
    acHostMeshApplyPeriodicBounds(&model);

    acGridInit(info);
    acGridLoadMesh(STREAM_DEFAULT, model);
    acGridPeriodicBoundconds(STREAM_DEFAULT);
    acGridStoreMesh(STREAM_DEFAULT, &candidate);

    if (!pid) {
        const AcResult res = acVerifyMesh("CPU-GPU Load/store", model, candidate);
        ERRCHK_ALWAYS(res == AC_SUCCESS);
    }

    /*
    // Write
    for (size_t i = 0; i < NUM_VTXBUF_HANDLES; ++i) {
        char file[4096] = "";
        sprintf(file, "field-%lu.out", i);
        acGridAccessMeshOnDisk((VertexBufferHandle)i, file, ACCESS_WRITE);
    }
    */
    launch_disk_io();

    auto status = future.wait_for(std::chrono::milliseconds(0));
    printf("Waiting");
    while (status != std::future_status::ready) {
        printf(".");
        fflush(stdout);
        usleep(100000);
        status = future.wait_for(std::chrono::milliseconds(0));
    }
    printf("\n");
    future.get(); // Sync

    // Scramble
    acHostMeshRandomize(&candidate);
    acGridLoadMesh(STREAM_DEFAULT, candidate);
    acGridAccessMeshOnDisk((VertexBufferHandle)0, "field-tmp.out",
                           ACCESS_WRITE); // Hacky, indirectly scramble vba.out to catch false
    // positives if the MPI calls fail completely.

    // Read
    for (size_t i = 0; i < NUM_VTXBUF_HANDLES; ++i) {
        char file[4096] = "";
        sprintf(file, "field-%lu.out", i);
        acGridAccessMeshOnDisk((VertexBufferHandle)i, file, ACCESS_READ);
    }

    acGridPeriodicBoundconds(STREAM_DEFAULT);
    acGridStoreMesh(STREAM_DEFAULT, &candidate);

    if (!pid) {
        const AcResult res = acVerifyMesh("MPI-IO disk read/write", model, candidate);
        ERRCHK_ALWAYS(res == AC_SUCCESS);
    }

    acGridQuit();

    MPI_Finalize();
    return EXIT_SUCCESS;
}
#endif
