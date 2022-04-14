#include <stdio.h>
#include <stdlib.h>

#include "astaroth.h"
#include "astaroth_utils.h"
#include "errchk.h"
#include "user_defines.h"

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

#define ARRAY_SIZE(arr) (sizeof(arr) / sizeof(arr[0]))

/*
cmake -DMPI_ENABLED=ON .. && make -j && $SRUNMPI4 ./pc-varfile-import\
--input=../mesh-scaler/build --volume=256,256,256
*/

int
main(int argc, char* argv[])
{
    MPI_Init(NULL, NULL);
    int pid, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    char input[4096] = "";
    size_t nx        = 0;
    size_t ny        = 0;
    size_t nz        = 0;

    for (int i = 0; i < argc; ++i) {
        sscanf(argv[i], "--input=%4095s", input);
        sscanf(argv[i], "--volume=%lu,%lu,%lu", &nx, &ny, &nz);
    }
    printf("Input: %s\n", input);
    printf("Volume: (%lu, %lu, %lu)\n", nx, ny, nz);

    ERRCHK(mx > 0);
    ERRCHK(my > 0);
    ERRCHK(mz > 0);
    ERRCHK(num_fields == NUM_VTXBUF_HANDLES);

    AcMeshInfo info;
    acLoadConfig(AC_DEFAULT_CONFIG, &info);
    info.int_params[AC_nx] = nx;
    info.int_params[AC_ny] = ny;
    info.int_params[AC_nz] = nz;
    acHostUpdateBuiltinParams(&info);

    // Init
    acGridInit(info);

    // Read
    for (size_t i = 0; i < NUM_VTXBUF_HANDLES; ++i) {
        char file[2 * 4096] = "";
        sprintf(file, "%s/%s.out", input, vtxbuf_names[i]);
        printf("Reading `%s`\n", file);
        acGridAccessMeshOnDiskSynchronous((VertexBufferHandle)i, file, ACCESS_READ);
        // acGridLoadFieldFromFile(file, (VertexBufferHandle)i);

        AcReal max, min, sum;
        acGridReduceScal(STREAM_DEFAULT, RTYPE_MAX, (VertexBufferHandle)i, &max);
        acGridReduceScal(STREAM_DEFAULT, RTYPE_MIN, (VertexBufferHandle)i, &min);
        acGridReduceScal(STREAM_DEFAULT, RTYPE_SUM, (VertexBufferHandle)i, &sum);
        printf("max %g, min %g, sum %g\n", (double)max, (double)min, (double)sum);
    }

    // Quit
    acGridQuit();
    MPI_Finalize();

    printf("OK!\n");

    return EXIT_SUCCESS;
}
#endif