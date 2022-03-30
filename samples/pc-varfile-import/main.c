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

int
main(int argc, char* argv[])
{
    MPI_Init(NULL, NULL);
    int pid, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    char input[4096] = "";
    size_t mx        = 0;
    size_t my        = 0;
    size_t mz        = 0;

    for (int i = 0; i < argc; ++i) {
        sscanf(argv[i], "--input=%4095s", input);
        sscanf(argv[i], "--volume=%lu,%lu,%lu", &mx, &my, &mz);
    }
    printf("Input: %s\n", input);
    printf("Volume: (%lu, %lu, %lu)\n", mx, my, mz);

    ERRCHK(mx > 0);
    ERRCHK(my > 0);
    ERRCHK(mz > 0);
    ERRCHK(num_fields == NUM_VTXBUF_HANDLES);

    AcMeshInfo info;
    acLoadConfig(AC_DEFAULT_CONFIG, &info);
    info.int_params[AC_nx] = mx - 2 * STENCIL_ORDER;
    info.int_params[AC_ny] = my - 2 * STENCIL_ORDER;
    info.int_params[AC_nz] = mz - 2 * STENCIL_ORDER;
    acHostUpdateBuiltinParams(&info);

    // Init
    acGridInit(info);

    // Read
    for (size_t i = 0; i < NUM_VTXBUF_HANDLES; ++i) {
        char file[4096] = "";
        sprintf(file, "%s/%s.out", input, vtxbuf_names[i]);
        acGridAccessMeshOnDiskSynchronous((VertexBufferHandle)i, file, ACCESS_READ);
    }

    // Quit
    acGridQuit();
    MPI_Finalize();

    printf("OK!\n");

    return EXIT_SUCCESS;
}
#endif