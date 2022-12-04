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

    // Modify these based on the varfile format
    const char* file = "/scratch/project_462000077/mkorpi/forced/mahti_4096/data/allprocs/var.dat";
    const Field fields[] = {
        VTXBUF_AX,      VTXBUF_AY,  VTXBUF_AZ,  VTXBUF_LNRHO,
        VTXBUF_ENTROPY, VTXBUF_UUX, VTXBUF_UUY, VTXBUF_UUZ,

    };
    const size_t num_fields = ARRAY_SIZE(fields);
    const int3 nn = (int3){4096, 4096, 4096};
    const int3 rr = (int3){3, 3, 3};

    AcMeshInfo info;
    acLoadConfig(AC_DEFAULT_CONFIG, &info);
    info.int_params[AC_nx] = nn.x;
    info.int_params[AC_ny] = nn.y;
    info.int_params[AC_nz] = nn.z;
    acHostUpdateBuiltinParams(&info);

    // Init
    acGridInit(info);

    acGridReadVarfileToMesh(file, fields, num_fields, nn, rr);

    const int job_id = 12345;
    char job_dir[4096];
    snprintf(job_dir, 4096, "output-%d", job_id);
    for (size_t i = 0; i < NUM_VTXBUF_HANDLES; ++i)
        acGridAccessMeshOnDiskSynchronous((VertexBufferHandle)i, job_dir, vtxbuf_names[i], ACCESS_WRITE);

    // Quit
    acGridQuit();
    MPI_Finalize();

    if (!pid)
        printf("OK!\n");

    return EXIT_SUCCESS;
}
#endif