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
    const Field fields[] = {VTXBUF_UUX, VTXBUF_UUY, VTXBUF_UUZ, VTXBUF_LNRHO, VTXBUF_AX, VTXBUF_AY, VTXBUF_AZ};
    /*
    const Field fields[] = {
        vel, density, magnetic field
        //VTXBUF_AX,      VTXBUF_AY,  VTXBUF_AZ,  VTXBUF_LNRHO,
        //VTXBUF_ENTROPY, VTXBUF_UUX, VTXBUF_UUY, VTXBUF_UUZ,

    };
    */
    const size_t num_fields = ARRAY_SIZE(fields);
    const int3 nn = (int3){4096, 4096, 4096};
    const int3 rr = (int3){3, 3, 3};

    AcMeshInfo info;
    acLoadConfig(AC_DEFAULT_CONFIG, &info);
    info.int_params[AC_nx] = nx;
    info.int_params[AC_ny] = ny;
    info.int_params[AC_nz] = nz;
    acHostUpdateBuiltinParams(&info);

    // Init
    acGridInit(info);

    acGridReadVarfileToMesh(file, fields, num_fields, nn, rr);
    
    for (size_t i = 0; i < NUM_VTXBUF_HANDLES; ++i) {
        AcReal buf_max, buf_min, buf_rms;
        acGridReduceScal(STREAM_DEFAULT, RTYPE_MAX, i, &buf_max);
        acGridReduceScal(STREAM_DEFAULT, RTYPE_MIN, i, &buf_min);
        acGridReduceScal(STREAM_DEFAULT, RTYPE_RMS, i, &buf_rms);

        printf("  %*s: min %.3e,\trms %.3e,\tmax %.3e\n", 8, vtxbuf_names[i],
               (double)(buf_min), (double)(buf_rms), (double)(buf_max));
    }

    // Create a tmpdir for output
    const int job_id = 12345;
    char job_dir[4096];
    snprintf(job_dir, 4096, "output-%d", job_id);
    
    char cmd[4096];
    snprintf(cmd, 4096, "mkdir -p %s", job_dir);
    system(cmd);
    
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