#include "comm.h"
#include <stddef.h>
#include <stdlib.h>

#include <mpi.h>

#include "comm_data.h"
#include "print.h"

#define SUCCESS (0)
#define FAILURE (-1)
#define ARRAY_SIZE(arr) (sizeof(arr) / sizeof(arr[0]))

#define ERRCHK_MPI(retval)                                                                         \
    ERRCHK(retval);                                                                                \
    if ((retval) == 0) {                                                                           \
        MPI_Abort(MPI_COMM_WORLD, 0);                                                              \
        exit(EXIT_FAILURE);                                                                        \
    }

#define ERRCHK_MPI_API(errorcode)                                                                  \
    if ((errorcode) != MPI_SUCCESS) {                                                              \
        char description[MPI_MAX_ERROR_STRING];                                                    \
        int resultlen;                                                                             \
        MPI_Error_string(errorcode, description, &resultlen);                                      \
        ERRCHKK((errorcode) == MPI_SUCCESS, description);                                          \
        MPI_Abort(MPI_COMM_WORLD, 0);                                                              \
        exit(EXIT_FAILURE);                                                                        \
    }

#include "errchk.h"

int
acCommInit(void)
{
    // ERRCHK_MPI_API(MPI_Init(NULL, NULL));
    // return SUCCESS;
    return FAILURE;
}

int
acCommRun(void)
{
    const size_t dims[]   = {4};
    const size_t ndims    = ARRAY_SIZE(dims);
    const size_t fields[] = {1, 2};
    const size_t nfields  = ARRAY_SIZE(fields);
    CommData comm_data    = acCommDataCreate(ndims, nfields);

    acCommDataPrint("comm_data", comm_data);

    acCommDataDestroy(&comm_data);
    return SUCCESS;
}

int
acCommQuit(void)
{
    // ERRCHK_MPI_API(MPI_Finalize());
    // return SUCCESS;
    return FAILURE;
}
