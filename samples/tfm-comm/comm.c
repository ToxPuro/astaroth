#include "comm.h"

#define SUCCESS (0)
#define FAILURE (-1)

#define ERRCHK_MPI(retval)                                                                         \
    ERRCHK(retval);                                                                                \
    if ((retval) == 0) {                                                                           \
        MPI_Abort(MPI_COMM_WORLD, 0);                                                              \
        exit(EXIT_FAILURE);                                                                        \
    }

int
acCommInit(void)
{
    return SUCCESS;
}

int
acCommQuit(void)
{
    return SUCCESS;
}
