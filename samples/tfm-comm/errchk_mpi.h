#pragma once
#include "errchk.h"

#define ERRCHK_MPI(retval)                                                                         \
    do {                                                                                           \
        ERRCHK(retval);                                                                            \
        if ((retval) == 0) {                                                                       \
            MPI_Abort(MPI_COMM_WORLD, 0);                                                          \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
    } while (0)

#define ERRCHK_MPI_API(errorcode)                                                                  \
    do {                                                                                           \
        if ((errorcode) != MPI_SUCCESS) {                                                          \
            char description[MPI_MAX_ERROR_STRING];                                                \
            int resultlen;                                                                         \
            MPI_Error_string(errorcode, description, &resultlen);                                  \
            ERRCHKK((errorcode) == MPI_SUCCESS, description);                                      \
            MPI_Abort(MPI_COMM_WORLD, 0);                                                          \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
    } while (0)

// Debug: Disable all MPI API calls
// #undef ERRCHK_MPI_API
// #define ERRCHK_MPI_API(x)
