#pragma once
#include "errchk.h"

// #define ERRCHK_MPI(expr)                                                                           \
//     ((expr) ? (expr) : (ERROR_EXPR(expr), MPI_Abort(MPI_COMM_WORLD, 0), (expr)))

#define ERRCHK_MPI(retval)                                                                         \
    do {                                                                                           \
        ERRCHK(retval);                                                                            \
        if (!(retval)) {                                                                           \
            MPI_Abort(MPI_COMM_WORLD, 0);                                                          \
        }                                                                                          \
    } while (0)

#define ERRCHK_MPI_API(errorcode)                                                                  \
    do {                                                                                           \
        if ((errorcode) != MPI_SUCCESS) {                                                          \
            ERRCHK((errorcode) == MPI_SUCCESS);                                                    \
            char description__[MPI_MAX_ERROR_STRING];                                              \
            int resultlen__;                                                                       \
            MPI_Error_string((errorcode), description__, &resultlen__);                            \
            ERROR(description__);                                                                  \
            MPI_Abort(MPI_COMM_WORLD, 0);                                                          \
        }                                                                                          \
    } while (0)

// Debug: Disable all MPI API calls
// #undef ERRCHK_MPI_API
// #define ERRCHK_MPI_API(x)
