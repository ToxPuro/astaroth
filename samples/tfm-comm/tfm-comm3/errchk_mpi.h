#pragma once
#include "errchk.h"

#include <mpi.h>

static inline void
raise_mpi_api_error(const int errorcode, const char* function, const char* file, const long line,
                    const char* expression)
{
    char description[MPI_MAX_ERROR_STRING];
    int resultlen;
    MPI_Error_string(errorcode, description, &resultlen);
    errchk_raise_error(function, file, line, expression, description);
}

#define ERRCHK_MPI_API(errorcode)                                                                  \
    (((errorcode) == MPI_SUCCESS)                                                                  \
         ? 0                                                                                       \
         : ((raise_mpi_api_error((errorcode), __func__, __FILE__, __LINE__, #errorcode)), -1))

#define ERRCHK_MPI(expr) (ERRCHK(expr))

// Debug: Disable all MPI API calls
// #undef ERRCHK_MPI_API
// #define ERRCHK_MPI_API(x)
