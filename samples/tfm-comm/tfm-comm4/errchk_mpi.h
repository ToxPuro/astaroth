#pragma once

#include "errchk.h"

#include <mpi.h>

static inline void
handle_error(const char* function, const char* file, const long line, const char* expression)
{
    errchk_raise_error(function, file, line, expression, "");
    MPI_Abort(MPI_COMM_WORLD, -1);
}

static inline void
handle_mpi_error(const int errorcode, const char* function, const char* file, const long line,
                 const char* expression)
{
    char description[MPI_MAX_ERROR_STRING];
    int resultlen;
    MPI_Error_string(errorcode, description, &resultlen);
    errchk_raise_error(function, file, line, expression, description);
    MPI_Abort(MPI_COMM_WORLD, -1);
}

#define ERRCHK_MPI_API(errorcode)                                                                  \
    (((errorcode) == MPI_SUCCESS)                                                                  \
         ? 0                                                                                       \
         : ((handle_mpi_error((errorcode), __func__, __FILE__, __LINE__, #errorcode)), -1))

#define ERRCHK_MPI(expr) ((expr) ? 0 : ((handle_error(__func__, __FILE__, __LINE__, #expr)), -1))
