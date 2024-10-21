#pragma once
#include "errchk_print.h"

#include <mpi.h>

static inline void
errchk_print_mpi_api_error(const int errorcode, const char* function, const char* file,
                           const long line, const char* expression)
{
    char description[MPI_MAX_ERROR_STRING];
    int resultlen;
    MPI_Error_string(errorcode, description, &resultlen);
    errchk_print_error(function, file, line, expression, description);
}

#define ERRCHK_MPI_API(errcode)                                                                    \
    do {                                                                                           \
        const int errchk_mpi_api_code__ = (errcode);                                               \
        if (errchk_mpi_api_code__ != MPI_SUCCESS) {                                                \
            errchk_print_mpi_api_error(errchk_mpi_api_code__, __func__, __FILE__, __LINE__,        \
                                       #errcode);                                                  \
            throw std::runtime_error("MPI API error");                                             \
        }                                                                                          \
    } while (0)

// Debug: Disable all MPI API calls
// #undef ERRCHK_MPI_API
// #define ERRCHK_MPI_API(x)
