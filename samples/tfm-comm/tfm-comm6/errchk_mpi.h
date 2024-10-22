#pragma once

#include "errchk.h"

#include <mpi.h>

static inline void
errchk_print_mpi_api_error(const int errorcode, const char* function, const char* file,
                           const int line, const char* expression)
{
    char description[MPI_MAX_ERROR_STRING];
    int resultlen;
    MPI_Error_string(errorcode, description, &resultlen);
    errchk_print_error(function, file, line, expression, description);
}

#define ERRCHK_MPI_API(errcode)                                                                    \
    do {                                                                                           \
        const int _tmp_mpi_api_errcode_ = (errcode);                                               \
        if (_tmp_mpi_api_errcode_ != MPI_SUCCESS) {                                                \
            errchk_print_mpi_api_error(_tmp_mpi_api_errcode_, __func__, __FILE__, __LINE__,        \
                                       #errcode);                                                  \
            errchk_print_stacktrace();                                                             \
            throw std::runtime_error("MPI API error");                                             \
        }                                                                                          \
    } while (0)
