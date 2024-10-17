#pragma once
#include "errchk.h"

void handle_error(const char* function, const char* file, const long line, const char* expression);

void handle_mpi_error(const int errorcode, const char* function, const char* file, const long line,
                      const char* expression);

#define ERRCHK_MPI_API(errorcode)                                                                  \
    (((errorcode) == MPI_SUCCESS)                                                                  \
         ? 0                                                                                       \
         : ((handle_mpi_error((errorcode), __func__, __FILE__, __LINE__, #errorcode)), -1))

#define ERRCHK_MPI(expr) ((expr) ? 0 : ((handle_error(__func__, __FILE__, __LINE__, #expr)), -1))
#define ERRCHK_MPI(expr) ((expr) ? 0 : ((handle_error(__func__, __FILE__, __LINE__, #expr)), -1))

// Debug: Disable all MPI API calls
// #undef ERRCHK_MPI_API
// #define ERRCHK_MPI_API(x)
