#include "errchk_mpi.h"

#include <mpi.h>

void
handle_error(const char* function, const char* file, const long line, const char* expression)
{
    errchk_print_error(function, file, line, expression, "");
    MPI_Abort(MPI_COMM_WORLD, -1);
}

void
handle_mpi_error(const int errorcode, const char* function, const char* file, const long line,
                 const char* expression)
{
    char description[MPI_MAX_ERROR_STRING];
    int resultlen;
    MPI_Error_string(errorcode, description, &resultlen);
    errchk_print_error(function, file, line, expression, description);
    MPI_Abort(MPI_COMM_WORLD, -1);
}
