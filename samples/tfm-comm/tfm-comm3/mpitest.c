#include <stdio.h>
#include <stdlib.h>

#include <mpi.h>

#include "errchk.h"

// #include <stdarg.h>
// void
// print(const char* expr, const char* fmt, ...)
// {
//     if (expr && expr[0] != '\0')
//         fprintf(stderr, "Expression '%s' evaluated false\n", expr);
//     if (fmt && fmt[0] != '\0') {
//         fprintf(stderr, "Description: ");
//         va_list args;
//         va_start(args, fmt);
//         vfprintf(stderr, fmt, args);
//         va_end(args);
//         fprintf(stderr, "\n");
//     }
//     printf("\n");
// }

// #define ERROR_VA(...) print("", __VA_ARGS__)
// #define ERROR_EXPR(expr) print(#expr, "")
// #define ERROR_EXPR_VA(expr, ...) print(#expr, __VA_ARGS__)

int
main(void)
{
    MPI_Init(NULL, NULL);

    MPI_Finalize();
    return EXIT_SUCCESS;
}
