#include <cstdio>
#include <cstdlib>

#include <mpi.h>

static inline void
errchk_print_mpi_api_error(const int errorcode, const char* function, const char* file,
                           const int line, const char* expression)
{
    char description[MPI_MAX_ERROR_STRING];
    int resultlen;
    MPI_Error_string(errorcode, description, &resultlen);
    fprintf(stderr, "%s\n", description);
}

#define ERRCHK_MPI_API(errcode)                                                                    \
    do {                                                                                           \
        const int _tmp_mpi_api_errcode_ = (errcode);                                               \
        if (_tmp_mpi_api_errcode_ != MPI_SUCCESS) {                                                \
            errchk_print_mpi_api_error(_tmp_mpi_api_errcode_, __func__, __FILE__, __LINE__,        \
                                       #errcode);                                                  \
            MPI_Abort(MPI_COMM_WORLD, -1);                                                         \
        }                                                                                          \
    } while (0)


int
main(void)
{
    int provided, claimed, is_thread_main;
    ERRCHK_MPI_API(MPI_Init_thread(NULL, NULL, MPI_THREAD_FUNNELED, &provided));
    ERRCHK_MPI_API(MPI_Query_thread(&claimed));
    ERRCHK_MPI_API(MPI_Is_thread_main(&is_thread_main));
    if (provided != claimed || !is_thread_main) {
        fprintf(stderr, "No multithreading support\n");
    }

    int rank, nprocs;
    ERRCHK_MPI_API(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    ERRCHK_MPI_API(MPI_Comm_size(MPI_COMM_WORLD, &nprocs));
    {
        int buf{1};
        MPI_File file{};
        MPI_Request req{};
        fprintf(stderr, "Opening file\n");
        ERRCHK_MPI_API(MPI_File_open(MPI_COMM_WORLD, "out.dat", MPI_MODE_CREATE | MPI_MODE_WRONLY,
                                     MPI_INFO_NULL, &file));
        // fprintf(stderr, "Launching iwrite_all\n");
        // ERRCHK_MPI_API(
        //     MPI_File_iwrite_all(file, &buf, 1, MPI_INT, &req)); // This line causes a segmentation
        //                                                         // on one machine but not on another
        // fprintf(stderr, "Waiting\n");
        // ERRCHK_MPI_API(MPI_Wait(&req, MPI_STATUS_IGNORE));
        ERRCHK_MPI_API(MPI_File_write_all(file, &buf, 1, MPI_INT, MPI_STATUS_IGNORE)); // This completes without errors
        fprintf(stderr, "File closed\n");
        ERRCHK_MPI_API(MPI_File_close(&file));
    }

    fprintf(stderr, "Completed succesfully\n");
    ERRCHK_MPI_API(MPI_Finalize());
    return EXIT_SUCCESS;
}
