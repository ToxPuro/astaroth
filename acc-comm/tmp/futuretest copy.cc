/*
    Copyright (C) 2024, Johannes Pekkila.

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/
#include <future>
#include <iostream>
#include <vector>

#include <mpi.h>

/**
 * Building and running:
 * mpic++ -std=c++17 ../futuretest.cc && mpirun -n 4 ./a.out
 */

#define MPI_SYNCHRONOUS_BLOCK_START(communicator)                                                  \
    {                                                                                              \
        MPI_Barrier(communicator);                                                                 \
        fflush(stdout);                                                                            \
        MPI_Barrier(communicator);                                                                 \
        int rank__, nprocs_;                                                                       \
        ERRCHK_MPI_API(MPI_Comm_rank(communicator, &rank__));                                      \
        ERRCHK_MPI_API(MPI_Comm_size(communicator, &nprocs_));                                     \
        for (int i__ = 0; i__ < nprocs_; ++i__) {                                                  \
            MPI_Barrier(communicator);                                                             \
            if (i__ == rank__) {                                                                   \
                printf("---Rank %d---\n", rank__);

#define MPI_SYNCHRONOUS_BLOCK_END(communicator)                                                    \
    }                                                                                              \
    fflush(stdout);                                                                                \
    MPI_Barrier(communicator);                                                                     \
    }                                                                                              \
    MPI_Barrier(communicator);                                                                     \
    }

#define PRINT_DEBUG_MPI(expr, communicator)                                                        \
    do {                                                                                           \
        MPI_SYNCHRONOUS_BLOCK_START((communicator))                                                \
        PRINT_DEBUG((expr));                                                                       \
        MPI_SYNCHRONOUS_BLOCK_END((communicator))                                                  \
    } while (0)

#define PRINT_LOG(...)                                                                             \
    do {                                                                                           \
        std::fprintf(stderr, __VA_ARGS__);                                                         \
        std::fprintf(stderr, "\n");                                                                \
    } while (0)

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

template <typename T>
static void
print(const std::string& label, const std::vector<T>& vec)
{
    std::cout << label << ": { ";
    for (const auto& elem : vec)
        std::cout << elem << " ";
    std::cout << "}" << std::endl;
}

int
main(void)
{
    int provided, claimed, is_thread_main;
    ERRCHK_MPI_API(MPI_Init_thread(NULL, NULL, MPI_THREAD_MULTIPLE, &provided));
    ERRCHK_MPI_API(MPI_Query_thread(&claimed));
    ERRCHK_MPI_API(MPI_Is_thread_main(&is_thread_main));
    if (provided != claimed || !is_thread_main) {
        fprintf(stderr, "No multithreading support\n");
    }

    int rank, nprocs;
    ERRCHK_MPI_API(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    ERRCHK_MPI_API(MPI_Comm_size(MPI_COMM_WORLD, &nprocs));

    const size_t count = 10;
    const size_t rr    = 2;
    std::vector<int> vec(2 * rr + count, rank);

    MPI_SYNCHRONOUS_BLOCK_START(MPI_COMM_WORLD);
    print("before", vec);
    MPI_SYNCHRONOUS_BLOCK_END(MPI_COMM_WORLD);

    std::future<MPI_Request> forward = std::async(std::launch::async, [&vec, rank, nprocs, count,
                                                                       rr]() {
        MPI_Comm comm;
        PRINT_LOG("fdup %d", rank);
        ERRCHK_MPI_API(MPI_Comm_dup(MPI_COMM_WORLD, &comm));

        MPI_Request recv_req;
        PRINT_LOG("  frecv %d <- %d", rank, (nprocs + rank + 1) % nprocs);
        ERRCHK_MPI_API(MPI_Irecv(vec.data() + vec.size() - rr, rr, MPI_INT,
                                 (nprocs + rank + 1) % nprocs, 0, comm, &recv_req));
        PRINT_LOG(" fsend %d -> %d", rank, (nprocs + rank - 1) % nprocs);
        ERRCHK_MPI_API(MPI_Send(vec.data(), rr, MPI_INT, (nprocs + rank - 1) % nprocs, 0, comm));

        PRINT_LOG("        ffree %d (DONE)", rank);
        ERRCHK_MPI_API(MPI_Comm_free(&comm));
        PRINT_LOG("            freturn %d (DONE)", rank);
        return recv_req;
    });
    // Deadlock here 50% of the time: why? Works if everything is in the same future but not if
    // separate.
    // ---------------------
    // Potential solution: MPI standard specifies (where?) that calls are executed sequentially even
    // if called concurrently: therefore, if one thread is blocking on one MPI call, another thread
    // cannot launch any further MPI calls before the first thread has completed its MPI call
    //
    // Gropp lecture slides: "When multiple threads make MPI calls concurrently, the outcome will be
    // as if the calls executed sequentially in some (any) order"
    //
    // Immediate result: achieving asynchronous execution with synchronous MPI + threads *is not
    // possible* on a single core. The only way to achieve truly asynchronous calls with MPI is to
    // uses the asynchronous interface or use a separate program with its own MPI context.
    // ---------------------
    std::future<MPI_Request> backward = std::async(std::launch::async, [&vec, rank, nprocs, count,
                                                                        rr]() {
        MPI_Comm comm;
        PRINT_LOG("bdup %d", rank);
        ERRCHK_MPI_API(MPI_Comm_dup(MPI_COMM_WORLD, &comm));

        MPI_Request recv_req;
        PRINT_LOG("  brecv %d <- %d", rank, (nprocs + rank - 1) % nprocs);
        ERRCHK_MPI_API(
            MPI_Irecv(vec.data(), rr, MPI_INT, (nprocs + rank - 1) % nprocs, 1, comm, &recv_req));
        PRINT_LOG(" bsend %d -> %d", rank, (nprocs + rank + 1) % nprocs);
        ERRCHK_MPI_API(MPI_Send(vec.data() + vec.size() - 2 * rr, rr, MPI_INT,
                                (nprocs + rank + 1) % nprocs, 1, comm));

        PRINT_LOG("        bfree %d (DONE)", rank);
        ERRCHK_MPI_API(MPI_Comm_free(&comm));
        PRINT_LOG("            breturn %d (DONE)", rank);
        return recv_req;
    });
    PRINT_LOG("            fget %d", rank);
    MPI_Request fwd_req = forward.get();
    PRINT_LOG("            bget %d", rank);
    MPI_Request bwd_req = backward.get();
    PRINT_LOG("            waitfwd %d", rank);
    ERRCHK_MPI_API(MPI_Wait(&fwd_req, MPI_STATUS_IGNORE));
    PRINT_LOG("            waitbwd %d", rank);
    ERRCHK_MPI_API(MPI_Wait(&bwd_req, MPI_STATUS_IGNORE));
    // ERRCHK_MPI_API(MPI_Waitall(2, (MPI_Request[]){fwd_req, bwd_req}, MPI_STATUSES_IGNORE));
    PRINT_LOG("            done %d", rank);

    // MPI_SYNCHRONOUS_BLOCK_START(MPI_COMM_WORLD);
    // print("after", vec);
    // MPI_SYNCHRONOUS_BLOCK_END(MPI_COMM_WORLD);

    ERRCHK_MPI_API(MPI_Finalize());
    return EXIT_SUCCESS;
}
