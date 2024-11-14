#include <future>
#include <iostream>
#include <vector>

#include "errchk_mpi.h"
#include "mpi_utils.h"
#include "print_debug.h"

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
    ERRCHK_MPI(provided == claimed);
    ERRCHK_MPI_API(MPI_Is_thread_main(&is_thread_main));
    ERRCHK_MPI(is_thread_main);

    int rank, nprocs;
    ERRCHK_MPI_API(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    ERRCHK_MPI_API(MPI_Comm_size(MPI_COMM_WORLD, &nprocs));

    const size_t count = 10;
    const size_t rr    = 2;
    std::vector<int> vec(2 * rr + count, rank);

    MPI_SYNCHRONOUS_BLOCK_START(MPI_COMM_WORLD);
    print("before", vec);
    MPI_SYNCHRONOUS_BLOCK_END(MPI_COMM_WORLD);

#if 0
    std::future<void> forward = std::async(std::launch::async, [&vec, rank, nprocs, count, rr]() {
        MPI_Comm comm;
        PRINT_LOG("fdup %d", rank);
        ERRCHK_MPI_API(MPI_Comm_dup(MPI_COMM_WORLD, &comm));
        PRINT_LOG(" fsend %d -> %d", rank, (nprocs + rank - 1) % nprocs);

        ERRCHK_MPI_API(MPI_Send(vec.data(), rr, MPI_INT, (nprocs + rank - 1) % nprocs, 0, comm));
        PRINT_LOG("  frecv %d <- %d", rank, (nprocs + rank + 1) % nprocs);
        ERRCHK_MPI_API(MPI_Recv(vec.data() + vec.size() - rr, rr, MPI_INT,
                                (nprocs + rank + 1) % nprocs, 0, comm, MPI_STATUS_IGNORE));

        PRINT_LOG("        ffree %d (DONE)", rank);
        ERRCHK_MPI_API(MPI_Comm_free(&comm));
    });
    // Deadlock here: why? Works if everything is in the same future but not if separate
    // There may be some synchronization going on under the hood (also MPI calls not interrupt safe)
    // Likely solution: MPI uses mutexes under the hood and does not allow true concurrency,
    // e.g. if MPI_Send is started on one thread, it sets a mutex that blocks the send of another
    // thread
    std::future<void> backward = std::async(std::launch::async, [&vec, rank, nprocs, count, rr]() {
        MPI_Comm comm;
        PRINT_LOG("bdup %d", rank);
        ERRCHK_MPI_API(MPI_Comm_dup(MPI_COMM_WORLD, &comm));
        PRINT_LOG(" bsend %d -> %d", rank, (nprocs + rank + 1) % nprocs);

        ERRCHK_MPI_API(MPI_Send(vec.data() + vec.size() - 2 * rr, rr, MPI_INT,
                                (nprocs + rank + 1) % nprocs, 1, comm));
        PRINT_LOG("  brecv %d <- %d", rank, (nprocs + rank - 1) % nprocs);
        ERRCHK_MPI_API(MPI_Recv(vec.data(), rr, MPI_INT, (nprocs + rank - 1) % nprocs, 1, comm,
                                MPI_STATUS_IGNORE));

        PRINT_LOG("        bfree %d (DONE)", rank);
        ERRCHK_MPI_API(MPI_Comm_free(&comm));
    });
    forward.wait();
    backward.wait();
#endif

#if 0
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
        return recv_req;
    });
    // Deadlock here again: why? Works if everything is in the same future but not if separate
    // There may be some synchronization going on under the hood (also MPI calls not interrupt safe)
    // Likely solution: MPI uses mutexes under the hood and does not allow true concurrency,
    // e.g. if MPI_Send is started on one thread, it sets a mutex that blocks the send of another
    // thread
    // ---------------------
    // Solution: MPI standard specifies that the calls are executed sequentially even if called
    // concurrently: therefore, if one thread is blocking on one MPI call, another thread
    // cannot launch any further MPI calls before the first thread has completed its MPI call
    //
    // Gropp lecture slides: "When multiple threads make MPI calls concurrently, the outcome will be as if the calls executed sequentially in some (any) order"
    //
    // Immediate result: achieving asynchronous execution with synchronous MPI + threads *is not possible*.
    // The only way to achieve truly asynchronous calls with MPI is to uses the asynchronous
    // interface or use a separate program with its own MPI context.
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
        return recv_req;
    });
    MPI_Request fwd_req               = forward.get();
    MPI_Request bwd_req               = backward.get();
    ERRCHK_MPI_API(MPI_Waitall(2, (MPI_Request[]){fwd_req, bwd_req}, MPI_STATUSES_IGNORE));
    // ERRCHK_MPI_API(MPI_Wait(&fwd_req, MPI_STATUS_IGNORE));
    // ERRCHK_MPI_API(MPI_Wait(&bwd_req, MPI_STATUS_IGNORE));
#endif

    MPI_SYNCHRONOUS_BLOCK_START(MPI_COMM_WORLD);
    print("after", vec);
    MPI_SYNCHRONOUS_BLOCK_END(MPI_COMM_WORLD);

    ERRCHK_MPI_API(MPI_Finalize());
    return EXIT_SUCCESS;
}
