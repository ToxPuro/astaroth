#pragma once

#include <vector>

#include "math_utils.h"
#include "partition.h"

#include "errchk_mpi.h"
#include "mpi_utils.h"
#include <mpi.h>

/** Launches the halo exchange and returns recv requests that must
 * be waited on to confirm that the exchange is complete.
 * Wait on send requests are called automatically, and when this function
 * returns, the send buffer can be freely modified.
 */
template <typename T>
std::vector<MPI_Request>
launch_halo_exchange(const MPI_Comm& parent_comm, const Shape& local_mm, const Shape& local_nn,
                     const Shape& rr, const T* send_data, T* recv_data)
{
    // Duplicate the communicator to ensure the operation does not interfere
    // with other operations on the parent communicator
    MPI_Comm cart_comm;
    ERRCHK_MPI_API(MPI_Comm_dup(parent_comm, &cart_comm));

    // Partition the domain
    auto segments{partition(local_mm, local_nn, rr)};

    // Prune the segment containing the computational domain
    for (size_t i{0}; i < segments.size(); ++i) {
        if (within_box(segments[i].offset, local_nn, rr)) {
            segments.erase(segments.begin() + as<long>(i));
            --i;
        }
    }

    std::vector<MPI_Request> send_reqs;
    std::vector<MPI_Request> recv_reqs;
    int16_t tag{0};
    for (const ac::segment& segment : segments) {
        const Index recv_offset{segment.offset};
        const Index send_offset{((local_nn + recv_offset - rr) % local_nn) + rr};
        MPI_Datatype recv_subarray{
            ac::mpi::subarray_create(local_mm, segment.dims, recv_offset, ac::mpi::get_dtype<T>())};
        MPI_Datatype send_subarray{
            ac::mpi::subarray_create(local_mm, segment.dims, send_offset, ac::mpi::get_dtype<T>())};

        const Direction recv_direction{ac::mpi::get_direction(segment.offset, local_nn, rr)};
        const int recv_neighbor{ac::mpi::get_neighbor(cart_comm, recv_direction)};
        const int send_neighbor{ac::mpi::get_neighbor(cart_comm, -recv_direction)};

        MPI_Request recv_req;
        ERRCHK_MPI_API(
            MPI_Irecv(recv_data, 1, recv_subarray, recv_neighbor, tag, cart_comm, &recv_req));
        recv_reqs.push_back(recv_req);

        MPI_Request send_req;
        ERRCHK_MPI_API(
            MPI_Isend(send_data, 1, send_subarray, send_neighbor, tag, cart_comm, &send_req));
        send_reqs.push_back(send_req);

        ERRCHK_MPI_API(MPI_Type_free(&send_subarray));
        ERRCHK_MPI_API(MPI_Type_free(&recv_subarray));
        ac::mpi::increment_tag(tag);
    }
    while (!send_reqs.empty()) {
        ac::mpi::request_wait_and_destroy(&send_reqs.back());
        send_reqs.pop_back();
    }

    ERRCHK_MPI_API(MPI_Comm_free(&cart_comm));
    return recv_reqs;
}

void test_halo_exchange(void);
