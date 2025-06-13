#include "mpi_utils_experimental.h"
#include "acm/detail/mpi_utils.h"

// For selecting the device
#if defined(ACM_DEVICE_ENABLED)
#include "acm/detail/cuda_utils.h"
#include "acm/detail/errchk_cuda.h"
#endif

namespace ac::mpi {

ac::mpi::comm
split(const MPI_Comm& parent_comm, const int color, const int key)
{
    MPI_Comm comm{MPI_COMM_NULL};
    ERRCHK_MPI_API(MPI_Comm_split(parent_comm, color, key, &comm));
    ERRCHK_MPI(comm != MPI_COMM_NULL);
    return ac::mpi::comm{comm, true};
}

void
barrier(const ac::mpi::comm& comm)
{
    ERRCHK_MPI_API(MPI_Barrier(comm.get()));
}

ac::shape
global_mm(const cart_comm& comm, const ac::index& rr)
{
    return ac::mpi::get_global_mm(comm.global_nn(), rr);
}

ac::shape
global_nn(const cart_comm& comm)
{
    return comm.global_nn();
}

ac::shape
local_mm(const cart_comm& comm, const ac::index& rr)
{
    return ac::mpi::get_local_mm(comm.get(), comm.global_nn(), rr);
}

ac::shape
local_nn(const cart_comm& comm)
{
    return ac::mpi::get_local_nn(comm.get(), comm.global_nn());
}

std::vector<ac::index>
get_rank_ordering(const MPI_Comm& cart_comm)
{
    std::vector<ac::index> coords;

    int nprocs{-1};
    ERRCHK_MPI_API(MPI_Comm_size(cart_comm, &nprocs));

    for (int i{0}; i < nprocs; ++i) {
        int       translated_rank{MPI_PROC_NULL};
        MPI_Group world_group{MPI_GROUP_NULL};
        ERRCHK_MPI_API(MPI_Comm_group(MPI_COMM_WORLD, &world_group));

        MPI_Group cart_group{MPI_GROUP_NULL};
        ERRCHK_MPI_API(MPI_Comm_group(cart_comm, &cart_group));

        ERRCHK_MPI_API(MPI_Group_translate_ranks(world_group, 1, &i, cart_group, &translated_rank));
        coords.push_back(ac::mpi::get_coords(cart_comm, translated_rank));
    }

    return coords;
}

int
select_device_lumi()
{
#if !defined(ACM_HOST_ONLY_MODE_ENABLED) && !defined(ACM_DEVICE_ENABLED)
#error "Tried to select device but both ACM_DEVICE_ENABLED and AC_HOST_ONLY_MODE_ENABLED were false"
#endif

#if defined(ACM_DEVICE_ENABLED)
    int device_count{0};
    ERRCHK_CUDA_API(cudaGetDeviceCount(&device_count));
    int device_id{ac::mpi::get_rank(MPI_COMM_WORLD) % device_count};
    if (device_count == 8) { // Do manual GPU mapping for LUMI
        ac::ntuple<int> device_ids{6, 7, 0, 1, 2, 3, 4, 5};
        device_id = device_ids[as<size_t>(device_id)];
    }
    ERRCHK_CUDA_API(cudaSetDevice(device_id));
    return device_id;
#else
    return -1;
#endif
}

} // namespace ac::mpi

namespace ac::mpi {

uint64_t
rank(const ac::mpi::comm& comm)
{
    return as<uint64_t>(ac::mpi::get_rank(comm.get()));
}

uint64_t
size(const ac::mpi::comm& comm)
{
    return as<uint64_t>(ac::mpi::get_size(comm.get()));
}

ac::index coords(const ac::mpi::cart_comm& comm)
{
    return ac::mpi::get_coords(comm.get());
}

} // namespace ac::mpi
