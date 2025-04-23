#include "mpi_utils_experimental.h"

// For selecting the device
#if defined(ACM_DEVICE_ENABLED)
#include "acm/detail/cuda_utils.h"
#include "acm/detail/errchk_cuda.h"
#endif

namespace ac::mpi {

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
