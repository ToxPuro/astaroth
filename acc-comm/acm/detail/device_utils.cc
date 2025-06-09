#include "device_utils.h"

#include "acm/detail/errchk.h"

#if defined(ACM_DEVICE_ENABLED)

namespace ac::device {

int
device_count()
{
    int device_count{0};
    ERRCHK_CUDA_API(cudaGetDeviceCount(&device_count));
    return device_count;
}

} // namespace ac::device

#endif
