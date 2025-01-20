#include "device_utils.h"

#include "acm/detail/errchk.h"

namespace ac::device {

void
swap_buffers(const Device& device)
{
    ERRCHK(acDeviceSwapBuffers(device) == AC_SUCCESS);
}

void
swap_buffers(const Device& device, const VertexBufferHandle& handle)
{
    ERRCHK(acDeviceSwapBuffer(device, handle) == AC_SUCCESS);
}
void
swap_buffers(const Device& device, const std::vector<VertexBufferHandle>& handles)
{
    for (const auto& handle : handles)
        ERRCHK(acDeviceSwapBuffer(device, handle) == AC_SUCCESS);
}

} // namespace ac::device
