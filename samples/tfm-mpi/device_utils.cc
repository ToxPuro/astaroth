#include "device_utils.h"

#include "acm/detail/errchk.h"

#include "device_detail.h"

namespace ac {

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

std::vector<ac::device_view<AcReal>>
get_ptrs(const Device& device, const std::vector<Field>& fields, const BufferGroup& type)
{
    VertexBufferArray vba{};
    ERRCHK_AC(acDeviceGetVBA(device, &vba));

    return acr::get_ptrs(vba, fields, type);
}

std::vector<ac::device_view<AcReal>>
get_ptrs(const Device& device, const std::vector<Profile>& profiles, const BufferGroup& type)
{
    VertexBufferArray vba{};
    ERRCHK_AC(acDeviceGetVBA(device, &vba));

    return acr::get_ptrs(vba, profiles, type);
}

} // namespace ac
