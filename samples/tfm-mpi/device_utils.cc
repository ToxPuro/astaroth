#include "device_utils.h"

#include "acm/detail/errchk.h"

#include "device_detail.h"

namespace ac {

AcMeshInfo get_info(const Device& device)
{
    AcMeshInfo info{};
    ERRCHK_AC(acDeviceGetLocalConfig(device, &info));
    return info;
}

VertexBufferArray get_vba(const Device& device) {
    VertexBufferArray vba{};
    ERRCHK_AC(acDeviceGetVBA(device, &vba));
    return vba;
}

// AcReal* get_input_field(const Device& device, const Field& handle) {
//     auto vba{get_vba(device)};
//     ERRCHK(handle < NUM_FIELDS);
//     return vba.in[handle];
// }

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

ac::device_view<AcReal> get_dfield(const Device& device, const Field& field, const BufferGroup& group)
{
    VertexBufferArray vba{};
    ERRCHK_AC(acDeviceGetVBA(device, &vba));
    return acr::make_ptr(vba, field, group);
}

} // namespace ac
