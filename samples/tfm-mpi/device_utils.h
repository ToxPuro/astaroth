#pragma once
#include <vector>

#include "astaroth.h"

#include "acr_utils.h"

namespace ac {

AcMeshInfo get_info(const Device& device);
VertexBufferArray get_vba(const Device& device);

void swap_buffers(const Device& device);
void swap_buffers(const Device& device, const VertexBufferHandle& handle);
void swap_buffers(const Device& device, const std::vector<VertexBufferHandle>& handles);

std::vector<ac::device_view<AcReal>>
get_ptrs(const Device& device, const std::vector<Field>& fields, const BufferGroup& type);

std::vector<ac::device_view<AcReal>>
get_ptrs(const Device& device, const std::vector<Profile>& profiles, const BufferGroup& type);

// Revised 2025-06
ac::device_view<AcReal> get_dfield(const Device& device, const Field& field, const BufferGroup& group);

template<typename T>
auto pull_param(const Device& device, const T& param)
{
    return acr::get(get_info(device), param);
}

template<typename T, typename U>
auto push_param(const Device& device, const T& param, const U& value) { 
    auto info{get_info(device)}; 
    acr::set(param, value, info); 
    ERRCHK_AC(acDeviceLoadMeshInfo(device, info));
}

} // namespace ac
