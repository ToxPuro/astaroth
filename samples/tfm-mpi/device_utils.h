#pragma once
#include <vector>

#include "astaroth.h"

#include "acr_utils.h"

namespace ac {
void swap_buffers(const Device& device);
void swap_buffers(const Device& device, const VertexBufferHandle& handle);
void swap_buffers(const Device& device, const std::vector<VertexBufferHandle>& handles);

std::vector<ac::device_view<AcReal>>
get_ptrs(const Device& device, const std::vector<Field>& fields, const BufferGroup& type);

std::vector<ac::device_view<AcReal>>
get_ptrs(const Device& device, const std::vector<Profile>& profiles, const BufferGroup& type);
} // namespace ac
