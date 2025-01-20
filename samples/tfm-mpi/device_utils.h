#pragma once

#include "astaroth.h"

#include <vector>

namespace ac::device {
void swap_buffers(const Device& device);
void swap_buffers(const Device& device, const VertexBufferHandle& handle);
void swap_buffers(const Device& device, const std::vector<VertexBufferHandle>& handles);
} // namespace ac::device
