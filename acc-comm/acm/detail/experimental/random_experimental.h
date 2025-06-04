#pragma once
#include "acm/detail/view.h"

namespace acm::experimental {

void randomize(ac::host_view<double> ptr);
void randomize(ac::device_view<double> ptr);

void randomize(ac::host_view<uint64_t> ptr);
void randomize(ac::device_view<uint64_t> ptr);

} // namespace acm::experimental
