#pragma once

#include "buffer.h"
#include "ntuple.h"
#include "view.h"

using UserDatatype = double;

using Dims = ac::ntuple<UserDatatype>;

using HostPointer   = ac::host_view<UserDatatype>;
using DevicePointer = ac::device_view<UserDatatype>;

using DeviceBuffer                  = ac::device_buffer<UserDatatype>;
using HostBuffer                    = ac::host_buffer<UserDatatype>;
using PinnedHostBuffer              = ac::pinned_host_buffer<UserDatatype>;
using PinnedWriteCombinedHostBuffer = ac::pinned_write_combined_host_buffer<UserDatatype>;

Dims make_dims(const size_t count, const UserDatatype& fill_value);
