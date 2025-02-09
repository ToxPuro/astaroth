#pragma once

#include "buffer.h"
#include "ntuple.h"
#include "pointer.h"

using UserDatatype = double;

using Dims = ac::ntuple<UserDatatype>;

using HostPointer   = ac::mr::host_pointer<UserDatatype>;
using DevicePointer = ac::mr::device_pointer<UserDatatype>;

using DeviceBuffer                  = ac::device_buffer<UserDatatype>;
using HostBuffer                    = ac::host_buffer<UserDatatype>;
using PinnedHostBuffer              = ac::pinned_host_buffer<UserDatatype>;
using PinnedWriteCombinedHostBuffer = ac::pinned_write_combined_host_buffer<UserDatatype>;

Dims make_dims(const size_t count, const int64_t& fill_value);

void test_datatypes();
