#pragma once

#include "buffer.h"
#include "ntuple.h"
#include "pointer.h"

using UserDatatype = double;

using Index     = ac::ntuple<uint64_t>;
using Shape     = ac::ntuple<uint64_t>;
using Direction = ac::ntuple<int64_t>;
using Dims      = ac::ntuple<UserDatatype>;

using HostPointer   = ac::mr::host_pointer<UserDatatype>;
using DevicePointer = ac::mr::device_pointer<UserDatatype>;

using DeviceBuffer                  = ac::device_buffer<UserDatatype>;
using HostBuffer                    = ac::host_buffer<UserDatatype>;
using PinnedHostBuffer              = ac::pinned_host_buffer<UserDatatype>;
using PinnedWriteCombinedHostBuffer = ac::pinned_write_combined_host_buffer<UserDatatype>;

// using DeviceNdBuffer = ac::ndbuffer<UserDatatype, ac::mr::device_allocator>;
// using HostNdBuffer   = ac::ndbuffer<UserDatatype, ac::mr::host_allocator>;

Index     make_index(const size_t count, const uint64_t& fill_value);
Shape     make_shape(const size_t count, const uint64_t& fill_value);
Direction make_direction(const size_t count, const int64_t& fill_value);
Dims      make_dims(const size_t count, const int64_t& fill_value);

void test_datatypes();
