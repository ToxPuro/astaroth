#pragma once

#include "buffer.h"
#include "ntuple.h"
#include "pointer.h"

using UserDatatype = double;

using Index     = ac::ntuple<uint64_t>;
using Shape     = ac::ntuple<uint64_t>;
using Direction = ac::ntuple<int64_t>;
using Dims      = ac::ntuple<UserDatatype>;

using HostPointer   = ac::mr::pointer<UserDatatype, ac::mr::host_allocator>;
using DevicePointer = ac::mr::pointer<UserDatatype, ac::mr::device_allocator>;

using DeviceBuffer     = ac::buffer<UserDatatype, ac::mr::device_allocator>;
using HostBuffer       = ac::buffer<UserDatatype, ac::mr::host_allocator>;
using PinnedHostBuffer = ac::buffer<UserDatatype, ac::mr::pinned_host_allocator>;

// using DeviceNdBuffer = ac::ndbuffer<UserDatatype, ac::mr::device_allocator>;
// using HostNdBuffer   = ac::ndbuffer<UserDatatype, ac::mr::host_allocator>;

Index     make_index(const size_t count, const uint64_t& fill_value);
Shape     make_shape(const size_t count, const uint64_t& fill_value);
Direction make_direction(const size_t count, const int64_t& fill_value);
Dims      make_dims(const size_t count, const int64_t& fill_value);

void test_datatypes();
