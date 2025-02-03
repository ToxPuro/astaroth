#pragma once

#include "buffer.h"
#include "ntuple.h"

using UserDatatype = double;

using Index     = ac::ntuple<uint64_t>;
using Shape     = ac::ntuple<uint64_t>;
using Direction = ac::ntuple<int64_t>;
using Dims      = ac::ntuple<UserDatatype>;

using DeviceBuffer     = ac::buffer<UserDatatype, ac::mr::device_memory_resource>;
using HostBuffer       = ac::buffer<UserDatatype, ac::mr::host_memory_resource>;
using PinnedHostBuffer = ac::buffer<UserDatatype, ac::mr::pinned_host_memory_resource>;

Index     make_index(const size_t count, const uint64_t& fill_value);
Shape     make_shape(const size_t count, const uint64_t& fill_value);
Direction make_direction(const size_t count, const int64_t& fill_value);
Dims      make_dims(const size_t count, const int64_t& fill_value);
