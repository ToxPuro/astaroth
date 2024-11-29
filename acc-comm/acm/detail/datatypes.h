#pragma once

#include "vector.h"
using Index     = ac::vector<uint64_t>;
using Shape     = ac::vector<uint64_t>;
using Direction = ac::vector<int64_t>;

#include "buffer.h"
using UserDatatype     = double;
using DeviceBuffer     = ac::buffer<UserDatatype, ac::mr::device_memory_resource>;
using HostBuffer       = ac::buffer<UserDatatype, ac::mr::host_memory_resource>;
using PinnedHostBuffer = ac::buffer<UserDatatype, ac::mr::pinned_host_memory_resource>;

// #include "ndbuffer.h"
// using DeviceNdBuffer = ac::ndbuffer<UserDatatype, ac::mr::device_memory_resource>;
// using HostNdBuffer = ac::ndbuffer<UserDatatype, ac::mr::host_memory_resource>;
// using PinnedHostNdBuffer = ac::ndbuffer<UserDatatype, ac::mr::pinned_host_memory_resource>;
