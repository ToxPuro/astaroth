#pragma once
#include <stddef.h>

enum DeviceBufferType {
    DEVICE_BUFFER_HOST,
    DEVICE_BUFFER_HOST_PINNED,
    DEVICE_BUFFER_HOST_PINNED_WRITE_COMBINED,
    DEVICE_BUFFER_DEVICE,
};

template <typename T> struct DeviceBuffer {
    const size_t count;
    const DeviceBufferType type;
    T* data;

    DeviceBuffer(const size_t count, const DeviceBufferType type = DEVICE_BUFFER_HOST);

    ~DeviceBuffer();

    // Delete all other types of constructors
    DeviceBuffer(const DeviceBuffer&)            = delete; // Copy constructor
    DeviceBuffer& operator=(const DeviceBuffer&) = delete; // Copy assignment operator
    DeviceBuffer(DeviceBuffer&&)                 = delete; // Move constructor
    DeviceBuffer& operator=(DeviceBuffer&&)      = delete; // Move assignment operator
};
