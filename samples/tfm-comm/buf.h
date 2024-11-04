#pragma once
#include <cstddef>

#include "mem.h"

template <typename T> struct GenericBuffer {
    size_t count;
    mem_t<T> data;

    GenericBuffer(const size_t in_count, mem_t<T> in_data)
        : count(in_count), data(std::move(in_data))
    {
    }
};

template <typename T> struct HostBuffer : public GenericBuffer<T> {
    HostBuffer(const size_t in_count, mem_t<T> in_data)
        : GenericBuffer<T>(in_count, std::move(in_data))
    {
    }

    // Enable the subscript[] operator
    __host__ __device__ T& operator[](size_t i) { return this->data.get()[i]; }
    __host__ __device__ const T& operator[](size_t i) const { return this->data.get()[i]; }
};

template <typename T> struct HostBufferDefault : public HostBuffer<T> {
    HostBufferDefault(const size_t in_count)
        : HostBuffer<T>(in_count, host::make_unique<T>(in_count))
    {
    }
};

#define DEVICE_ENABLED
#if defined(DEVICE_ENABLED)
template <typename T> struct HostBufferPinned : public HostBuffer<T> {
    HostBufferPinned(const size_t in_count)
        : HostBuffer<T>(in_count, host::pinned::make_unique<T>(in_count))
    {
    }
};

template <typename T> struct HostBufferPinnedWriteCombined : public HostBuffer<T> {
    HostBufferPinnedWriteCombined(const size_t in_count)
        : HostBuffer<T>(in_count, host::pinned::wc::make_unique<T>(in_count))
    {
    }
};

template <typename T> struct DeviceBuffer : public GenericBuffer<T> {
    DeviceBuffer(const size_t in_count, mem_t<T> in_data)
        : GenericBuffer<T>(in_count, std::move(in_data))
    {
    }
};

template <typename T> struct DeviceBufferDefault : public DeviceBuffer<T> {
    DeviceBufferDefault(const size_t in_count)
        : DeviceBuffer<T>(in_count, device::make_unique<T>(in_count))
    {
    }
};

template <typename T>
void
fill(HostBuffer<T>& buffer)
{
    for (size_t i = 0; i < buffer.count; ++i)
        buffer[i] = 1;
}

template <typename T>
void
print(HostBuffer<T>& buffer)
{
    for (size_t i = 0; i < buffer.count; ++i)
        std::cout << buffer[i] << " ";
}

template <typename T>
void
migrate(const HostBuffer<T>& in, HostBuffer<T>& out)
{
    std::cout << "htoh" << std::endl;
}

template <typename T>
void
migrate(const HostBuffer<T>& in, DeviceBuffer<T>& out)
{
    std::cout << "htod" << std::endl;
}

template <typename T>
void
migrate(const DeviceBuffer<T>& in, HostBuffer<T>& out)
{
    std::cout << "dtoh" << std::endl;
}

template <typename T>
void
migrate(const DeviceBuffer<T>& in, DeviceBuffer<T>& out)
{
    std::cout << "dtod" << std::endl;
}
#else
template <typename T> using DeviceBuffer                  = HostBufferDefault<T>;
template <typename T> using DeviceBufferDefault           = HostBufferDefault<T>;
template <typename T> using HostBufferPinned              = HostBufferDefault<T>;
template <typename T> using HostBufferPinnedWriteCombined = HostBufferDefault<T>;

template <typename T>
void
migrate(const HostBuffer<T>& in, HostBuffer<T>& out)
{
    std::cout << "standard htoh" << std::endl;
}
#endif

// template <typename T> class HostBuffer {
//     // size_t count;
//     // mem_t<T> data;
// };
// template <typename T> struct DeviceBuffer {
//     // size_t count;
//     // mem_t<T> data;
// };

// template <typename T> struct HostBufferDefault : public HostBuffer<T> {
//     size_t count;
//     mem_t<T> data;

//     HostBufferDefault(const size_t in_count)
//         : count{in_count}, data{host::make_unique<T>(in_count)}
//     {
//     }
// };

// template <typename T> struct HostBufferPinned : public HostBuffer<T> {
//   public:
//     size_t count;
//     mem_t<T> data;

//     HostBufferPinned(const size_t in_count)
//         : count(in_count), data{host::pinned::make_unique<T>(in_count)}
//     {
//     }
// };

// template <typename T> struct HostBufferPinnedWriteCombined : public HostBuffer<T> {
//   public:
//     size_t count;
//     mem_t<T> data;

//     HostBufferPinnedWriteCombined(const size_t in_count)
//         : count(in_count), data{host::pinned::wc::make_unique<T>(in_count)}
//     {
//     }
// };

// template <typename T> struct DeviceBufferDefault : public DeviceBuffer<T> {
//   public:
//     size_t count;
//     mem_t<T> data;

//     DeviceBufferDefault(const size_t in_count)
//         : count(in_count), data{device::make_unique<T>(in_count)}
//     {
//     }
// };
