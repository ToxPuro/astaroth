#pragma once

#include <iostream>

#include "errchk.h"

template <typename T> struct Buffer {
    size_t count;
    T* data;

    // Constructor
    Buffer(const size_t count)
        : count(count), data(new T[count]())
    {
    }

    // Enable the subscript operator
    // __host__ __device__ T& operator[](size_t i) { return data[i]; }
    // __host__ __device__ const T& operator[](size_t i) const { return data[i]; }

    Buffer(const size_t count, const T& fill_value)
        : count(count), data(new T[count])
    {
        for (size_t i = 0; i < count; ++i)
            data[i] = fill_value;
    }

    // Destructor
    ~Buffer() { delete[] data; }

    // Delete all other types of constructors
    Buffer(const Buffer&)            = delete; // Copy constructor
    Buffer& operator=(const Buffer&) = delete; // Copy assignment operator
    Buffer(Buffer&&)                 = delete; // Move constructor
    Buffer& operator=(Buffer&&)      = delete; // Move assignment operator

    // // Copy constructor
    // Buffer(const Buffer& other)
    //     : count(other.count), data(new T[other.count])
    // {
    //     std::copy(other.data, other.data + count, data);
    // }

    // // Copy assignment operator
    // Buffer& operator=(const Buffer& other)
    // {
    //     // Self-assignment
    //     if (this == &other)
    //         return *this;

    //     T* new_data = new T[other.count];
    //     std::copy(other.data, other.data + other.count, new_data);
    //     delete[] data;
    //     data  = new_data;
    //     count = other.count;
    //     return *this;
    // }

    // // Move constructor
    // Buffer(Buffer&& other) noexcept
    //     : count(other.count), data(other.data)
    // {
    //     other.count = 0;
    //     other.data  = nullptr;
    // }

    // // Move assignment operator
    // Buffer& operator=(Buffer&& other) noexcept
    // {
    //     // Self-assignment
    //     if (this == &other)
    //         return *this;

    //     delete[] data;
    //     data        = other.data;
    //     count       = other.count;
    //     other.data  = nullptr;
    //     other.count = 0;
    //     return *this;
    // }
    void fill(const T& value)
    {
        for (size_t i = 0; i < count; ++i)
            data[i] = value;
    }
    void fill_arange(const T& min, const T& max)
    {
        ERRCHK(min < max);
        ERRCHK(max <= count);
        for (size_t i = 0; i < max - min; ++i)
            data[i] = min + i;
    }
};

template <typename T>
std::ostream&
operator<<(std::ostream& os, const Buffer<T>& buf)
{
    os << "{";
    for (size_t i = 0; i < buf.count; ++i)
        os << buf.data[i] << (i + 1 < buf.count ? ", " : "}");
    return os;
}

/**
 * Other utility functions
 */
template <typename T>
Buffer<T>
ones(const size_t count)
{
    Buffer<T> buffer(count);
    for (size_t i = 0; i < count; ++i)
        buffer.data[i] = 1;
    return buffer;
}

template <typename T>
Buffer<T>
zeros(const size_t count)
{
    Buffer<T> buffer(count);
    for (size_t i = 0; i < count; ++i)
        buffer.data[i] = 1;
    return buffer;
}

template <typename T>
Buffer<T>
arange(const size_t min, const size_t max)
{
    ERRCHK(max > min);
    Buffer<T> buffer(max - min);
    for (size_t i = 0; i < buffer.count; ++i)
        buffer.data[i] = min + i;
    return buffer;
}
