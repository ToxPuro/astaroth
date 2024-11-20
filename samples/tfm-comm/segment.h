#pragma once

#include "datatypes.h"

template <size_t N> struct Segment {
    ac::shape<N> dims{};   // Dimensions of the segment
    ac::index<N> offset{}; // Offset of the segment

    // Constructors
    explicit Segment(const ac::shape<N>& in_dims)
        : dims{in_dims}
    {
    }
    Segment(const ac::shape<N>& in_dims, const ac::index<N>& in_offset)
        : dims{in_dims}, offset{in_offset}
    {
    }

    friend __host__ std::ostream& operator<<(std::ostream& os, const Segment& obj)
    {
        os << "{";
        os << "dims: " << obj.dims << ", ";
        os << "offset: " << obj.offset;
        os << "}";
        return os;
    }
};

template <size_t N> __host__ std::ostream& operator<<(std::ostream& os, const Segment<N>& obj);
