#pragma once

#include "datatypes.h"

template <size_t N> struct Segment {
    Shape<N> dims{};   // Dimensions of the segment
    Index<N> offset{}; // Offset of the segment

    // Constructors
    explicit Segment(const Shape<N>& in_dims)
        : dims{in_dims}
    {
    }
    Segment(const Shape<N>& in_dims, const Index<N>& in_offset)
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
