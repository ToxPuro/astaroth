#include "segment.h"

__host__ std::ostream&
operator<<(std::ostream& os, const Segment& obj)
{
    os << "{";
    os << "dims: " << obj.dims << ", ";
    os << "offset: " << obj.offset;
    os << "}";
    return os;
}
