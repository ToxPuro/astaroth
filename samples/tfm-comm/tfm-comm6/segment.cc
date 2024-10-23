#include "segment.h"

__host__ std::ostream&
operator<<(std::ostream& os, const Segment& obj)
{
    os << "{\n";
    os << "    dims: " << obj.dims << "," << std::endl;
    // os << "    subdims: " << obj.subdims << "," << std::endl;
    os << "    offset: " << obj.offset << std::endl;
    os << "}";
    return os;
}
