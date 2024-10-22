#pragma once

#include "shape.h"

// struct Segment {
//     Shape dims;    // Dimensions of the parent body
//     Shape subdims; // Dimensions of the segment
//     Index offset;  // Offset of the segment

//     Segment(const Shape& dims, const Shape& subdims, const Index& offset)
//         : dims(dims), subdims(subdims), offset(offset)
//     {
//     }
// };

// __host__ std::ostream&
// operator<<(std::ostream& os, const Segment& obj)
// {
//     os << "{\n";
//     os << "    dims: " << obj.dims << "," << std::endl;
//     os << "    subdims: " << obj.subdims << "," << std::endl;
//     os << "    offset: " << obj.offset << std::endl;
//     os << "}";
//     return os;
// }
