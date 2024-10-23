#include "packet.h"

template <typename T>
__host__ std::ostream&
operator<<(std::ostream& os, const Packet<T>& obj)
{
    os << "{\n";
    os << "    segment: " << obj.segment << "," << std::endl;
    os << "    buffer: " << obj.buffer << "," << std::endl;
    os << "    req: " << obj.req << std::endl;
    os << "}";
    return os;
}
