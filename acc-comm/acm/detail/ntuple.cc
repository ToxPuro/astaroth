#include "ntuple.h"

namespace ac {

ac::index
make_index(const size_t count, const uint64_t& fill_value)
{
    return ac::make_ntuple(count, fill_value);
}

ac::shape
make_shape(const size_t count, const uint64_t& fill_value)
{
    return ac::make_ntuple(count, fill_value);
}

ac::direction
make_direction(const size_t count, const int64_t& fill_value)
{
    return ac::make_ntuple(count, fill_value);
}

} // namespace ac
