#include "datatypes.h"

#include "ntuple.h"

Dims
make_dims(const size_t count, const UserDatatype& fill_value)
{
    return ac::make_ntuple(count, fill_value);
}
