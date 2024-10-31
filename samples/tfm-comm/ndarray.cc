#include "ndarray.h"

void
test_ndarray(void)
{
    const Shape shape{64, 32, 16};
    NdArray<double> arr(shape);
    ERRCHK(arr.buffer.data);
}
