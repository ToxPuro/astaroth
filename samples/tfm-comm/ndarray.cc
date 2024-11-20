#include "ndarray.h"

void
test_ndarray(void)
{
    const ac::shape<3> shape{64, 32, 16};
    NdArray<double, 3> arr(shape);
    ERRCHK(arr.buffer.data());
}
