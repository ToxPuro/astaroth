#include "ndvector.h"

void
test_ndvector(void)
{
    using Shape = ac::shape<2>;
    ac::ndvector<double, 2> ndvec(Shape{4, 4});
    fill<double, 2>(1, Shape{2, 2}, Shape{1, 1}, ndvec);
    ndvec.display();
}
