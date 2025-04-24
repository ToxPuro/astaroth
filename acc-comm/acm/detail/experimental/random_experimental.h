#pragma once
#include "acm/detail/pointer.h"


namespace acm::experimental {

void randomize(ac::mr::host_pointer<double> ptr);
void randomize(ac::mr::device_pointer<double> ptr);

}


