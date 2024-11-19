#include <cstdlib>
#include <iostream>
#include <memory>

#include "errchk.h"
#include "errchk_cuda.h"

// #include "buffer.h"
// #include "buffer_exchange.h"

#include "decomp.h"

int
main(void)
{
    const auto lala = decompose<3>(Shape<3>{4,4,4}, 2);

    return EXIT_SUCCESS;
}
