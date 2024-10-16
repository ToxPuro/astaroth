#include <cstdlib>
#include <iostream>

#include "comm.h"

int
main(void)
{
    int errcount = 0;
    errcount += acCommTest();
    if (errcount == 0)
        std::cout << "---C++ test success---" << std::endl;
    else
        std::cout << "---C++ test failed: " << errcount << " errors found---" << std::endl;
    return errcount == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
