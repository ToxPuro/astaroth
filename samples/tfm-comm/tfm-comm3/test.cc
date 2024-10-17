#include <cstdlib>
#include <iostream>

#include "comm.h"

// and all this could be done with C++ templates as
template <typename T>
T*
nalloc(const size_t count)
{
    return (T*)calloc(count, sizeof(T));
}

template <typename T>
void
ndealloc(T** ptr)
{
    free(*ptr);
    *ptr = NULL;
}

int
main(void)
{
    int errcount = 0;
    errcount += acCommTest();

    size_t* arr = nalloc<size_t>(10);
    ndealloc(&arr);

    if (errcount == 0)
        std::cout << "---C++ test success---" << std::endl;
    else
        std::cout << "---C++ test failed: " << errcount << " errors found---" << std::endl;
    return errcount == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
