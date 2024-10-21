#include <iostream>
#include <stdexcept>

#include "errchk.h"
#include "type_conversion.h"

void
h()
{
    ERRCHK(1 == 2);
}
void
g()
{
    HANDLE(h());
}
void
f()
{
    HANDLE(g());
}

int
main(void)
{
    // HANDLE(f());
    HANDLE(as<size_t>(-1));

    return EXIT_SUCCESS;
}
