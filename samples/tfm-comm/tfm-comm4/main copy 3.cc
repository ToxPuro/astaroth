// #include "comm.h"
// #include "segment.h"
// #include "print.h"
// #include "vecn.h"

#define BOOST_STACKTRACE_GNU_SOURCE_NOT_REQUIRED
#define BOOST_STACKTRACE_USE_ADDR2LINE
#include <boost/stacktrace.hpp>
#include <iostream>

void
f()
{
    if (1 != 2) {
        std::cerr << "ERROR in f()" << std::endl;
        std::cerr << "Stacktrace:\n" << boost::stacktrace::stacktrace();
        throw std::overflow_error("overflow");
    }
}

void
g()
{
    f();
}

void
h()
{
    g();
}

static void
exit_func()
{
    std::cerr << "EXIT FUNC CALLED\n";
}

int
main(void)
{
    try {
        h();
    }
    catch (const std::exception& e) {
        exit_func();
        throw;
    }
    // acCommInit();

    // Shape global_nn = {4, 4};
    // Shape local_nn(global_nn.count);
    // Index global_nn_offset(global_nn.count);
    // PRINTD(global_nn);
    // PRINTD(local_nn);
    // PRINTD(global_nn_offset);

    // VecN<int, 3> a = {-1, 2, 3};
    // VecN<uint64_t, 3> b(a);
    // PRINTD(a);
    // PRINTD(b);

    // acCommSetup(global_nn.count, global_nn.data, local_nn.data, global_nn_offset.data);
    // acCommPrint();
    // acCommQuit();
    return 0;
}
