// #include "comm.h"
// #include "segment.h"
// #include "print.h"
// #include "vecn.h"

#include <iostream>

#define TRY_CATCH(expr)                                                                            \
    try {                                                                                          \
        (expr);                                                                                    \
    }                                                                                              \
    catch (const std::exception& e) {                                                              \
        std::cerr << "Caught exception " << e.what() << " in " << __func__ << std::endl;           \
        throw;                                                                                     \
    }

int
some_function()
{
    return -1;
}

void
f()
{
    int arr[5];
    if (some_function() != 0) {
        throw std::runtime_error("Mpi error");
    }
    if (1 != 2) {
        std::cerr << "ERROR in f()" << std::endl;
        throw std::overflow_error("overflow");
    }
}

void
g()
{
    TRY_CATCH(f());
}

void
h()
{
    try {
        g();
    }
    catch (const std::exception& e) {
        std::cerr << "Caught exception " << e.what() << " in " << __func__ << std::endl;
        throw;
    }
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
        std::cerr << "Caught exception " << e.what() << " in " << __func__ << std::endl;
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
