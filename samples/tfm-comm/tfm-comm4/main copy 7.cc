// #include "comm.h"
// #include "segment.h"
// #include "print.h"
// #include "vecn.h"

#include <iostream>
#include <stdexcept>

struct A {

    A() { std::cerr << "A created" << std::endl; }
    void print() { std::cerr << "A printed" << std::endl; }
    ~A() { std::cerr << "A destructed" << std::endl; }
};

void
h()
{
    std::cerr << "H called" << std::endl;
    throw std::runtime_error("Error message");
}
void
g()
{
    h();
}
void
f()
{
    g();
}

int
main(void)
{

    A a;
    f();
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
