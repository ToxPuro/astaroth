#include <iostream>
#include <stdexcept>

#define ERRCHK_MPI_API(errcode)                                                                    \
    do {                                                                                           \
        if ((errcode) != 0)                                                                        \
            throw std::runtime_error("mpi api error");                                             \
    } while (0)

#define ERRCHK_CUDA_API(errcode)                                                                   \
    do {                                                                                           \
        if ((errcode) != 0)                                                                        \
            throw std::runtime_error("cuda api error");                                            \
    } while (0)

#define ERRCHK(expr)                                                                               \
    do {                                                                                           \
        if ((expr) != 0)                                                                           \
            throw std::runtime_error("general error");                                             \
    } while (0)

#define TRY_CATCH(expr)                                                                            \
    try {                                                                                          \
        expr;                                                                                      \
    }                                                                                              \
    catch (const std::exception& e) {                                                              \
        std::cerr << "Exception caught in w. expr " << #expr << " fn[" << __func__                 \
                  << "],         \
            line " << __LINE__                                                                     \
                  << std::endl;                                                                    \
        throw;                                                                                     \
    }

#define ERROR() (std::cerr << "Failure in function " << __func__ << std::endl)

int
mpi_func()
{
    ERROR();
    return -1;
}

int
cuda_func()
{
    ERROR();
    return -1;
}

void
f()
{
    ERROR();
    throw std::runtime_error("f failed");
}

void
internal_func()
{
    TRY_CATCH(f());
}

int
main(void)
{
    TRY_CATCH(ERRCHK_MPI_API(mpi_func()));
    TRY_CATCH(ERRCHK_CUDA_API(cuda_func()));
    TRY_CATCH(internal_func());
    return EXIT_SUCCESS;
}
