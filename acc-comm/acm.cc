#include "acm.h"

#include <exception>

#include "mpi_utils.h"

template <size_t N>
static ac::shape<N>
make_shape(const size_t ndims, const uint64_t* data)
{
    ERRCHK(ndims == N);
    ac::shape<N> shape{};
    std::copy_n(data, ndims, shape.begin());
    return shape;
}

ACM_Errorcode
ACM_MPI_Init_funneled(void)
{
    try {
        init_mpi_funneled();
        return ACM_ERRORCODE_SUCCESS;
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return ACM_ERRORCODE_GENERIC_FAILURE;
    }
}

ACM_Errorcode
ACM_MPI_Abort(void)
{
    MPI_Abort(MPI_COMM_WORLD, -1);
    return ACM_ERRORCODE_SUCCESS;
}

ACM_Errorcode
ACM_MPI_Finalize(void)
{
    try {
        finalize_mpi();
        return ACM_ERRORCODE_SUCCESS;
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return ACM_ERRORCODE_GENERIC_FAILURE;
    }
}

ACM_Errorcode
ACM_MPI_Cart_comm_create(const MPI_Comm parent_comm, const size_t ndims, const uint64_t* global_nn,
                         MPI_Comm* cart_comm)
{
    try {
        switch (ndims) {
        case 1:
            *cart_comm = cart_comm_create(parent_comm, make_shape<1>(ndims, global_nn));
            return ACM_ERRORCODE_SUCCESS;
        case 2:
            *cart_comm = cart_comm_create(parent_comm, make_shape<2>(ndims, global_nn));
            return ACM_ERRORCODE_SUCCESS;
        case 3:
            *cart_comm = cart_comm_create(parent_comm, make_shape<3>(ndims, global_nn));
            return ACM_ERRORCODE_SUCCESS;
        case 4:
            *cart_comm = cart_comm_create(parent_comm, make_shape<4>(ndims, global_nn));
            return ACM_ERRORCODE_SUCCESS;
        default:
            return ACM_ERRORCODE_UNSUPPORTED_NDIMS;
        }
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return ACM_ERRORCODE_GENERIC_FAILURE;
    }
}

ACM_Errorcode
ACM_MPI_Cart_comm_destroy(MPI_Comm* cart_comm)
{
    try {
        cart_comm_destroy(*cart_comm);
        return ACM_ERRORCODE_SUCCESS;
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return ACM_ERRORCODE_GENERIC_FAILURE;
    }
}

ACM_Errorcode
ACM_Get_decomposition(const MPI_Comm cart_comm, const size_t ndims, uint64_t* decomp_out)
{
    try {
        switch (ndims) {
        case 1: {
            const auto decomp = get_decomposition<1>(cart_comm);
            std::copy(decomp.begin(), decomp.end(), decomp_out);
            return ACM_ERRORCODE_SUCCESS;
        }
        case 2: {
            const auto decomp = get_decomposition<2>(cart_comm);
            std::copy(decomp.begin(), decomp.end(), decomp_out);
            return ACM_ERRORCODE_SUCCESS;
        }
        case 3: {
            const auto decomp = get_decomposition<3>(cart_comm);
            std::copy(decomp.begin(), decomp.end(), decomp_out);
            return ACM_ERRORCODE_SUCCESS;
        }
        default:
            return ACM_ERRORCODE_UNSUPPORTED_NDIMS;
        }
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return ACM_ERRORCODE_GENERIC_FAILURE;
    }
}

ACM_Errorcode
ACM_Get_coords(const MPI_Comm cart_comm, const size_t ndims, uint64_t* coords_out)
{
    try {
        switch (ndims) {
        case 1: {
            const auto coords = get_coords<1>(cart_comm);
            std::copy(coords.begin(), coords.end(), coords_out);
            return ACM_ERRORCODE_SUCCESS;
        }
        case 2: {
            const auto coords = get_coords<2>(cart_comm);
            std::copy(coords.begin(), coords.end(), coords_out);
            return ACM_ERRORCODE_SUCCESS;
        }
        case 3: {
            const auto coords = get_coords<3>(cart_comm);
            std::copy(coords.begin(), coords.end(), coords_out);
            return ACM_ERRORCODE_SUCCESS;
        }
        default:
            return ACM_ERRORCODE_UNSUPPORTED_NDIMS;
        }
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return ACM_ERRORCODE_GENERIC_FAILURE;
    }
}

ACM_Errorcode
ACM_Get_local_nn(const MPI_Comm cart_comm, const size_t ndims, const uint64_t* global_nn_in,
                 uint64_t* local_nn_out)
{
    try {
        switch (ndims) {
        case 1: {
            constexpr size_t NDIMS = 1;
            const auto global_nn{make_shape<NDIMS>(ndims, global_nn_in)};
            const auto decomp   = get_decomposition<NDIMS>(cart_comm);
            const auto local_nn = global_nn / decomp;
            std::copy(local_nn.begin(), local_nn.end(), local_nn_out);
            return ACM_ERRORCODE_SUCCESS;
        }
        case 2: {
            constexpr size_t NDIMS = 2;
            const auto global_nn{make_shape<NDIMS>(ndims, global_nn_in)};
            const auto decomp   = get_decomposition<NDIMS>(cart_comm);
            const auto local_nn = global_nn / decomp;
            std::copy(local_nn.begin(), local_nn.end(), local_nn_out);
            return ACM_ERRORCODE_SUCCESS;
        }
        case 3: {
            constexpr size_t NDIMS = 3;
            const auto global_nn{make_shape<NDIMS>(ndims, global_nn_in)};
            const auto decomp   = get_decomposition<NDIMS>(cart_comm);
            const auto local_nn = global_nn / decomp;
            std::copy(local_nn.begin(), local_nn.end(), local_nn_out);
            return ACM_ERRORCODE_SUCCESS;
        }
        default:
            return ACM_ERRORCODE_UNSUPPORTED_NDIMS;
        }
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return ACM_ERRORCODE_GENERIC_FAILURE;
    }
}

ACM_Errorcode
ACM_Get_global_nn_offset(const MPI_Comm cart_comm, const size_t ndims, const uint64_t* global_nn_in,
                         uint64_t* global_nn_offset_out)
{
    try {
        switch (ndims) {
        case 1: {
            constexpr size_t NDIMS = 1;
            const auto global_nn{make_shape<NDIMS>(ndims, global_nn_in)};
            const auto decomp{get_decomposition<NDIMS>(cart_comm)};
            const auto local_nn{global_nn / decomp};
            const auto coords{get_coords<NDIMS>(cart_comm)};
            const auto global_nn_offset{coords * local_nn};
            std::copy(global_nn_offset.begin(), global_nn_offset.end(), global_nn_offset_out);
            return ACM_ERRORCODE_SUCCESS;
        }
        case 2: {
            constexpr size_t NDIMS = 2;
            const auto global_nn{make_shape<NDIMS>(ndims, global_nn_in)};
            const auto decomp{get_decomposition<NDIMS>(cart_comm)};
            const auto local_nn{global_nn / decomp};
            const auto coords{get_coords<NDIMS>(cart_comm)};
            const auto global_nn_offset{coords * local_nn};
            std::copy(global_nn_offset.begin(), global_nn_offset.end(), global_nn_offset_out);
            return ACM_ERRORCODE_SUCCESS;
        }
        case 3: {
            constexpr size_t NDIMS = 3;
            const auto global_nn{make_shape<NDIMS>(ndims, global_nn_in)};
            const auto decomp{get_decomposition<NDIMS>(cart_comm)};
            const auto local_nn{global_nn / decomp};
            const auto coords{get_coords<NDIMS>(cart_comm)};
            const auto global_nn_offset{coords * local_nn};
            std::copy(global_nn_offset.begin(), global_nn_offset.end(), global_nn_offset_out);
            return ACM_ERRORCODE_SUCCESS;
        }
        default:
            return ACM_ERRORCODE_UNSUPPORTED_NDIMS;
        }
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return ACM_ERRORCODE_GENERIC_FAILURE;
    }
}
