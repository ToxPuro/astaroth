#include "acm/acm.h"

#include <exception>

#include "acm/detail/ntuple.h"
#include "mpi_utils.h"

#include "allocator.h"
#include "halo_exchange_packed.h"
#include "io.h"

ACM_Errorcode
ACM_MPI_Init_funneled(void)
{
    try {
        ac::mpi::init_funneled();
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
    ERRCHK_MPI_API(MPI_Abort(MPI_COMM_WORLD, -1));
    return ACM_ERRORCODE_SUCCESS;
}

ACM_Errorcode
ACM_MPI_Finalize(void)
{
    try {
        ac::mpi::finalize();
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
        *cart_comm = ac::mpi::cart_comm_create(parent_comm,
                                               ac::make_ntuple_from_ptr(ndims, global_nn));
        return ACM_ERRORCODE_SUCCESS;
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
        ac::mpi::cart_comm_destroy(cart_comm);
        return ACM_ERRORCODE_SUCCESS;
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return ACM_ERRORCODE_GENERIC_FAILURE;
    }
}

ACM_Errorcode
ACM_Get_ndims(const MPI_Comm cart_comm, int* ndims)
{
    try {
        *ndims = ac::mpi::get_ndims(cart_comm);
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
        ERRCHK(ac::mpi::get_ndims(cart_comm) == as<int>(ndims));
        const auto decomp = ac::mpi::get_decomposition(cart_comm);
        std::copy(decomp.begin(), decomp.end(), decomp_out);
        return ACM_ERRORCODE_SUCCESS;
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
        ERRCHK(ac::mpi::get_ndims(cart_comm) == as<int>(ndims));
        const auto coords = ac::mpi::get_coords(cart_comm);
        std::copy(coords.begin(), coords.end(), coords_out);
        return ACM_ERRORCODE_SUCCESS;
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
        const auto global_nn{ac::make_ntuple_from_ptr(ndims, global_nn_in)};
        const auto decomp   = ac::mpi::get_decomposition(cart_comm);
        const auto local_nn = global_nn / decomp;
        std::copy(local_nn.begin(), local_nn.end(), local_nn_out);
        return ACM_ERRORCODE_SUCCESS;
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
        const auto global_nn{ac::make_ntuple_from_ptr(ndims, global_nn_in)};
        const auto decomp{ac::mpi::get_decomposition(cart_comm)};
        const auto local_nn{global_nn / decomp};
        const auto coords{ac::mpi::get_coords(cart_comm)};
        const auto global_nn_offset{coords * local_nn};
        std::copy(global_nn_offset.begin(), global_nn_offset.end(), global_nn_offset_out);
        return ACM_ERRORCODE_SUCCESS;
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return ACM_ERRORCODE_GENERIC_FAILURE;
    }
}

ACM_Errorcode
ACM_IO_Read_collective(const MPI_Comm cart_comm, const size_t ndims, const uint64_t* file_dims,
                       const uint64_t* file_offset, const uint64_t* mesh_dims,
                       const uint64_t* mesh_subdims, const uint64_t* mesh_offset, const char* path,
                       double* data)
{
    try {
        ac::mpi::read_collective(cart_comm,
                                 ac::mpi::get_dtype<double>(),
                                 ac::make_ntuple_from_ptr(ndims, file_dims),
                                 ac::make_ntuple_from_ptr(ndims, file_offset),
                                 ac::make_ntuple_from_ptr(ndims, mesh_dims),
                                 ac::make_ntuple_from_ptr(ndims, mesh_subdims),
                                 ac::make_ntuple_from_ptr(ndims, mesh_offset),
                                 std::string(path),
                                 data);
        return ACM_ERRORCODE_SUCCESS;
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return ACM_ERRORCODE_GENERIC_FAILURE;
    }
}

ACM_Errorcode
ACM_IO_Write_collective(const MPI_Comm parent_comm, const size_t ndims, const uint64_t* file_dims,
                        const uint64_t* file_offset, const uint64_t* mesh_dims,
                        const uint64_t* mesh_subdims, const uint64_t* mesh_offset,
                        const double* data, const char* path)
{
    try {
        ac::mpi::write_collective(parent_comm,
                                  ac::mpi::get_dtype<double>(),
                                  ac::make_ntuple_from_ptr(ndims, file_dims),
                                  ac::make_ntuple_from_ptr(ndims, file_offset),
                                  ac::make_ntuple_from_ptr(ndims, mesh_dims),
                                  ac::make_ntuple_from_ptr(ndims, mesh_subdims),
                                  ac::make_ntuple_from_ptr(ndims, mesh_offset),
                                  data,
                                  std::string(path));
        return ACM_ERRORCODE_SUCCESS;
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return ACM_ERRORCODE_GENERIC_FAILURE;
    }
}

// ACM_Errorcode
// ACM_IO_Task_create(const size_t ndims, const uint64_t* file_dims, const uint64_t*
// file_offset,
//                    const uint64_t* mesh_dims, const uint64_t* mesh_subdims,
//                    const uint64_t* mesh_offset, ACM_IO_Task* task)
// {
//     return ACM_ERRORCODE_NOT_IMPLEMENTED;
// }

// ACM_Errorcode
// ACM_IO_Task_destroy(ACM_IO_Task* task)
// {
//     return ACM_ERRORCODE_NOT_IMPLEMENTED;
// }

// ACM_Errorcode
// ACM_IO_Launch_write_collective(const MPI_Comm parent_comm, const double* input, const char* path,
//                                ACM_IO_Task* task)
// {
//     return ACM_ERRORCODE_NOT_IMPLEMENTED;
// }

// ACM_Errorcode
// ACM_IO_Wait_write_collective(ACM_IO_Task* task)
// {
//     return ACM_ERRORCODE_NOT_IMPLEMENTED;
// }

// ACM_Errorcode
// ACM_COMM_Launch_halo_exchange(const MPI_Comm parent_comm, const size_t ndims,
//                               const uint64_t* local_mm, const uint64_t* local_nn,
//                               const uint64_t* rr, const double* send_data, double* recv_data,
//                               size_t* recv_req_count, MPI_Request* recv_reqs)
// {
//     return ACM_ERRORCODE_NOT_IMPLEMENTED;
// }

// ACM_Errorcode
// ACM_COMM_Halo_Exchange_Task_create(const size_t ndims, const uint64_t* local_mm,
//                                    const uint64_t* local_nn, const uint64_t* local_rr,
//                                    const size_t n_aggregate_buffers,
//                                    ACM_COMM_Halo_Exchange_Task* task)
// {
//     return ACM_ERRORCODE_NOT_IMPLEMENTED;
// }

// ACM_Errorcode
// ACM_COMM_Halo_Exchange_Task_destroy(ACM_COMM_Halo_Exchange_Task* task)
// {
//     return ACM_ERRORCODE_NOT_IMPLEMENTED;
// }

// ACM_Errorcode
// ACM_COMM_Halo_Exchange_Task_launch(const MPI_Comm parent_comm, const size_t ninputs,
//                                    const double* inputs[], ACM_COMM_Halo_Exchange_Task* task)
// {
//     return ACM_ERRORCODE_NOT_IMPLEMENTED;
// }

// ACM_Errorcode
// ACM_COMM_Halo_Exchange_Task_wait(const size_t noutputs, double* outputs,
//                                  ACM_COMM_Halo_Exchange_Task* task)
// {
//     return ACM_ERRORCODE_NOT_IMPLEMENTED;
// }
