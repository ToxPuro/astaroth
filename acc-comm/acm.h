#pragma once
#include <stdint.h>

#include <mpi.h>

#include "acm_error.h"

typedef void* ACM_IO_Task;
typedef void* ACM_COMM_Halo_Exchange_Task;
typedef void* ACM_COMM_Device_Host_Transfer_Task;

#ifdef __cplusplus
extern "C" {
#endif

ACM_Errorcode ACM_MPI_Init_funneled(void);

ACM_Errorcode ACM_MPI_Abort(void);

ACM_Errorcode ACM_MPI_Finalize(void);

ACM_Errorcode ACM_MPI_Cart_comm_create(const MPI_Comm parent_comm, const size_t ndims,
                                       const uint64_t* global_nn, MPI_Comm* cart_comm);

ACM_Errorcode ACM_MPI_Cart_comm_destroy(MPI_Comm* cart_comm);

ACM_Errorcode ACM_Get_decomposition(const MPI_Comm cart_comm, const size_t ndims,
                                    uint64_t* decomp_out);

ACM_Errorcode ACM_Get_coords(const MPI_Comm cart_comm, const size_t ndims, uint64_t* coords_out);

ACM_Errorcode ACM_Get_local_nn(const MPI_Comm cart_comm, const size_t ndims,
                               const uint64_t* global_nn_in, uint64_t* local_nn_out);

ACM_Errorcode ACM_Get_global_nn_offset(const MPI_Comm cart_comm, const size_t ndims,
                                       const uint64_t* global_nn_in,
                                       uint64_t* global_nn_offset_out);

ACM_Errorcode ACM_IO_Read_collective(const MPI_Comm parent_comm, const size_t ndims,
                                     const uint64_t* in_file_dims, const uint64_t* in_file_offset,
                                     const uint64_t* in_mesh_dims, const uint64_t* in_mesh_subdims,
                                     const uint64_t* in_mesh_offset, const char* path,
                                     double* data);

ACM_Errorcode ACM_IO_Write_collective(const MPI_Comm parent_comm, const size_t ndims,
                                      const uint64_t* in_file_dims, const uint64_t* in_file_offset,
                                      const uint64_t* in_mesh_dims, const uint64_t* in_mesh_subdims,
                                      const uint64_t* in_mesh_offset, const double* data,
                                      const char* path);

ACM_Errorcode ACM_IO_Task_create(const size_t ndims, const uint64_t* in_file_dims,
                                 const uint64_t* in_file_offset, const uint64_t* in_mesh_dims,
                                 const uint64_t* in_mesh_subdims, const uint64_t* in_mesh_offset,
                                 ACM_IO_Task* task);

ACM_Errorcode ACM_IO_Task_destroy(ACM_IO_Task* task);

ACM_Errorcode ACM_IO_Launch_write_collective(const MPI_Comm parent_comm, const double* input,
                                             const char* path, ACM_IO_Task* task);

ACM_Errorcode ACM_IO_Wait_write_collective(ACM_IO_Task* task);

ACM_Errorcode ACM_COMM_Launch_halo_exchange(const MPI_Comm parent_comm, const size_t ndims,
                                            const uint64_t* local_mm, const uint64_t* local_nn,
                                            const uint64_t* rr, const double* send_data,
                                            double* recv_data, size_t* recv_req_count,
                                            MPI_Request* recv_reqs);

ACM_Errorcode ACM_COMM_Halo_Exchange_Task_create(const size_t ndims, const uint64_t* local_mm,
                                                 const uint64_t* local_nn, const uint64_t* local_rr,
                                                 const size_t n_aggregate_buffers,
                                                 ACM_COMM_Halo_Exchange_Task* task);

ACM_Errorcode ACM_COMM_Halo_Exchange_Task_destroy(ACM_COMM_Halo_Exchange_Task* task);

ACM_Errorcode ACM_COMM_Halo_Exchange_Task_launch(const MPI_Comm parent_comm, const size_t ninputs,
                                                 const double* inputs[],
                                                 ACM_COMM_Halo_Exchange_Task* task);

ACM_Errorcode ACM_COMM_Halo_Exchange_Task_wait(const size_t noutputs, double* outputs,
                                               ACM_COMM_Halo_Exchange_Task* task);

#ifdef __cplusplus
}
#endif
