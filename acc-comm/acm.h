#pragma once
#include <stdint.h>

#include <mpi.h>

#include "acm_error.h"

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

#ifdef __cplusplus
}
#endif
