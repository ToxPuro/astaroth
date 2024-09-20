#pragma once
#include <stddef.h>

#include <mpi.h>

MPI_Comm create_rank_reordered_cart_comm(const MPI_Comm parent, const size_t ndims,
                                         const size_t* global_dims);
