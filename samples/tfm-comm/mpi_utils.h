#pragma once
#include <stddef.h>

/** Translates an array in Astaroth (columns first) format to MPI_ORDER_C format (rows first) */
void to_mpi_format(const size_t ndims, const size_t* dims, int* mpi_dims);

/** Translates an array in MPI (rows first) format to Astaroth format (columns first) */
void to_astaroth_format(const size_t ndims, const int* mpi_dims, size_t* dims);

int get_tag(void);

void test_mpi_utils(void);
