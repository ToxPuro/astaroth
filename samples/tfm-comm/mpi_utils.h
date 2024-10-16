#pragma once
#include <stddef.h>

/** Translates an array in Astaroth (columns first) format to MPI_ORDER_C format (rows first) */
void to_mpi_format(const size_t ndims, const size_t* dims, int* mpi_dims);

/** Translates an array in MPI (rows first) format to Astaroth format (columns first) */
void to_astaroth_format(const size_t ndims, const int* mpi_dims, size_t* dims);

/** Allocates memory for an MPI array and copies the translated Astaroth array to it.
 * The memory allocated must be freed by the user when no longer needed.
 */
int* to_mpi_format_alloc(const size_t ndims, const size_t* arr);

/** Allocates memory for an Astaroth array and copies the translated MPI array to it.
 * The memory allocated must be freed by the user when no longer needed.
 */
size_t* to_astaroth_format_alloc(const size_t ndims, const int* mpi_arr);

/** Returns the next positive integer that can be used as a tag */
int get_tag(void);

void test_mpi_utils(void);
