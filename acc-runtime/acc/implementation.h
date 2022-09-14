#pragma once
#include <stdlib.h>

#define ORIGINAL (0)
#define ORIGINAL_WITH_ILP (1)
#define EXPL_REG_VARS (2)
#define FULLY_EXPL_REG_VARS (3)
#define EXPL_REG_VARS_AND_CT_CONST_STENCILS (4)
#define FULLY_EXPL_REG_VARS_AND_PINGPONG_REGISTERS (5)
#define SMEM_AND_VECTORIZED_LOADS (6)
#define SMEM_AND_VECTORIZED_LOADS_PINGPONG (7)
#define SMEM_AND_VECTORIZED_LOADS_FULL (8)
#define SMEM_AND_VECTORIZED_LOADS_FULL_ASYNC (9)
#define SMEM_HIGH_OCCUPANCY (10)

#define IMPLEMENTATION (3)

#if IMPLEMENTATION == SMEM_AND_VECTORIZED_LOADS ||                             \
    IMPLEMENTATION == SMEM_AND_VECTORIZED_LOADS_FULL ||                        \
    IMPLEMENTATION == SMEM_AND_VECTORIZED_LOADS_FULL_ASYNC ||                  \
    IMPLEMENTATION == SMEM_AND_VECTORIZED_LOADS_PINGPONG

#define USE_VECTORIZED_LOADS (1)
const char* realtype   = "double";
const char* veclen_str = "2";
const size_t veclen    = 2;

#if IMPLEMENTATION == SMEM_AND_VECTORIZED_LOADS ||                             \
    IMPLEMENTATION == SMEM_AND_VECTORIZED_LOADS_PINGPONG
const size_t buffers = 2;
#elif IMPLEMENTATION == SMEM_AND_VECTORIZED_LOADS_FULL ||                      \
    IMPLEMENTATION == SMEM_AND_VECTORIZED_LOADS_FULL_ASYNC
const size_t buffers = NUM_FIELDS;
#endif

size_t
get_smem(const size_t x, const size_t y, const size_t z,
         const size_t stencil_order, const size_t bytes_per_elem)
{
  return buffers * (x + stencil_order) * (y + stencil_order) *
         (z + stencil_order) * bytes_per_elem;
}
#elif IMPLEMENTATION == SMEM_HIGH_OCCUPANCY
size_t
get_smem(const size_t x, const size_t y, const size_t z,
         const size_t stencil_order, const size_t bytes_per_elem)
{
  (void)y; // Unused
  (void)z; // Unused
  return bytes_per_elem * (x + stencil_order);
}
#else
size_t
get_smem(const size_t x, const size_t y, const size_t z,
         const size_t stencil_order, const size_t bytes_per_elem)
{
  (void)x;              // Unused
  (void)y;              // Unused
  (void)z;              // Unused
  (void)stencil_order;  // Unused
  (void)bytes_per_elem; // Unused
  return 0;
}
#endif
