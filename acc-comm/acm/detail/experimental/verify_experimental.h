#pragma once

#include "acm/detail/ndbuffer.h"
#include "acm/detail/pointer.h"

namespace ac::verify {

/** Initializes the local mesh to global iota */
template <typename T, typename Allocator>
void to_global_iota(const ac::shape& global_nn, const ac::shape& global_nn_offset,
                    const ac::shape& local_mm, const ac::shape& local_nn, const ac::index& local_rr,
                    ac::mr::pointer<T, Allocator> ptr);

/** Fills a region in the local mesh */
template <typename T, typename Allocator>
void fill(const T& fill_value, const ac::shape& dims, const ac::index& offset,
          ac::mr::pointer<T, Allocator> ptr);

/**
 * Verify that each cell in the local mesh corresponds to its global index.
 * Throws an exception in the case of a failure.
 */
template <typename T, typename Allocator>
void verify_global_iota(const ac::shape& global_nn, const ac::shape& global_nn_offset,
                        const ac::shape& local_mm, const ac::shape& local_nn,
                        const ac::index& local_rr, const ac::mr::pointer<T, Allocator>& ptr);

} // namespace ac::verify
