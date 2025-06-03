#pragma once

#include "acm/detail/math_utils.h"
#include "acm/detail/ndbuffer.h"
#include "acm/detail/ntuple.h"
#include "acm/detail/pack.h"
#include "acm/detail/pointer.h"
#include "acm/detail/segment.h"

namespace ac {

/** Initializes the local mesh to global iota */
template <typename T>
void
to_global_iota(const ac::shape& global_nn, const ac::shape& global_nn_offset,
               const ac::shape& local_mm, const ac::shape& local_nn, const ac::index& local_rr,
               ac::mr::host_pointer<T> ptr, const T& initial_value = 0)
{
    (void)local_nn; // Unused
    for (uint64_t i{0}; i < ptr.size(); ++i) {
        const auto global_coords{
            (global_nn + global_nn_offset + ac::to_spatial(i, local_mm) - local_rr) % global_nn};
        const auto global_index{ac::to_linear(global_coords, global_nn)};
        ptr[i] = static_cast<T>(global_index) + initial_value;
    }
}

#if defined(ACM_DEVICE_ENABLED)
template <typename T>
void
to_global_iota(const ac::shape& global_nn, const ac::shape& global_nn_offset,
               const ac::shape& local_mm, const ac::shape& local_nn, const ac::index& local_rr,
               ac::mr::device_pointer<T> ptr, const T& initial_value = 0)
{
    ac::host_buffer<T> tmp{ptr.size()};
    ac::mr::copy(ptr, tmp.get());
    to_global_iota(global_nn,
                   global_nn_offset,
                   local_mm,
                   local_nn,
                   local_rr,
                   tmp.get(),
                   initial_value);
    ac::mr::copy(tmp.get(), ptr);
}
#endif

/**
 * Verify that each cell in the local mesh corresponds to its global index.
 * Throws an exception in the case of a failure.
 */
template <typename T>
void
verify_global_iota(const ac::shape& global_nn, const ac::shape& global_nn_offset,
                   const ac::shape& local_mm, const ac::shape& local_nn, const ac::index& local_rr,
                   const ac::mr::host_pointer<T>& ptr, const T& initial_value = 0)
{
    (void) local_nn; // Unused
    ERRCHK(ptr.size() == prod(local_mm));
    for (uint64_t i{0}; i < ptr.size(); ++i) {
        const auto global_coords{
            (global_nn + global_nn_offset + ac::to_spatial(i, local_mm) - local_rr) % global_nn};
        const auto global_index{ac::to_linear(global_coords, global_nn)};
        ERRCHK(within_machine_epsilon(ptr[i], static_cast<T>(global_index) + initial_value));
    }
}

#if defined(ACM_DEVICE_ENABLED)
template <typename T>
void
verify_global_iota(const ac::shape& global_nn, const ac::shape& global_nn_offset,
                   const ac::shape& local_mm, const ac::shape& local_nn, const ac::index& local_rr,
                   const ac::mr::device_pointer<T>& ptr, const T& initial_value = 0)
{
    ac::host_buffer<T> tmp{ptr.size()};
    ac::mr::copy(ptr, tmp.get());
    verify_global_iota(global_nn,
                       global_nn_offset,
                       local_mm,
                       local_nn,
                       local_rr,
                       tmp.get(),
                       initial_value);
    ac::mr::copy(tmp.get(), ptr);
}
#endif

template <typename T, typename Allocator, typename U>
void
fill_value(const ac::shape& dims, const ac::segment& segment, ac::mr::pointer<T, Allocator> ptr,
           const U& fill_value = 0)
{
    static_assert(std::is_convertible_v<T, U>);
    ERRCHK(ptr.size() == prod(dims));

    ac::host_ndbuffer<T>       init_buffer{segment.dims, static_cast<T>(fill_value)};
    ac::ndbuffer<T, Allocator> pack_buffer{segment.dims};
    ac::mr::copy(init_buffer.get(), pack_buffer.get());
    acm::unpack(pack_buffer.get(), dims, segment.dims, segment.offset, {ptr});
}

template <typename T, typename Allocator, typename U>
void
fill_iota(const ac::shape& dims, const ac::segment& segment, ac::mr::pointer<T, Allocator> ptr,
          const U& initial_value = 0)
{
    static_assert(std::is_convertible_v<T, U>);
    ERRCHK(ptr.size() == prod(dims));

    ac::host_ndbuffer<T> init_buffer{segment.dims};
    std::iota(init_buffer.begin(), init_buffer.end(), static_cast<T>(initial_value));
    ac::ndbuffer<T, Allocator> pack_buffer{segment.dims};
    ac::mr::copy(init_buffer.get(), pack_buffer.get());
    acm::unpack(pack_buffer.get(), dims, segment.dims, segment.offset, {ptr});
}

} // namespace ac
