#include "transform.h"

#include "datatypes.h"
#include "ndbuffer.h"

template <typename T> using Pointer = ac::mr::pointer<T, ac::mr::host_allocator>;

namespace ac {
template <typename T>
void
transformm(const Shape& dims, const Shape& subdims, const Index& offset, const Pointer<T>& in,
           Pointer<T> out)
{
    for (uint64_t out_idx{0}; out_idx < prod(subdims); ++out_idx) {
        const ac::ntuple<uint64_t> out_coords{to_spatial(out_idx, subdims)};
        ERRCHK(out_coords < subdims);

        const ac::ntuple<uint64_t> in_coords{offset + out_coords};
        ERRCHK(in_coords < dims);

        const uint64_t in_idx{to_linear(in_coords, dims)};

        ERRCHK(in_idx < in.size());
        ERRCHK(out_idx < out.size());
        out[out_idx] = in[in_idx];
    }
}
} // namespace ac

void
test_transform()
{
    {
        const ac::ntuple<uint64_t> dims{3, 3, 3, 3};
        const ac::ntuple<uint64_t> subdims{1, 2, 1, 1};
        const ac::ntuple<uint64_t> offset{1, 1, 1, 1};
        auto                       in{std::make_unique<int[]>(prod(dims))};
        auto                       out{std::make_unique<int[]>(prod(subdims))};
        std::iota(in.get(), in.get() + prod(dims), 1);
        ac::transform(dims, subdims, offset, in.get(), out.get());
        ac::print("reference", dims, in.get());
        ac::print("candidate", subdims, out.get());
    }
    {
        const Shape                               mm{8, 8};
        const Shape                               nn{6, 6};
        const Index                               rr{1, 1};
        ac::ndbuffer<int, ac::mr::host_allocator> in{mm};
        std::iota(in.begin(), in.end(), 1);
        in.display();

        ac::ndbuffer<int, ac::mr::host_allocator> out{nn};
        ac::transformm(mm, nn, rr, in.get(), out.get());
        out.display();
    }
    PRINT_LOG_INFO("OK");
}
