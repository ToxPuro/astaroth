#include "transform.h"

#include "datatypes.h"
#include "ndbuffer.h"

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
        using DeviceNdBuffer = ac::ndbuffer<UserDatatype, ac::mr::device_allocator>;
        using HostNdBuffer   = ac::ndbuffer<UserDatatype, ac::mr::host_allocator>;
        const ac::shape mm{8, 8};
        const ac::shape nn{6, 6};
        const ac::index rr{1, 1};
        HostNdBuffer    hin{mm};
        HostNdBuffer    houtref{nn};
        HostNdBuffer    hout{nn};
        DeviceNdBuffer  din{mm};
        DeviceNdBuffer  dout{nn};

        std::iota(hin.begin(), hin.end(), 1);
        ac::transform(mm, nn, rr, hin.get(), houtref.get());
        hin.display();
        houtref.display();

        migrate(hin, din);
        ac::transform(mm, nn, rr, din.get(), dout.get());
        migrate(dout, hout);

        hout.display();

        ERRCHK(equals(houtref.get(), hout.get()));
    }
    PRINT_LOG_INFO("OK");
}
