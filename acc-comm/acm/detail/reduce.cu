#include "reduce.h"

#include "errchk.h"

#include "print_debug.h"

#if defined(ACM_DEVICE_ENABLED)
#if defined(ACM_CUDA_ENABLED)
#include <cub/cub.cuh>
#elif defined(ACM_HIP_ENABLED)
#include <hipcub/hipcub.hpp>
#define cub hipcub
#endif
#endif

#include <numeric>

#include "datatypes.h"
#include "ndbuffer.h"

void
test_reduce()
{
    const size_t    num_segments{5};
    const ac::Shape mm{8, 8, num_segments};
    const size_t    stride{prod(slice(mm, 0, mm.size() - 1))};

    std::vector<size_t> offsets;
    for (size_t i{0}; i <= num_segments; ++i)
        offsets.push_back(i * stride);
    ac::device_buffer<size_t> doffsets{offsets.size()};
    ac::mr::copy(ac::mr::host_pointer<size_t>{offsets.size(), offsets.data()}, doffsets.get());

    const cudaStream_t stream{nullptr};

    ac::host_ndbuffer<int> hin{mm};
    ac::host_buffer<int>   hout{num_segments};
    std::iota(hin.begin(), hin.end(), 1);
    hin.display();

    ac::device_ndbuffer<int> din{mm};
    ac::device_buffer<int>   dout{num_segments};

    migrate(hin, din);

    size_t bytes{0};
    cub::DeviceSegmentedReduce::Sum(nullptr,
                                    bytes,
                                    din.data(),
                                    dout.data(),
                                    num_segments,
                                    doffsets.data(),
                                    doffsets.data() + 1,
                                    stream);
    PRINT_DEBUG(bytes);

    ac::device_buffer<uint8_t> temporary{bytes};
    cub::DeviceSegmentedReduce::Sum(temporary.data(),
                                    bytes,
                                    din.data(),
                                    dout.data(),
                                    num_segments,
                                    doffsets.data(),
                                    doffsets.data() + 1,
                                    stream);
    ERRCHK_CUDA_API(cudaStreamSynchronize(stream));

    migrate(dout, hout);
    std::cout << hout << std::endl;

    // sum_{i}^{n} = n(n+1) / 2
    auto ref_fn = [stride](const size_t i) {
        return (i + 1) * stride * ((i + 1) * stride + 1) / 2 - i * stride * (i * stride + 1) / 2;
    };
    ERRCHK(hout[0] == ref_fn(0));
    ERRCHK(hout[1] == ref_fn(1));
    ERRCHK(hout[2] == ref_fn(2));
    ERRCHK(hout[3] == ref_fn(3));
    ERRCHK(hout[4] == ref_fn(4));

    PRINT_LOG_INFO("OK");
}
