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

#include "ndbuffer.h"

#include "errchk_cuda.h"
#include "ntuple.h"
#include "pack.h"
#include "pointer.h"
#include "type_conversion.h"

enum class ReduceType { sum, max, min };

template <typename T> class ReduceTask {
  private:
    // Local memory
    ac::device_buffer<T> pack_buffer;

  public:
    ReduceTask(const size_t count)
        : pack_buffer{count} {};

    void reduce(const ac::shape& dims, const ac::shape& subdims, const ac::index& offset,
                const std::vector<ac::mr::device_pointer<T>>& inputs,
                ac::mr::device_pointer<T>                     output)
    {
        // Check that the output memory resource can hold all segments
        ERRCHK(inputs.size() == output.size());
        ERRCHK(prod(subdims) <= pack_buffer.size())

        // Pack
        pack(dims, subdims, offset, inputs, pack_buffer.get());

        // Local reduce
        const auto         num_segments{inputs.size()};
        const auto         stride{prod(subdims)};
        ac::host_buffer<T> offsets{num_segments + 1};
        for (size_t i{0}; i <= num_segments; ++i)
            offsets[i] = i * stride;

        auto               device_offsets{offsets.to_device()};
        const cudaStream_t stream{nullptr};

        size_t bytes{0};
        ERRCHK_CUDA_API(cub::DeviceSegmentedReduce::Sum(nullptr,
                                                        bytes,
                                                        pack_buffer.data(),
                                                        output.data(),
                                                        as<int>(num_segments),
                                                        device_offsets.data(),
                                                        device_offsets.data() + 1,
                                                        stream));
        PRINT_DEBUG(bytes);

        ac::device_buffer<T> tmp{bytes};
        ERRCHK_CUDA_API(cub::DeviceSegmentedReduce::Sum(tmp.data(),
                                                        bytes,
                                                        pack_buffer.data(),
                                                        output.data(),
                                                        as<int>(num_segments),
                                                        device_offsets.data(),
                                                        device_offsets.data() + 1,
                                                        stream));
        ERRCHK_CUDA_API(cudaStreamSynchronize(stream));

        // Global reduce
        // TODO
    }
};

/** Return the expected sum of the `i`th block, where each block contains `stride` elements and the
 *  blocks have been initialized in an increasing order from 1.
 *
 * For example:
 *  Block 0: [1, 100], where subsequent values are 1, 2, 3, ...
 *  Block 1: [101, 201]
 *  ...
 *
 * Expression: sum_{i}^{n} = n(n+1) / 2
 */
static size_t
expected_sum(const size_t i, const size_t stride)
{
    return (i + 1) * stride * ((i + 1) * stride + 1) / 2 - i * stride * (i * stride + 1) / 2;
}

void
test_reduce()
{
    {
        const size_t    num_segments{5};
        const ac::shape mm{8, 8, num_segments};
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

        ERRCHK(as<size_t>(hout[0]) == expected_sum(0, stride));
        ERRCHK(as<size_t>(hout[1]) == expected_sum(1, stride));
        ERRCHK(as<size_t>(hout[2]) == expected_sum(2, stride));
        ERRCHK(as<size_t>(hout[3]) == expected_sum(3, stride));
        ERRCHK(as<size_t>(hout[4]) == expected_sum(4, stride));
    }
    {
        ac::shape mm{4, 4};
        ac::shape nn{2, 2};
        ac::index rr{1, 1};

        ac::host_ndbuffer<uint64_t> h0{mm};
        ac::host_ndbuffer<uint64_t> h1{mm};
        ac::host_ndbuffer<uint64_t> h2{mm};

        std::iota(h0.begin(), h0.end(), 1);
        std::iota(h1.begin(), h1.end(), 1 + h0.size());
        std::iota(h2.begin(), h2.end(), 1 + h0.size() + h1.size());

        h0.display();
        h1.display();
        h2.display();

        auto d0{h0.to_device()};
        auto d1{h1.to_device()};
        auto d2{h2.to_device()};

        const auto                  count{h0.size() + h1.size() + h2.size()};
        ReduceTask<uint64_t>        task{count};
        std::vector                 inputs{d0.get(), d1.get(), d2.get()};
        ac::device_buffer<uint64_t> output{inputs.size()};
        task.reduce(mm, nn, rr, std::vector{d0.get(), d1.get(), d2.get()}, output.get());

        auto host_output{output.to_host()};
        ERRCHK(host_output.size() == inputs.size());
        host_output.display();

        ERRCHK(host_output[0] == 6 + 7 + 10 + 11);
        ERRCHK(host_output[1] == 22 + 23 + 26 + 27);
        ERRCHK(host_output[2] == 38 + 39 + 42 + 43);
    }

    PRINT_LOG_INFO("OK");
}
