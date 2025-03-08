#include "algorithm.h"

#include "type_conversion.h"

constexpr size_t MAX_NDIMS{4};
using shape_t = ac::static_ntuple<uint64_t, MAX_NDIMS>;
using index_t = ac::static_ntuple<uint64_t, MAX_NDIMS>;

// namespace device {

// }

// template <typename T, typename Function>
// void
// transform(const ac::mr::device_pointer<T>& input, const Function& fn,
//           ac::mr::device_pointer<T> output)
// {
// }

// template void
// transform<double, std::function<void(const double&)>>(const ac::mr::device_pointer<double>&
// input,
//                                                       const std::function<void(const double&)>&
//                                                       fn, ac::mr::device_pointer<double> output);

namespace ac {

namespace device {

template <typename T>
__global__ void
xcorr(const shape_t& mm, const shape_t& nn, const shape_t& nn_offset, const T* input,
      const shape_t& nk, const T* kernel, T* output)
{
    const uint64_t block_idx{static_cast<uint64_t>(threadIdx.x) + blockIdx.x * blockDim.x};

    if (block_idx < prod(nn)) {
        const auto block_coords{to_spatial(block_idx, nn)};
        const auto out_coords{nn_offset + block_coords};
        const auto out_idx{to_linear(out_coords, mm)};

        T result{0};
        for (uint64_t kernel_idx{0}; kernel_idx < prod(nk); ++kernel_idx) {
            const auto kernel_coords{ac::to_spatial(kernel_idx, nk)};
            const auto diff{(nk - static_cast<uint64_t>(1)) / static_cast<uint64_t>(2)};
            const auto in_coords{out_coords - diff + kernel_coords};
            const auto in_idx{ac::to_linear(in_coords, mm)};
            result += input[in_idx] * kernel[kernel_idx];
        }
        output[out_idx] = result;
    }
}

} // namespace device

template <typename T>
void
xcorr(const ac::shape& in_mm, const ac::shape& in_nn, const ac::shape& in_nn_offset,
      const ac::mr::device_pointer<T>& in_input, const ac::shape& in_nk,
      const ac::mr::device_pointer<T>& in_kernel, ac::mr::device_pointer<T> in_output)
{
    ERRCHK(in_input.data() != in_output.data());
    ERRCHK(same_size(in_mm, in_nn, in_nn_offset, in_nk));
    ERRCHK(in_input.size() == in_output.size());
    ERRCHK(in_nn_offset + in_nn <= in_mm);

    const shape_t mm{in_mm};
    const shape_t nn{in_nn};
    const index_t nn_offset{in_nn_offset};
    const T*      input{in_input.data()};
    const shape_t nk{in_nk};
    const T*      kernel{in_kernel.data()};
    T*            output{in_output.data()};

    const uint64_t tpb{256};
    const uint64_t bpg{(prod(nn) + tpb - 1) / tpb};

    device::xcorr<<<as<uint32_t>(bpg), as<uint32_t>(tpb)>>>(mm,
                                                            nn,
                                                            nn_offset,
                                                            input,
                                                            nk,
                                                            kernel,
                                                            output);
    ERRCHK_CUDA_KERNEL();
    cudaDeviceSynchronize();
}

template void xcorr<double>(const ac::shape& mm, const ac::shape& nn, const ac::shape& nn_offset,
                            const ac::mr::device_pointer<double>& input, const ac::shape& nk,
                            const ac::mr::device_pointer<double>& kernel,
                            ac::mr::device_pointer<double>        output);

template void xcorr<uint64_t>(const ac::shape& mm, const ac::shape& nn, const ac::shape& nn_offset,
                              const ac::mr::device_pointer<uint64_t>& input, const ac::shape& nk,
                              const ac::mr::device_pointer<uint64_t>& kernel,
                              ac::mr::device_pointer<uint64_t>        output);

} // namespace ac
