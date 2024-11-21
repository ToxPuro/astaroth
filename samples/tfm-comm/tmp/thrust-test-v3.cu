#include <cstdlib>
#include <iostream>

#include <thrust/device_buffer.h>
#include <thrust/host_buffer.h>
#include <thrust/system/cuda/memory_resource.h>

template <typename vector>
static void
print(const std::string& label, const vector& vec)
{
    std::cout << label << ": { ";
    for (const auto& elem : vec)
        std::cout << elem << " ";
    std::cout << "}" << std::endl;
}

template <typename vector>
void
process_vector(vector& vec)
{
    for (size_t i{0}; i < vec.size(); ++i)
        vec[i] = 2 * vec[i];
}

int
main()
{
    std::cout << "lala" << std::endl;
    thrust::host_vector<double> vec(10, 1);
    thrust::device_vector<double> dvec(200, 10);
    print("vec", vec);
    process_vector(vec);
    print("vec", vec);
    process_vector(dvec);
    print("dvec", dvec);

    thrust::host_vector<double, thrust::mr::stateles_resource_allocator<
                                    double, thrust::system::cuda::pinned_memory_resource>>
        phvec(10);
    // thrust::mr::stateles_resource_allocator<double, thrust::system::cpp::pinned_memory_resource>
    // thrust::host_vector<double, thrust::cuda::experimental::pinned_allocator<double>> phvec(10);
    // thrust::universal_host_pinned_vector<double> phvec(10);
    // thrust::host_vector<double, thrust::mr::allocator<
    //                                 double,
    //                                 cuda::mr::pinned_memory_resource{cudaHostAllocDefault}>>
    //     phvec(10);

    return EXIT_SUCCESS;
}
