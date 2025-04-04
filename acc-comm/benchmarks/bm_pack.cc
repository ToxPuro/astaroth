#include <cstdlib>
#include <iostream>
#include <type_traits>

#include "acm/detail/convert.h"
#include "acm/detail/errchk_mpi.h"
#include "acm/detail/mpi_utils.h"
#include "acm/detail/ndbuffer.h"
#include "acm/detail/pack.h"
#include "acm/detail/partition.h"
#include "acm/detail/print_debug.h"

#if defined(ACM_DEVICE_ENABLED)
#include "acm/detail/cuda_utils.h"
#include "acm/detail/errchk_cuda.h"
#endif

#include "bm.h"

template <typename T, typename Allocator> class mpi_pack_s {
  private:
    MPI_Comm m_comm{MPI_COMM_NULL};

    std::vector<ac::segment>                m_segments;
    std::vector<MPI_Datatype>               m_subarrays;
    std::vector<ac::ndbuffer<T, Allocator>> m_pack_buffers;

  public:
    mpi_pack_s(const MPI_Comm& parent_comm, const ac::shape& nn, const ac::index& rr,
               const std::vector<ac::segment>& segments)
        : m_segments{segments}
    {
        m_comm = ac::mpi::cart_comm_create(parent_comm, nn);

        const auto mm{ac::mpi::get_local_mm(m_comm, nn, rr)};
        for (const auto& segment : m_segments) {
            m_subarrays.push_back(ac::mpi::subarray_create(mm,
                                                           segment.dims,
                                                           segment.offset,
                                                           ac::mpi::get_dtype<T>()));
            m_pack_buffers.push_back(ac::ndbuffer<T, Allocator>{segment.dims});
        }
    }
    ~mpi_pack_s()
    {
        for (auto& subarray : m_subarrays)
            ac::mpi::subarray_destroy(&subarray);
        ac::mpi::cart_comm_destroy(&m_comm);
    }

    mpi_pack_s(const mpi_pack_s&)            = delete; // Copy constructor
    mpi_pack_s& operator=(const mpi_pack_s&) = delete; // Copy assignment operator
    mpi_pack_s(mpi_pack_s&&)                 = delete; // Move constructor
    mpi_pack_s& operator=(mpi_pack_s&&)      = delete; // Move assignment operator

    void pack(const std::vector<ac::mr::pointer<T, Allocator>>& inputs)
    {
        for (size_t j{0}; j < inputs.size(); ++j) {
            for (size_t i{0}; i < m_pack_buffers.size(); ++i) {
                int          position{0};
                const size_t total_bytes{sizeof(T) * prod(m_segments[i].dims)};

                ERRCHK_MPI_API(MPI_Pack(inputs[j].data(),
                                        1,
                                        m_subarrays[i],
                                        m_pack_buffers[i].data(),
                                        as<int>(total_bytes),
                                        &position,
                                        m_comm));
                ERRCHK_MPI(as<size_t>(position) == total_bytes);
            }
        }
    }
    void unpack(const ac::shape& mm, std::vector<ac::mr::pointer<T, Allocator>>& outputs)
    {
        for (size_t j{0}; j < outputs.size(); ++j) {
            for (size_t i{0}; i < m_pack_buffers.size(); ++i) {
                int          position{0};
                const size_t total_bytes{sizeof(T) * prod(m_segments[i].dims)};

                ERRCHK_MPI_API(MPI_Unpack(m_pack_buffers[i].data(),
                                          as<int>(total_bytes),
                                          &position,
                                          outputs[j].data(),
                                          1,
                                          m_subarrays[i],
                                          m_comm));
                ERRCHK_MPI(as<size_t>(position) == total_bytes);
            }
        }
    }
};

template <typename T, typename Allocator> class acm_pack_s {
  private:
    std::vector<ac::segment>                m_segments;
    std::vector<ac::ndbuffer<T, Allocator>> m_pack_buffers;

  public:
    acm_pack_s(const std::vector<ac::segment>& segments)
        : m_segments{segments}
    {
        for (const auto& segment : m_segments)
            m_pack_buffers.push_back(ac::ndbuffer<T, Allocator>{segment.dims});
    }

    void pack(const ac::shape& mm, const std::vector<ac::mr::pointer<T, Allocator>>& inputs)
    {
        for (size_t j{0}; j < inputs.size(); ++j)
            for (size_t i{0}; i < m_pack_buffers.size(); ++i)
                acm::pack(mm,
                          m_segments[i].dims,
                          m_segments[i].offset,
                          {inputs[j]},
                          m_pack_buffers[i].get());
    }
    void unpack(const ac::shape& mm, std::vector<ac::mr::pointer<T, Allocator>>& outputs)
    {
        for (size_t j{0}; j < outputs.size(); ++j)
            for (size_t i{0}; i < m_pack_buffers.size(); ++i)
                acm::unpack(m_pack_buffers[i].get(),
                            mm,
                            m_segments[i].dims,
                            m_segments[i].offset,
                            {outputs[j]});
    }
};

template <typename T, typename Allocator> class acm_pack_packed_s {
  private:
    std::vector<ac::segment>                m_segments;
    std::vector<ac::ndbuffer<T, Allocator>> m_pack_buffers;

  public:
    acm_pack_packed_s(const size_t ninputs, const std::vector<ac::segment>& segments)
        : m_segments{segments}
    {
        for (const auto& segment : m_segments)
            m_pack_buffers.push_back(ac::ndbuffer<T, Allocator>{ninputs * segment.dims});
    }

    void pack(const ac::shape& mm, const std::vector<ac::mr::pointer<T, Allocator>>& inputs)
    {
        for (size_t i{0}; i < m_pack_buffers.size(); ++i)
            acm::pack(mm,
                      m_segments[i].dims,
                      m_segments[i].offset,
                      inputs,
                      m_pack_buffers[i].get());
    }
    void unpack(const ac::shape& mm, std::vector<ac::mr::pointer<T, Allocator>>& outputs)
    {
        for (size_t i{0}; i < m_pack_buffers.size(); ++i)
            acm::unpack(m_pack_buffers[i].get(),
                        mm,
                        m_segments[i].dims,
                        m_segments[i].offset,
                        outputs);
    }
};

template <typename T, typename Allocator> class acm_pack_batched_s {
  private:
    std::vector<ac::segment>                m_segments;
    std::vector<ac::ndbuffer<T, Allocator>> m_pack_buffers;

  public:
    acm_pack_batched_s(const size_t ninputs, const std::vector<ac::segment>& segments)
        : m_segments{segments}
    {
        for (const auto& segment : m_segments)
            m_pack_buffers.push_back(ac::ndbuffer<T, Allocator>{ninputs * segment.dims});
    }

    void pack(const ac::shape& mm, const std::vector<ac::mr::pointer<T, Allocator>>& inputs)
    {
        acm::pack_batched(mm, inputs, m_segments, ac::unwrap_get(m_pack_buffers));
    }
    void unpack(const ac::shape& mm, std::vector<ac::mr::pointer<T, Allocator>>& outputs)
    {
        for (size_t i{0}; i < m_pack_buffers.size(); ++i)
            acm::unpack_batched(m_segments, ac::unwrap_get(m_pack_buffers), mm, outputs);
    }
};

int
main(int argc, char* argv[])
{
    ac::mpi::init_funneled();
    try {
        std::cerr << "Usage: ./bm_pack <dim> <radius> <ndims> <nsamples> <jobid>" << std::endl;
        const size_t dim{(argc > 1) ? std::stoull(argv[1]) : 32};
        const size_t radius{(argc > 2) ? std::stoull(argv[2]) : 3};
        const size_t ndims{(argc > 3) ? std::stoull(argv[3]) : 3};
        const size_t nsamples{(argc > 4) ? std::stoull(argv[4]) : 10};
        const size_t jobid{(argc > 5) ? std::stoull(argv[5]) : 0};

        const auto nn{ac::make_shape(ndims, dim)};
        const auto rr{ac::make_index(ndims, radius)};
        const auto mm{nn + 2 * rr};

        PRINT_DEBUG(dim);
        PRINT_DEBUG(radius);
        PRINT_DEBUG(ndims);
        PRINT_DEBUG(nsamples);
        PRINT_DEBUG(jobid);
        PRINT_DEBUG(mm);
        PRINT_DEBUG(nn);
        PRINT_DEBUG(rr);

        const auto output_file{"bm-pack.csv"};
        FILE*      fp{fopen(output_file, "w")};
        ERRCHK(fp);
        fprintf(fp, "impl,dim,radius,ndims,sample,nsamples,jobid,ns\n");
        ERRCHK(fclose(fp) == 0);

        auto print = [&](const std::string&                                label,
                         const std::vector<std::chrono::nanoseconds::rep>& results) {
            FILE* fp{fopen(output_file, "a")};
            ERRCHK(fp);

            for (size_t i{0}; i < results.size(); ++i) {
                fprintf(fp, "%s", label.c_str());
                fprintf(fp, ",%zu", dim);
                fprintf(fp, ",%zu", radius);
                fprintf(fp, ",%zu", ndims);
                fprintf(fp, ",%zu", i);
                fprintf(fp, ",%zu", nsamples);
                fprintf(fp, ",%zu", jobid);
                fprintf(fp, ",%lld", results[i]);
                fprintf(fp, "\n");
            }
            ERRCHK(fclose(fp) == 0);
        };

        // Setup the benchmark
        using T         = double;
        using Allocator = ac::mr::device_allocator;
        constexpr size_t ninputs{3};

        // Inputs
        std::vector<ac::ndbuffer<T, Allocator>> input_buffers;
        for (size_t i{0}; i < ninputs; ++i)
            input_buffers.push_back(ac::ndbuffer<T, Allocator>{mm});
        auto inputs{ac::unwrap_get(input_buffers)};

        // Segments and outputs
        const auto                              segments{prune(partition(mm, nn, rr), nn, rr)};
        std::vector<ac::ndbuffer<T, Allocator>> output_buffers;
        for (const auto& segment : segments)
            output_buffers.push_back(ac::ndbuffer<T, Allocator>{segment.dims});
        auto outputs{ac::unwrap_get(output_buffers)};
        ERRCHK_MPI(segments.size() == outputs.size());

        // MPI subarrays
        MPI_Comm                  cart_comm{ac::mpi::cart_comm_create(MPI_COMM_WORLD, nn)};
        std::vector<MPI_Datatype> subarrays(segments.size(), MPI_DATATYPE_NULL);
        for (size_t i{0}; i < segments.size(); ++i)
            subarrays[i] = ac::mpi::subarray_create(mm,
                                                    segments[i].dims,
                                                    segments[i].offset,
                                                    ac::mpi::get_dtype<T>());

        // Functions
        auto init = [&inputs]() {
            for (auto& input : inputs)
                bm::randomize(input);
        };
        auto sync = []() {
#if defined(ACM_DEVICE_ENABLED)
            ERRCHK_CUDA_API(cudaDeviceSynchronize());
#endif
            ERRCHK_MPI_API(MPI_Barrier(MPI_COMM_WORLD));
        };

        auto bench_mpi = [&inputs, &segments, &subarrays, &outputs, &cart_comm]() {
            for (size_t j{0}; j < inputs.size(); ++j) {
                for (size_t i{0}; i < outputs.size(); ++i) {
                    int          position{0};
                    const size_t total_bytes{sizeof(T) * prod(segments[i].dims)};

                    ERRCHK_MPI_API(MPI_Pack(inputs[j].data(),
                                            1,
                                            subarrays[i],
                                            outputs[i].data(),
                                            as<int>(total_bytes),
                                            &position,
                                            cart_comm));
                }
            }
        };
        auto bench_acm = [&segments, &mm, &inputs, &outputs]() {
            for (size_t j{0}; j < inputs.size(); ++j)
                for (size_t i{0}; i < outputs.size(); ++i)
                    acm::pack(mm, segments[i].dims, segments[i].offset, {inputs[j]}, outputs[i]);
        };
        auto bench_acm_packed = [&segments, &mm, &inputs, &outputs]() {
            for (size_t i{0}; i < segments.size(); ++i)
                acm::pack(mm, segments[i].dims, segments[i].offset, inputs, outputs[i]);
        };
        auto bench_acm_packed_batched = [&segments, &mm, &inputs, &outputs]() {
            acm::pack_batched(mm, inputs, segments, outputs);
        };

        // Verify
        std::vector<ac::host_ndbuffer<T>> ref_buffers;
        for (const auto& input_buffer : input_buffers)
            ref_buffers.push_back(input_buffer.to_host());
        auto refs{ac::unwrap_get(ref_buffers)};

        T initial_value{1};
        for (auto& ref : ref_buffers) {
            std::iota(ref.begin(), ref.end(), initial_value);
            initial_value += ref.size();
        }

        auto reset_inputs = [&]() {
            for (size_t i{0}; i < ref_buffers.size(); ++i)
                ac::mr::copy(refs[i], inputs[i]);
        };
        auto verify = [&]() {
            for (size_t i{0}; i < inputs.size(); ++i)
                for (size_t j{0}; j < inputs[i].size(); ++j)
                    ERRCHK(within_machine_epsilon(inputs[i][j], refs[i][j]));
        };

        acm_pack_s<T, Allocator>         acm_pack{segments};
        acm_pack_packed_s<T, Allocator>  acm_pack_packed{inputs.size(), segments};
        acm_pack_batched_s<T, Allocator> acm_pack_batched{inputs.size(), segments};
        mpi_pack_s<T, Allocator>         mpi_pack{MPI_COMM_WORLD, nn, rr, segments};

        auto bm_acm         = [&]() { acm_pack.pack(mm, inputs); };
        auto bm_acm_packed  = [&]() { acm_pack_packed.pack(mm, inputs); };
        auto bm_acm_batched = [&]() { acm_pack_batched.pack(mm, inputs); };
        auto bm_mpi         = [&]() { mpi_pack.pack(inputs); };

        print("acm", bm::benchmark(init, bm_acm, sync, nsamples));
        print("acm_packed", bm::benchmark(init, bm_acm_packed, sync, nsamples));
        print("acm_batched", bm::benchmark(init, bm_acm_batched, sync, nsamples));
        print("mpi", bm::benchmark(init, bm_mpi, sync, nsamples));

        acm_pack.pack(mm, inputs);
        const auto results{bm::benchmark(
            init, [&]() { acm_pack.pack(mm, inputs); }, sync, 10)};
        print("test", results);

        reset_inputs();
        acm_pack.pack(mm, inputs);
        acm_pack.unpack(mm, inputs);
        input_buffers[0].display();
        // verify(); // TODO scramble

        // Cleanup
        for (auto& subarray : subarrays)
            ac::mpi::subarray_destroy(&subarray);
        subarrays.clear();

        ac::mpi::cart_comm_destroy(&cart_comm);
    }
    catch (const std::exception& e) {
        ac::mpi::abort();
        return EXIT_FAILURE;
    }
    ac::mpi::finalize();
    return EXIT_SUCCESS;
}
