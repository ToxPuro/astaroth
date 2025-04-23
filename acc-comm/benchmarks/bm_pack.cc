#include <cstdlib>
#include <iostream>
#include <sstream>

#include "acm/detail/convert.h"
#include "acm/detail/errchk_mpi.h"
#include "acm/detail/math_utils.h"
#include "acm/detail/mpi_utils.h"
#include "acm/detail/ndbuffer.h"
#include "acm/detail/pack.h"
#include "acm/detail/partition.h"
#include "acm/detail/print_debug.h"

#if defined(ACM_DEVICE_ENABLED)
#include "acm/detail/cuda_utils.h"
#include "acm/detail/errchk_cuda.h"
#endif

#include "acm/detail/experimental/mpi_utils_experimental.h"
#include "bm.h"

template <typename T, typename Allocator> class mpi_pack_strategy {
  private:
    ac::mpi::comm              m_comm;
    ac::mpi::subarray<T>       m_subarray;
    ac::ndbuffer<T, Allocator> m_pack_buffer;

  public:
    mpi_pack_strategy(const MPI_Comm& parent_comm, const ac::shape& dims, const ac::shape subdims,
                      const ac::index& offset)
        : m_comm{parent_comm}, m_subarray{dims, subdims, offset}, m_pack_buffer{subdims}
    {
    }

    void pack(const ac::mr::pointer<T, Allocator>& input)
    {
        int bytes{-1};
        ERRCHK_MPI_API(MPI_Type_size(m_subarray.get(), &bytes));
        ERRCHK_MPI(as<size_t>(bytes) == sizeof(T) * m_pack_buffer.size());

        int position{0};
        ERRCHK_MPI_API(MPI_Pack(input.data(),
                                1,
                                m_subarray.get(),
                                m_pack_buffer.data(),
                                bytes,
                                &position,
                                m_comm.get()));
        ERRCHK_MPI(position == bytes);
    }

    void unpack(ac::mr::pointer<T, Allocator> output) const
    {
        int bytes{-1};
        ERRCHK_MPI_API(MPI_Type_size(m_subarray.get(), &bytes));
        ERRCHK_MPI(as<size_t>(bytes) == sizeof(T) * m_pack_buffer.size());

        int position{0};
        ERRCHK_MPI_API(MPI_Unpack(m_pack_buffer.data(),
                                  bytes,
                                  &position,
                                  output.data(),
                                  1,
                                  m_subarray.get(),
                                  m_comm.get()));
        ERRCHK_MPI(position == bytes);
    }
};

template <typename T, typename Allocator> class mpi_pack_strategy_packed {
  private:
    std::vector<mpi_pack_strategy<T, Allocator>> m_tasks;

  public:
    mpi_pack_strategy_packed(const MPI_Comm& parent_comm, const ac::shape& dims,
                             const ac::segment& segment, const size_t ninputs)
    {
        for (size_t i{0}; i < ninputs; ++i)
            m_tasks.push_back({parent_comm, dims, segment.dims, segment.offset});
    }

    void pack(const std::vector<ac::mr::pointer<T, Allocator>>& inputs)
    {
        ERRCHK_MPI(inputs.size() == m_tasks.size());

        for (size_t i{0}; i < inputs.size(); ++i)
            m_tasks[i].pack(inputs[i]);
    }

    void unpack(std::vector<ac::mr::pointer<T, Allocator>> outputs) const
    {
        ERRCHK_MPI(outputs.size() == m_tasks.size());

        for (size_t i{0}; i < outputs.size(); ++i)
            m_tasks[i].unpack(outputs[i]);
    }
};

template <typename T, typename Allocator> class mpi_pack_strategy_batched {
  private:
    std::vector<mpi_pack_strategy_packed<T, Allocator>> m_tasks;

  public:
    mpi_pack_strategy_batched(const MPI_Comm& parent_comm, const ac::shape& dims,
                              const std::vector<ac::segment>& segments, const size_t ninputs)
    {
        for (const auto& segment : segments)
            m_tasks.push_back({parent_comm, dims, segment, ninputs});
    }

    void pack(const std::vector<ac::mr::pointer<T, Allocator>>& inputs)
    {
        for (auto& task : m_tasks)
            task.pack(inputs);
    }

    void unpack(std::vector<ac::mr::pointer<T, Allocator>> outputs) const
    {
        for (const auto& task : m_tasks)
            task.unpack(outputs);
    }
};

template <typename T, typename Allocator> class mpi_pack_strategy_hindexed {
  private:
    ac::mpi::comm                  m_comm;
    ac::mpi::hindexed_block<T>     m_input_block;
    ac::mpi::hindexed_block<T>     m_output_block;
    ac::buffer<uint8_t, Allocator> m_pack_buffer;

  public:
    mpi_pack_strategy_hindexed(const MPI_Comm& parent_comm, const ac::shape& dims,
                               const ac::segment&                                segment,
                               const std::vector<ac::mr::pointer<T, Allocator>>& inputs,
                               const std::vector<ac::mr::pointer<T, Allocator>>& outputs)
        : m_comm{parent_comm},
          m_input_block(dims, segment.dims, segment.offset, ac::unwrap_data(inputs)),
          m_output_block(dims, segment.dims, segment.offset, ac::unwrap_data(outputs)),
          m_pack_buffer{sizeof(T) * inputs.size() * prod(segment.dims)}
    {
        ERRCHK_MPI(inputs.size() == outputs.size());

        int pack_size{-1};
        ERRCHK_MPI_API(MPI_Pack_size(1, m_input_block.get(), m_comm.get(), &pack_size));
        ERRCHK_MPI(m_pack_buffer.size() == as<size_t>(pack_size));
    }

    void pack()
    {
        int position{0};
        ERRCHK_MPI_API(MPI_Pack(MPI_BOTTOM,
                                1,
                                m_input_block.get(),
                                m_pack_buffer.data(),
                                as<int>(m_pack_buffer.size()),
                                &position,
                                m_comm.get()));
        ERRCHK_MPI(as<size_t>(position) == m_pack_buffer.size());
    }

    void unpack() const
    {
        int position{0};
        ERRCHK_MPI_API(MPI_Unpack(m_pack_buffer.data(),
                                  as<int>(m_pack_buffer.size()),
                                  &position,
                                  MPI_BOTTOM,
                                  1,
                                  m_output_block.get(),
                                  m_comm.get()));
        ERRCHK_MPI(as<size_t>(position) == m_pack_buffer.size());
    }
};

template <typename T, typename Allocator> class mpi_pack_strategy_hindexed_batched {
  private:
    std::vector<mpi_pack_strategy_hindexed<T, Allocator>> m_tasks;

  public:
    mpi_pack_strategy_hindexed_batched(const MPI_Comm& parent_comm, const ac::shape& dims,
                                       const std::vector<ac::segment>&                   segments,
                                       const std::vector<ac::mr::pointer<T, Allocator>>& inputs,
                                       const std::vector<ac::mr::pointer<T, Allocator>>& outputs)
    {
        for (const auto& segment : segments)
            m_tasks.push_back({parent_comm, dims, segment, inputs, outputs});
    }

    void pack()
    {
        for (auto& task : m_tasks)
            task.pack();
    }

    void unpack() const
    {
        for (const auto& task : m_tasks)
            task.unpack();
    }
};

template <typename T, typename Allocator> class acm_pack_strategy {
  private:
    ac::segment              m_segment;
    ac::buffer<T, Allocator> m_pack_buffer;

  public:
    acm_pack_strategy(const ac::segment& segment)
        : m_segment{segment}, m_pack_buffer{prod(segment.dims)}
    {
    }

    void pack(const ac::shape& dims, const ac::mr::pointer<T, Allocator>& input)
    {
        acm::pack(dims, m_segment.dims, m_segment.offset, {input}, m_pack_buffer.get());
    }

    void unpack(const ac::shape& dims, ac::mr::pointer<T, Allocator>& output) const
    {
        acm::unpack(m_pack_buffer.get(), dims, m_segment.dims, m_segment.offset, {output});
    }
};

template <typename T, typename Allocator> class acm_pack_strategy_grouped {
  private:
    std::vector<acm_pack_strategy<T, Allocator>> m_packets;

  public:
    acm_pack_strategy_grouped(const ac::segment& segment, const size_t ninputs)
    {
        for (size_t i{0}; i < ninputs; ++i)
            m_packets.push_back(segment);
    }

    void pack(const ac::shape& dims, const std::vector<ac::mr::pointer<T, Allocator>>& inputs)
    {
        ERRCHK(inputs.size() == m_packets.size());

        for (size_t i{0}; i < inputs.size(); ++i)
            m_packets[i].pack(dims, inputs[i]);
    }

    void unpack(const ac::shape& dims, std::vector<ac::mr::pointer<T, Allocator>> outputs) const
    {
        ERRCHK(outputs.size() == m_packets.size());

        for (size_t i{0}; i < outputs.size(); ++i)
            m_packets[i].unpack(dims, outputs[i]);
    }
};

template <typename T, typename Allocator> class acm_pack_strategy_grouped_batched {
  private:
    std::vector<acm_pack_strategy_grouped<T, Allocator>> m_packets;

  public:
    acm_pack_strategy_grouped_batched(const std::vector<ac::segment>& segments,
                                      const size_t                    nfields)
    {
        for (const auto& segment : segments)
            m_packets.push_back({segment, nfields});
    }

    void pack(const ac::shape& dims, const std::vector<ac::mr::pointer<T, Allocator>>& inputs)
    {
        for (auto& packet : m_packets)
            packet.pack(dims, inputs);
    }

    void unpack(const ac::shape& dims, std::vector<ac::mr::pointer<T, Allocator>> outputs) const
    {
        for (auto& packet : m_packets)
            packet.unpack(dims, outputs);
    }
};

template <typename T, typename Allocator> class acm_pack_strategy_packed {
  private:
    ac::segment              m_segment;
    size_t                   m_npacked;
    ac::buffer<T, Allocator> m_pack_buffer;

  public:
    acm_pack_strategy_packed(const ac::segment& segment, const size_t npacked)
        : m_segment{segment}, m_npacked{npacked}, m_pack_buffer{prod(segment.dims) * npacked}
    {
    }

    void pack(const ac::shape& dims, const std::vector<ac::mr::pointer<T, Allocator>>& inputs)
    {
        ERRCHK(inputs.size() == m_npacked);
        acm::pack(dims, m_segment.dims, m_segment.offset, inputs, m_pack_buffer.get());
    }

    void unpack(const ac::shape& dims, std::vector<ac::mr::pointer<T, Allocator>> outputs) const
    {
        ERRCHK(outputs.size() == m_npacked);
        acm::unpack(m_pack_buffer.get(), dims, m_segment.dims, m_segment.offset, outputs);
    }
};

template <typename T, typename Allocator> class acm_pack_strategy_packed_batched {
  private:
    std::vector<acm_pack_strategy_packed<T, Allocator>> m_packets;

  public:
    acm_pack_strategy_packed_batched(const std::vector<ac::segment>& segments, const size_t npacked)
    {
        for (const auto& segment : segments)
            m_packets.push_back({segment, npacked});
    }

    void pack(const ac::shape& dims, const std::vector<ac::mr::pointer<T, Allocator>>& inputs)
    {
        for (auto& packet : m_packets)
            packet.pack(dims, inputs);
    }

    void unpack(const ac::shape& dims, std::vector<ac::mr::pointer<T, Allocator>> outputs) const
    {
        for (auto& packet : m_packets)
            packet.unpack(dims, outputs);
    }
};

template <typename T, typename Allocator> class acm_pack_strategy_batched {
  private:
    std::vector<ac::segment>              m_segments;
    size_t                                m_npacked;
    std::vector<ac::buffer<T, Allocator>> m_pack_buffers;

  public:
    acm_pack_strategy_batched(const std::vector<ac::segment>& segments, const size_t npacked)
        : m_segments{segments}, m_npacked{npacked}
    {
        for (const auto& segment : m_segments)
            m_pack_buffers.push_back(ac::buffer<T, Allocator>{prod(segment.dims) * npacked});
    }

    void pack(const ac::shape& mm, const std::vector<ac::mr::pointer<T, Allocator>>& inputs)
    {
        ERRCHK(inputs.size() == m_npacked);
        acm::pack_batched(mm, inputs, m_segments, ac::unwrap_get(m_pack_buffers));
    }

    void unpack(const ac::shape& mm, std::vector<ac::mr::pointer<T, Allocator>> outputs)
    {
        ERRCHK(outputs.size() == m_npacked);
        acm::unpack_batched(m_segments, ac::unwrap_get(m_pack_buffers), mm, outputs);
    }
};

int
main(int argc, char* argv[])
{
    ac::mpi::init_funneled();
    try {
        std::cerr << "Usage: ./bm_pack <dim> <radius> <ndims> <ninputs> <nsamples> <jobid>"
                  << std::endl;
        const size_t dim{(argc > 1) ? std::stoull(argv[1]) : 32};
        const size_t radius{(argc > 2) ? std::stoull(argv[2]) : 3};
        const size_t ndims{(argc > 3) ? std::stoull(argv[3]) : 3};
        const size_t ninputs{(argc > 4) ? std::stoull(argv[4]) : 4};
        const size_t nsamples{(argc > 5) ? std::stoull(argv[5]) : 10};
        const size_t jobid{(argc > 6) ? std::stoull(argv[6]) : 0};

        const auto nn{ac::make_shape(ndims, dim)};
        const auto rr{ac::make_index(ndims, radius)};
        const auto mm{nn + 2 * rr};

        PRINT_DEBUG(dim);
        PRINT_DEBUG(radius);
        PRINT_DEBUG(ndims);
        PRINT_DEBUG(ninputs);
        PRINT_DEBUG(nsamples);
        PRINT_DEBUG(jobid);
        PRINT_DEBUG(mm);
        PRINT_DEBUG(nn);
        PRINT_DEBUG(rr);

        std::ostringstream oss;
        oss << "bm-pack-" << jobid << "-" << ac::mpi::get_rank(MPI_COMM_WORLD) << ".csv";
        const auto output_file{oss.str()};
        FILE*      fp{fopen(output_file.c_str(), "w")};
        ERRCHK(fp);
        fprintf(fp, "impl,dim,radius,ndims,ninputs,sample,nsamples,jobid,ns\n");
        ERRCHK(fclose(fp) == 0);

        auto print = [&](const std::string&                                label,
                         const std::vector<std::chrono::nanoseconds::rep>& results) {
            FILE* fp{fopen(output_file.c_str(), "a")};
            ERRCHK(fp);

            for (size_t i{0}; i < results.size(); ++i) {
                fprintf(fp, "%s", label.c_str());
                fprintf(fp, ",%zu", dim);
                fprintf(fp, ",%zu", radius);
                fprintf(fp, ",%zu", ndims);
                fprintf(fp, ",%zu", ninputs);
                fprintf(fp, ",%zu", i);
                fprintf(fp, ",%zu", nsamples);
                fprintf(fp, ",%zu", jobid);
                fprintf(fp, ",%lld", as<long long>(results[i]));
                fprintf(fp, "\n");
            }
            ERRCHK(fclose(fp) == 0);
        };

        // Setup the benchmark
        using T         = double;
        using Allocator = ac::mr::device_allocator;

        // Buffers
        std::vector<ac::ndbuffer<T, Allocator>> inputs;
        std::vector<ac::ndbuffer<T, Allocator>> outputs;
        std::vector<ac::ndbuffer<T, Allocator>> refs;
        for (size_t i{0}; i < ninputs; ++i) {
            inputs.push_back(ac::ndbuffer<T, Allocator>{mm});
            outputs.push_back(ac::ndbuffer<T, Allocator>{mm});
            refs.push_back(ac::ndbuffer<T, Allocator>{mm});
        }
        ERRCHK_MPI(inputs.size() == refs.size());

        // Segments
        const auto segments{prune(partition(mm, nn, rr), nn, rr)};
        // const std::vector<ac::segment> segments{8, ac::segment{ac::shape{3,3,3},
        // ac::index{0,0,0}}}; const auto segments{ac::segment{128,3,3}}; const auto
        // segments{ac::segment{128,128,3}};

        // Initialize inputs and refs
        auto init = [](std::vector<ac::ndbuffer<T, Allocator>>& inputs,
                       std::vector<ac::ndbuffer<T, Allocator>>& refs) {
            // Initialize inputs
            for (size_t i{0}; i < inputs.size(); ++i)
                std::iota(inputs[i].begin(), inputs[i].end(), i * inputs[i].size());

            // Initialize refs
            for (size_t i{0}; i < inputs.size(); ++i)
                migrate(inputs[i], refs[i]);
        };

        // Helper functions
        auto reset_outputs = [](const std::vector<ac::ndbuffer<T, Allocator>>& inputs,
                                const std::vector<ac::segment>&                segments,
                                std::vector<ac::ndbuffer<T, Allocator>>&       outputs) {
            ERRCHK_MPI(inputs.size() == outputs.size());
            for (size_t i{0}; i < inputs.size(); ++i) {
                auto tmp{inputs[i].to_host()};
                for (const auto& segment : segments)
                    ac::fill<T>(-1, segment.dims, segment.offset, tmp);
                migrate(tmp, outputs[i]);
            }
        };

        auto verify = [](const std::vector<ac::ndbuffer<T, Allocator>>& outputs,
                         const std::vector<ac::ndbuffer<T, Allocator>>& refs) {
            ERRCHK_MPI(outputs.size() == refs.size());

            for (size_t j{0}; j < outputs.size(); ++j)
                for (size_t i{0}; i < outputs[j].size(); ++i)
                    ERRCHK_MPI(within_machine_epsilon(outputs[j][i], refs[j][i]));
        };

        auto init_random = [&]() {
            for (auto& input : inputs)
                bm::randomize(input.get());
        };

        auto sync = []() {
#if defined(ACM_DEVICE_ENABLED)
            ERRCHK_CUDA_API(cudaDeviceSynchronize());
#endif
            ERRCHK_MPI_API(MPI_Barrier(MPI_COMM_WORLD));
        };

        auto display = [](const std::vector<ac::ndbuffer<T, Allocator>>& bufs) {
            for (const auto& buf : bufs)
                buf.display();
        };

        // Verify and benchmark
        const auto input_ptrs{ac::unwrap_get(inputs)};
        auto       output_ptrs{ac::unwrap_get(outputs)};

        // MPI packer
        {
            mpi_pack_strategy_batched<T, Allocator> packer{MPI_COMM_WORLD,
                                                           mm,
                                                           segments,
                                                           inputs.size()};

            auto pack   = [&packer, &input_ptrs]() { packer.pack(input_ptrs); };
            auto unpack = [&packer, &output_ptrs]() { packer.unpack(output_ptrs); };

            // Verify that the benchmarked function works correctly
            init(inputs, refs);                       // Init inputs and refs
            pack();                                   // Pack inputs
            reset_outputs(inputs, segments, outputs); // Reset outputs
            unpack();
            verify(outputs, refs);
            // Run the benchmark if verification succeeded
            print("mpi-pack", bm::benchmark(init_random, pack, sync, nsamples));
        }

        // MPI indexed packer
        {
            mpi_pack_strategy_hindexed_batched<T, Allocator> packer{MPI_COMM_WORLD,
                                                                    mm,
                                                                    segments,
                                                                    input_ptrs,
                                                                    output_ptrs};

            auto pack   = [&packer]() { packer.pack(); };
            auto unpack = [&packer]() { packer.unpack(); };

            // Verify that the benchmarked function works correctly
            init(inputs, refs);                       // Init inputs and refs
            pack();                                   // Pack inputs
            reset_outputs(inputs, segments, outputs); // Reset outputs
            unpack();
            verify(outputs, refs);
            // Run the benchmark if verification succeeded
            print("mpi-pack-indexed", bm::benchmark(init_random, pack, sync, nsamples));
        }

        // ACM
        {
            acm_pack_strategy_grouped_batched<T, Allocator> packer{segments, inputs.size()};
            auto bench = [&packer, &mm, &input_ptrs]() { packer.pack(mm, input_ptrs); };

            // Verify that the benchmarked function works correctly
            init(inputs, refs);                       // Init inputs and refs
            bench();                                  // Pack inputs
            reset_outputs(inputs, segments, outputs); // Reset outputs
            packer.unpack(mm, ac::unwrap_get(outputs));
            verify(outputs, refs);
            // Run the benchmark if verification succeeded
            print("acm-pack", bm::benchmark(init_random, bench, sync, nsamples));
        }

        // ACM packed
        {
            acm_pack_strategy_packed_batched<T, Allocator> packer{segments, inputs.size()};
            auto bench = [&packer, &mm, &input_ptrs]() { packer.pack(mm, input_ptrs); };

            // Verify that the benchmarked function works correctly
            init(inputs, refs);                       // Init inputs and refs
            bench();                                  // Pack inputs
            reset_outputs(inputs, segments, outputs); // Reset outputs
            packer.unpack(mm, ac::unwrap_get(outputs));
            verify(outputs, refs);
            // Run the benchmark if verification succeeded
            print("acm-pack-packed", bm::benchmark(init_random, bench, sync, nsamples));
        }

        // ACM packer batched
        {
            acm_pack_strategy_batched<T, Allocator> packer{segments, inputs.size()};
            auto bench = [&packer, &mm, &input_ptrs]() { packer.pack(mm, input_ptrs); };
            // Verify that the benchmarked function works correctly
            init(inputs, refs);                       // Init inputs and refs
            bench();                                  // Pack inputs
            reset_outputs(inputs, segments, outputs); // Reset outputs
            packer.unpack(mm, ac::unwrap_get(outputs));
            verify(outputs, refs);
            // Run the benchmark if verification succeeded
            print("acm-pack-batched", bm::benchmark(init_random, bench, sync, nsamples));
        }
    }
    catch (const std::exception& e) {
        ac::mpi::abort();
        return EXIT_FAILURE;
    }
    ac::mpi::finalize();
    return EXIT_SUCCESS;
}
