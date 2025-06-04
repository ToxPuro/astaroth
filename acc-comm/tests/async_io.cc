#include <cstdio>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <mpi.h>
#include <numeric>

#include <future>

#include "acm/detail/allocator.h"
#include "acm/detail/buffer.h"
#include "acm/detail/errchk.h"
#include "acm/detail/errchk_mpi.h"
#include "acm/detail/mpi_utils.h"
#include "acm/detail/ndbuffer.h"
#include "acm/detail/pack.h"
#include "acm/detail/type_conversion.h"
#include "acm/detail/view.h"

template <typename T>
static int
write(const ac::host_ndbuffer<T>&& input, const ac::shape& file_dims, const ac::index& file_offset,
      const std::string& outfile)
{
    FILE* fp{fopen(outfile.c_str(), "r+")};
    if (!fp)
        return -1;

    for (uint64_t i{0}; i < input.size(); ++i) {

        const auto in_coords{to_spatial(i, input.shape())};
        const auto out_coords{file_offset + in_coords};
        const auto out_idx{to_linear(out_coords, file_dims)};

        const long offset_bytes{as<long>(out_idx * sizeof(T))};
        if (fseek(fp, offset_bytes, SEEK_SET) != 0) {
            fclose(fp);
            return -1;
        }

        if (fwrite(&input[i], sizeof(T), 1, fp) != 1) {
            fclose(fp);
            return -1;
        }
    }

    fclose(fp);
    return 0;
}

template <typename T, typename Allocator>
static int
async_write_pipeline(const int device_id, const ac::ndbuffer<T, Allocator>&& staging_buffer, //
                     const ac::shape file_dims, const ac::index file_offset,
                     const std::string outfile)
{
#if defined(ACM_DEVICE_ENABLED)
    ERRCHK_CUDA_API(cudaSetDevice(device_id));
#else
    (void)device_id; // Unused
#endif

    ac::host_ndbuffer<T> hbuf{staging_buffer.to_host()};
    write(std::move(hbuf), file_dims, file_offset, outfile);
    return 0;
}

/** Note: data must remain valid until the future has completed. */
template <typename T, typename Allocator>
static std::future<int>
write_async(const MPI_Comm& parent_comm, const int device_id, const ac::shape mm,
            const ac::shape nn, const ac::index nn_offset, const ac::view<T, Allocator>&& input, //
            const ac::shape file_dims, const ac::index file_offset, const std::string outfile)
{
    if (parent_comm != MPI_COMM_NULL) {
        MPI_Comm comm{MPI_COMM_NULL};
        ERRCHK_MPI_API(MPI_Comm_dup(parent_comm, &comm));

        // Create the file for concurrent writing
        ERRCHK_MPI_API(MPI_Barrier(comm));

        const int root{0};
        if (ac::mpi::get_rank(comm) == root) {
            FILE* fp{fopen(outfile.c_str(), "w")};
            ERRCHK_MPI(fp);
            ERRCHK_MPI(fclose(fp) == 0);
        }

        ERRCHK_MPI_API(MPI_Barrier(comm));
        ERRCHK_MPI_API(MPI_Comm_free(&comm));

        ac::ndbuffer<T, Allocator> staging_buffer{nn};
        acm::pack(mm, nn, nn_offset, {input}, staging_buffer.get());
        return std::future<int>{std::async(std::launch::async,
                                           async_write_pipeline<T, Allocator>,
                                           device_id,
                                           std::move(staging_buffer),
                                           file_dims,
                                           file_offset,
                                           outfile)};
    }
    else {
        return std::future<int>{std::async(std::launch::async, []() { return 0; })};
    }
}

static int
wait(const MPI_Comm& comm, std::future<int>& task)
{
    ERRCHK(task.get() == 0);
    ERRCHK_MPI_API(MPI_Barrier(comm));
    return 0;
}

int
main()
{
    ac::mpi::init_funneled();
    try {

        const ac::shape global_nn{128};
        MPI_Comm        cart_comm{ac::mpi::cart_comm_create(MPI_COMM_WORLD, global_nn)};
        const auto      rank{ac::mpi::get_rank(cart_comm)};

        const ac::shape             local_nn{ac::mpi::get_local_nn(cart_comm, global_nn)};
        ac::host_ndbuffer<uint64_t> buf{local_nn};
        std::iota(buf.begin(), buf.end(), as<size_t>(rank) * prod(local_nn));
        // buf.display();

        const std::string outfile{"test.out"};
        auto              task{write_async(cart_comm,
                              0,
                              local_nn,
                              local_nn,
                              ac::make_index(local_nn.size(), 0),
                              buf.get(),
                              global_nn,
                              ac::mpi::get_coords(cart_comm) * local_nn,
                              outfile)};
        ERRCHK_MPI(wait(cart_comm, task) == 0);

        if (ac::mpi::get_rank(cart_comm) == 0) {
            FILE* fp{fopen(outfile.c_str(), "r")};
            ERRCHK_MPI(fp);

            ac::host_ndbuffer<uint64_t> gbuf{global_nn};
            ERRCHK(fread(gbuf.data(), sizeof(gbuf[0]), gbuf.size(), fp) == gbuf.size());

            // gbuf.display();
            for (size_t i{0}; i < gbuf.size(); ++i)
                ERRCHK_MPI(gbuf[i] == as<uint64_t>(i));

            ERRCHK_MPI(fclose(fp) == 0);
        }

        ac::mpi::cart_comm_destroy(&cart_comm);
        ac::mpi::finalize();
        return EXIT_SUCCESS;
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        ac::mpi::abort();
        return EXIT_FAILURE;
    }
}

/*
static std::future<int>
write_profile_to_disk_async(const MPI_Comm& cart_comm, const Device& device, const Profile& profile,
                            const size_t step)
{
    VertexBufferArray vba{};
    ERRCHK_AC(acDeviceGetVBA(device, &vba));

    AcMeshInfo info{};
    ERRCHK_AC(acDeviceGetLocalConfig(device, &info));

    const auto global_nn{acr::get_global_nn(info)};
    const auto coords{ac::mpi::get_coords(cart_comm)};

    // Delegate one of the processes as the file creator
    ERRCHK_MPI_API(MPI_Barrier(cart_comm));
    char outfile[4096];
    sprintf(outfile, "%s-%012zu.profile", profile_names[profile], step);
    FILE* fp{fopen(outfile, "w")};
    ERRCHK_MPI(fp);
    fclose(fp);
    ERRCHK_MPI_API(MPI_Barrier(cart_comm));

    if ((coords[0] == 0) && (coords[1] == 0)) {

        const auto global_nn_offset{acr::get_global_nn_offset(info)};
        const auto local_mm{acr::get_local_mm(info)};
        const auto local_nn{acr::get_local_nn(info)};
        const auto rr{acr::get_local_rr()};
        ERRCHK(global_nn_offset == ac::mpi::get_global_nn_offset(cart_comm, global_nn));

        PRINT_DEBUG(ac::mpi::get_coords(cart_comm));
        PRINT_DEBUG(global_nn_offset);
        PRINT_DEBUG(ac::mpi::get_decomposition(cart_comm));

        ac::device_buffer<double> staging_buffer{local_nn[2]};
        acm::pack(ac::shape{local_mm[2]},
             ac::shape{local_nn[2]},
             ac::index{rr[2]},
             {acr::make_ptr(vba, profile, BufferGroup::input)},
             staging_buffer.get());

        auto write_to_file = [](const int device_id,
                                const ac::device_buffer<double>&& dbuf,
                                const uint64_t file_offset,
                                const std::string outfile) {
            ERRCHK_CUDA_API(cudaSetDevice(device_id));
            const auto buf{dbuf.to_host()};

            FILE* fp{fopen(outfile.c_str(), "r+")};
            ERRCHK_MPI(fp);

            const long offset_bytes{as<long>(file_offset * sizeof(buf[0]))};
            ERRCHK_MPI(fseek(fp, offset_bytes, SEEK_SET) == 0);

            const size_t count{buf.size()};
            const size_t res{fwrite(buf.data(), sizeof(buf[0]), buf.size(), fp)};
            ERRCHK_MPI(res == count);

            fclose(fp);
            return 0;
        };

        int id{-1};
        ERRCHK_AC(acDeviceGetId(device, &id));
        std::future<int> task{std::async(std::launch::async,
                                         write_to_file,
                                         id,
                                         std::move(staging_buffer),
                                         global_nn_offset[2],
                                         std::string(outfile))};
        return task;
    }
    else {
        return std::future<int>{std::async(std::launch::async, []() { return 0; })};
    }
}
    */
