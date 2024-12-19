#pragma once

#include <mpi.h>

#include "mpi_utils.h"

#include "buffer.h"

namespace ac::io {

template <typename T, typename StagingMemoryResource = ac::mr::pinned_host_memory_resource>
class AsyncWriteTask {
  private:
    MPI_Info info{MPI_INFO_NULL};
    MPI_Datatype global_subarray{MPI_DATATYPE_NULL};
    MPI_Datatype local_subarray{MPI_DATATYPE_NULL};

    ac::buffer<T, StagingMemoryResource> staging_buffer;

    MPI_Comm comm{MPI_COMM_NULL};
    MPI_File file{MPI_FILE_NULL};
    MPI_Request req{MPI_REQUEST_NULL};

    bool in_progress{false};

  public:
    AsyncWriteTask(const Shape& in_file_dims, const Index& in_file_offset,
                   const Shape& in_mesh_dims, const Shape& in_mesh_subdims,
                   const Index& in_mesh_offset)
        : info{ac::mpi::info_create()},
          global_subarray{ac::mpi::subarray_create(in_file_dims, in_mesh_subdims, in_file_offset,
                                                   ac::mpi::get_dtype<T>())},
          local_subarray{ac::mpi::subarray_create(in_mesh_dims, in_mesh_subdims, in_mesh_offset,
                                                  ac::mpi::get_dtype<T>())},
          staging_buffer{prod(in_mesh_dims)}
    {
    }

    ~AsyncWriteTask()
    {
        ERRCHK_MPI(!in_progress);

        ERRCHK_MPI(req == MPI_REQUEST_NULL);
        if (req != MPI_REQUEST_NULL)
            ERRCHK_MPI_API(MPI_Request_free(&req));

        ERRCHK_MPI(file == MPI_FILE_NULL);
        if (file != MPI_FILE_NULL)
            ERRCHK_MPI_API(MPI_File_close(&file));

        ERRCHK_MPI(comm == MPI_COMM_NULL);
        if (comm != MPI_COMM_NULL)
            ERRCHK_MPI_API(MPI_Comm_free(&comm));

        ERRCHK_MPI_API(MPI_Type_free(&local_subarray));
        ERRCHK_MPI_API(MPI_Type_free(&global_subarray));
        ERRCHK_MPI_API(MPI_Info_free(&info));
    }

    template <typename MemoryResource>
    void launch_write_collective(const MPI_Comm& parent_comm,
                                 const ac::mr::base_ptr<T, MemoryResource>& input,
                                 const std::string& path)
    {
        ERRCHK_MPI(!in_progress);
        in_progress = true;

        // Communicator
        ERRCHK_MPI(comm == MPI_COMM_NULL);
        ERRCHK_MPI_API(MPI_Comm_dup(parent_comm, &comm));

        // TODO: transfers the whole buffer at the moment, would be
        // better to migrate only in_mesh_subdims instead (but need
        // to pack and change in_mesh_offset to zero)
        // migrate(input, staging_buffer);
        ac::mr::copy<T>(input, staging_buffer.get());

        ERRCHK_MPI(file == MPI_FILE_NULL);
        ERRCHK_MPI_API(
            MPI_File_open(comm, path.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY, info, &file));
        ERRCHK_MPI_API(
            MPI_File_set_view(file, 0, ac::mpi::get_dtype<T>(), global_subarray, "native", info));

        ERRCHK_MPI(file != MPI_FILE_NULL);
        ERRCHK_MPI(req == MPI_REQUEST_NULL);
        ERRCHK_MPI_API(MPI_File_iwrite_all(file, staging_buffer.data(), 1, local_subarray, &req));
    }

    void wait_write_collective()
    {
        ERRCHK_MPI(in_progress);
        ERRCHK_MPI_API(MPI_Wait(&req, MPI_STATUS_IGNORE));
        ERRCHK_MPI_API(MPI_File_close(&file));

        // Ensure all processess have written their segment to disk.
        // Not strictly needed here and comes with a slight overhead
        // However, will cause hard-to-trace issues if the reader
        // tries to access the data without barrier.
        ERRCHK_MPI_API(MPI_Barrier(comm));
        ERRCHK_MPI_API(MPI_Comm_free(&comm));

        // Check that the MPI implementation reset the resources
        ERRCHK_MPI(req == MPI_REQUEST_NULL);
        ERRCHK_MPI(file == MPI_FILE_NULL);
        ERRCHK_MPI(comm == MPI_COMM_NULL);

        // Complete
        in_progress = false;
    }

    bool complete() const { return !in_progress; };

    AsyncWriteTask(const AsyncWriteTask&)            = delete; // Copy constructor
    AsyncWriteTask& operator=(const AsyncWriteTask&) = delete; // Copy assignment operator
    AsyncWriteTask(AsyncWriteTask&&)                 = delete; // Move constructor
    AsyncWriteTask& operator=(AsyncWriteTask&&)      = delete; // Move assignment operator
};

template <typename T, typename StagingMemoryResource = ac::mr::pinned_host_memory_resource>
class BatchedAsyncWriteTask {
  private:
    std::vector<std::unique_ptr<AsyncWriteTask<T, StagingMemoryResource>>> write_tasks{};

  public:
    BatchedAsyncWriteTask(const Shape& in_file_dims, const Index& in_file_offset,
                          const Shape& in_mesh_dims, const Shape& in_mesh_subdims,
                          const Index& in_mesh_offset, const size_t n_aggregate_buffers)
    {
        for (size_t i = 0; i < n_aggregate_buffers; ++i)
            write_tasks.push_back(
                std::make_unique<AsyncWriteTask<T, StagingMemoryResource>>(in_file_dims,
                                                                           in_file_offset,
                                                                           in_mesh_dims,
                                                                           in_mesh_subdims,
                                                                           in_mesh_offset));
    }

    template <typename MemoryResource>
    void launch(const MPI_Comm& parent_comm,
                const std::vector<ac::mr::base_ptr<T, MemoryResource>>& inputs,
                const std::vector<std::string>& paths)
    {
        for (size_t i = 0; i < inputs.size(); ++i)
            write_tasks[i]->launch_write_collective(parent_comm, inputs[i], paths[i]);
    }

    void wait()
    {
        for (auto& ptr : write_tasks)
            ptr->wait_write_collective();
    }

    bool complete() const
    {
        return std::all_of(write_tasks.begin(),
                           write_tasks.end(),
                           std::mem_fn(&AsyncWriteTask<T, StagingMemoryResource>::complete));
    }
};

} // namespace ac::io
