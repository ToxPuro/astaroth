#pragma once

#include <mpi.h>

#include "mpi_utils.h"

#include "buffer.h"

namespace ac::io {

template <typename T, typename StagingMemoryResource = ac::mr::pinned_host_memory_resource>
class AsyncWriteTask {
  private:
    MPI_Info m_info{MPI_INFO_NULL};
    MPI_Datatype m_global_subarray{MPI_DATATYPE_NULL};
    MPI_Datatype m_local_subarray{MPI_DATATYPE_NULL};

    ac::buffer<T, StagingMemoryResource> m_staging_buffer;

    MPI_Comm m_comm{MPI_COMM_NULL};
    MPI_File m_file{MPI_FILE_NULL};
    MPI_Request m_req{MPI_REQUEST_NULL};

    bool m_in_progress{false};

  public:
    AsyncWriteTask(const Shape& file_dims, const Index& file_offset, const Shape& mesh_dims,
                   const Shape& mesh_subdims, const Index& mesh_offset)
        : m_info{ac::mpi::info_create()},
          m_global_subarray{ac::mpi::subarray_create(file_dims, mesh_subdims, file_offset,
                                                     ac::mpi::get_dtype<T>())},
          m_local_subarray{ac::mpi::subarray_create(mesh_dims, mesh_subdims, mesh_offset,
                                                    ac::mpi::get_dtype<T>())},
          m_staging_buffer{prod(mesh_dims)}
    {
    }

    ~AsyncWriteTask()
    {
        ERRCHK_MPI(!m_in_progress);

        ERRCHK_MPI(m_req == MPI_REQUEST_NULL);
        if (m_req != MPI_REQUEST_NULL)
            ERRCHK_MPI_API(MPI_Request_free(&m_req));

        ERRCHK_MPI(m_file == MPI_FILE_NULL);
        if (m_file != MPI_FILE_NULL)
            ERRCHK_MPI_API(MPI_File_close(&m_file));

        ERRCHK_MPI(m_comm == MPI_COMM_NULL);
        if (m_comm != MPI_COMM_NULL)
            ERRCHK_MPI_API(MPI_Comm_free(&m_comm));

        ERRCHK_MPI_API(MPI_Type_free(&m_local_subarray));
        ERRCHK_MPI_API(MPI_Type_free(&m_global_subarray));
        ERRCHK_MPI_API(MPI_Info_free(&m_info));
    }

    template <typename MemoryResource>
    void launch_write_collective(const MPI_Comm& parent_m_comm,
                                 const ac::mr::pointer<T, MemoryResource>& input,
                                 const std::string& path)
    {
        ERRCHK_MPI(!m_in_progress);
        m_in_progress = true;

        // Communicator
        ERRCHK_MPI(m_comm == MPI_COMM_NULL);
        ERRCHK_MPI_API(MPI_Comm_dup(parent_m_comm, &m_comm));

        // TODO: transfers the whole buffer at the moment, would be
        // better to migrate only mesh_subdims instead (but need
        // to pack and change mesh_offset to zero)
        // migrate(input, m_staging_buffer);
        ac::mr::copy<T>(input, m_staging_buffer.get());

        ERRCHK_MPI(m_file == MPI_FILE_NULL);
        ERRCHK_MPI_API(MPI_File_open(m_comm,
                                     path.c_str(),
                                     MPI_MODE_CREATE | MPI_MODE_WRONLY,
                                     m_info,
                                     &m_file));
        ERRCHK_MPI_API(MPI_File_set_view(m_file,
                                         0,
                                         ac::mpi::get_dtype<T>(),
                                         m_global_subarray,
                                         "native",
                                         m_info));

        ERRCHK_MPI(m_file != MPI_FILE_NULL);
        ERRCHK_MPI(m_req == MPI_REQUEST_NULL);
        ERRCHK_MPI_API(
            MPI_File_iwrite_all(m_file, m_staging_buffer.data(), 1, m_local_subarray, &m_req));
    }

    void wait_write_collective()
    {
        ERRCHK_MPI(m_in_progress);
        ERRCHK_MPI_API(MPI_Wait(&m_req, MPI_STATUS_IGNORE));
        ERRCHK_MPI_API(MPI_File_close(&m_file));

        // Ensure all processess have written their segment to disk.
        // Not strictly needed here and comes with a slight overhead
        // However, will cause hard-to-trace issues if the reader
        // tries to access the data without barrier.
        ERRCHK_MPI(m_comm != MPI_COMM_NULL);
        ERRCHK_MPI_API(MPI_Barrier(m_comm));
        ERRCHK_MPI_API(MPI_Comm_free(&m_comm));

        // Check that the MPI implementation reset the resources
        ERRCHK_MPI(m_req == MPI_REQUEST_NULL);
        ERRCHK_MPI(m_file == MPI_FILE_NULL);
        ERRCHK_MPI(m_comm == MPI_COMM_NULL);

        // Complete
        m_in_progress = false;
    }

    bool complete() const { return !m_in_progress; };

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
    BatchedAsyncWriteTask() = default;

    BatchedAsyncWriteTask(const Shape& file_dims, const Index& file_offset, const Shape& mesh_dims,
                          const Shape& mesh_subdims, const Index& mesh_offset,
                          const size_t n_aggregate_buffers)
    {
        for (size_t i = 0; i < n_aggregate_buffers; ++i)
            write_tasks.push_back(
                std::make_unique<AsyncWriteTask<T, StagingMemoryResource>>(file_dims,
                                                                           file_offset,
                                                                           mesh_dims,
                                                                           mesh_subdims,
                                                                           mesh_offset));
    }

    template <typename MemoryResource>
    void launch(const MPI_Comm& parent_m_comm,
                const std::vector<ac::mr::pointer<T, MemoryResource>>& inputs,
                const std::vector<std::string>& paths)
    {
        for (size_t i = 0; i < inputs.size(); ++i)
            write_tasks[i]->launch_write_collective(parent_m_comm, inputs[i], paths[i]);
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
