#pragma once

#include <mpi.h>

#include "mpi_utils.h"

// #include "buffer.h"

template <typename T, size_t N>
void
mpi_read_collective(const MPI_Comm& parent_comm, const ac::shape<N>& in_file_dims,
                    const ac::index<N>& in_file_offset, const ac::shape<N>& in_mesh_dims,
                    const ac::shape<N>& in_mesh_subdims, const ac::index<N>& in_mesh_offset,
                    const std::string& path, T* data)
{
    // Communicator
    MPI_Comm comm{MPI_COMM_NULL};
    ERRCHK_MPI_API(MPI_Comm_dup(parent_comm, &comm));

    // Info
    MPI_Info info = info_create();

    // Subarrays
    MPI_Datatype global_subarray = subarray_create(in_file_dims, in_mesh_subdims, in_file_offset,
                                                   get_mpi_dtype<T>());
    MPI_Datatype local_subarray  = subarray_create(in_mesh_dims, in_mesh_subdims, in_mesh_offset,
                                                   get_mpi_dtype<T>());

    // File
    MPI_File file{MPI_FILE_NULL};
    ERRCHK_MPI_API(MPI_File_open(comm, path.c_str(), MPI_MODE_RDONLY, info, &file));
    ERRCHK_MPI_API(MPI_File_set_view(file, 0, get_mpi_dtype<T>(), global_subarray, "native", info));

    // Check that the file is in the expected format
    MPI_Offset bytes{0};
    ERRCHK_MPI_API(MPI_File_get_size(file, &bytes));
    ERRCHK_MPI_EXPR_DESC(as<uint64_t>(bytes) == prod(in_file_dims) * sizeof(T),
                         "Tried to read a file that had unexpected file size. Ensure that "
                         "the file read/written using the same grid dimensions.");

    ERRCHK_MPI_API(MPI_File_read_all(file, data, 1, local_subarray, MPI_STATUS_IGNORE));

    ERRCHK_MPI_API(MPI_File_close(&file));
    ERRCHK_MPI_API(MPI_Type_free(&local_subarray));
    ERRCHK_MPI_API(MPI_Type_free(&global_subarray));
    ERRCHK_MPI_API(MPI_Info_free(&info));
    ERRCHK_MPI_API(MPI_Comm_free(&comm));
}

template <typename T, size_t N>
void
mpi_write_collective(const MPI_Comm& parent_comm, const ac::shape<N>& in_file_dims,
                     const ac::index<N>& in_file_offset, const ac::shape<N>& in_mesh_dims,
                     const ac::shape<N>& in_mesh_subdims, const ac::index<N>& in_mesh_offset,
                     const T* data, const std::string& path)
{
    // Communicator
    MPI_Comm comm{MPI_COMM_NULL};
    ERRCHK_MPI_API(MPI_Comm_dup(parent_comm, &comm));

    // Info
    MPI_Info info = info_create();

    // Subarrays
    MPI_Datatype global_subarray = subarray_create(in_file_dims, in_mesh_subdims, in_file_offset,
                                                   get_mpi_dtype<T>());
    MPI_Datatype local_subarray  = subarray_create(in_mesh_dims, in_mesh_subdims, in_mesh_offset,
                                                   get_mpi_dtype<T>());

    // File
    MPI_File file{MPI_FILE_NULL};
    ERRCHK_MPI_API(
        MPI_File_open(comm, path.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY, info, &file));
    ERRCHK_MPI_API(MPI_File_set_view(file, 0, get_mpi_dtype<T>(), global_subarray, "native", info));

    ERRCHK_MPI_API(MPI_File_write_all(file, data, 1, local_subarray, MPI_STATUS_IGNORE));

    ERRCHK_MPI_API(MPI_File_close(&file));
    ERRCHK_MPI_API(MPI_Type_free(&local_subarray));
    ERRCHK_MPI_API(MPI_Type_free(&global_subarray));
    ERRCHK_MPI_API(MPI_Info_free(&info));
    ERRCHK_MPI_API(MPI_Comm_free(&comm));
}

template <typename T, size_t N, typename StagingMemoryResource = PinnedHostMemoryResource>
class IOTaskAsync {
  private:
    MPI_Info info{MPI_INFO_NULL};
    MPI_Datatype global_subarray{MPI_DATATYPE_NULL};
    MPI_Datatype local_subarray{MPI_DATATYPE_NULL};

    ac::vector<T, StagingMemoryResource> staging_buffer;

    MPI_Comm comm{MPI_COMM_NULL};
    MPI_File file{MPI_FILE_NULL};
    MPI_Request req{MPI_REQUEST_NULL};

    bool in_progress{false};

  public:
    IOTaskAsync(const ac::shape<N>& in_file_dims, const ac::index<N>& in_file_offset,
                const ac::shape<N>& in_mesh_dims, const ac::shape<N>& in_mesh_subdims,
                const ac::index<N>& in_mesh_offset)
        : info{info_create()},
          global_subarray{
              subarray_create(in_file_dims, in_mesh_subdims, in_file_offset, get_mpi_dtype<T>())},
          local_subarray{
              subarray_create(in_mesh_dims, in_mesh_subdims, in_mesh_offset, get_mpi_dtype<T>())},
          staging_buffer{prod(in_mesh_dims)}
    {
    }

    ~IOTaskAsync()
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
                                 const ac::vector<T, MemoryResource>& input, const std::string& path)
    {
        ERRCHK_MPI(!in_progress);
        in_progress = true;

        // Communicator
        ERRCHK_MPI(comm == MPI_COMM_NULL);
        ERRCHK_MPI_API(MPI_Comm_dup(parent_comm, &comm));

        migrate(input, staging_buffer); // TODO: transfers the whole buffer at the moment, would be
                                        // better to migrate only in_mesh_subdims instead (but need
                                        // to pack and change in_mesh_offset to zero)

        ERRCHK_MPI(file == MPI_FILE_NULL);
        ERRCHK_MPI_API(
            MPI_File_open(comm, path.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY, info, &file));
        ERRCHK_MPI_API(
            MPI_File_set_view(file, 0, get_mpi_dtype<T>(), global_subarray, "native", info));

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

    IOTaskAsync(const IOTaskAsync&)            = delete; // Copy constructor
    IOTaskAsync& operator=(const IOTaskAsync&) = delete; // Copy assignment operator
    IOTaskAsync(IOTaskAsync&&)                 = delete; // Move constructor
    IOTaskAsync& operator=(IOTaskAsync&&)      = delete; // Move assignment operator
};
