#pragma once

#include <mpi.h>

#include "buffer.h"

template <typename T> class IOTask {
  private:
    MPICommWrapper comm;
    MPIRequestWrapper req;
    MPIFileWrapper file;

    Buffer<T, HostMemoryResource> staging_buffer;

    subarray_ptr_t global_subarray;
    subarray_ptr_t local_subarray;
    MPI_Info info{MPI_INFO_NULL};

    bool in_progress{false};

  public:
    IOTask(const Shape& in_file_dims, const Index& in_file_offset, const Shape& in_mesh_dims,
           const Shape& in_mesh_subdims, const Index& in_mesh_offset)
        : staging_buffer{prod(in_mesh_dims)},
          global_subarray{datatype_make_unique<T>(in_file_dims, in_mesh_subdims, in_file_offset)},
          local_subarray{datatype_make_unique<T>(in_mesh_dims, in_mesh_subdims, in_mesh_offset)}
    {
    }

    void read(const MPI_Comm& parent_comm, const std::string& path, T* data)
    {
        ERRCHK(!in_progress);

        // Duplicate the parent communicator
        comm = MPICommWrapper(parent_comm);

        ERRCHK_MPI_API(
            MPI_File_open(comm.value(), path.c_str(), MPI_MODE_RDONLY, info, file.get()));
        ERRCHK_MPI_API(MPI_File_set_view(file.value(), 0, get_mpi_dtype<T>(), *global_subarray,
                                         "native", info));

        MPI_Status status;
        status.MPI_ERROR = MPI_SUCCESS;
        ERRCHK_MPI_API(MPI_File_read_all(file.value(), data, 1, *local_subarray, &status));
        ERRCHK_MPI_API(status.MPI_ERROR);

        ERRCHK_MPI_API(MPI_File_close(file.get()));
    }

    void write(const MPI_Comm& parent_comm, const T* data, const std::string& path)
    {
        ERRCHK(!in_progress);

        // Duplicate the parent communicator
        comm = MPICommWrapper(parent_comm);

        ERRCHK_MPI_API(MPI_File_open(comm.value(), path.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY,
                                     info, file.get()));
        ERRCHK_MPI_API(MPI_File_set_view(file.value(), 0, get_mpi_dtype<T>(), *global_subarray,
                                         "native", info));

        MPI_Status status;
        status.MPI_ERROR = MPI_SUCCESS;
        ERRCHK_MPI_API(MPI_File_write_all(file.value(), data, 1, *local_subarray, &status));
        ERRCHK_MPI_API(status.MPI_ERROR);
        ERRCHK_MPI_API(MPI_File_close(file.get()));
    }

    template <typename MemoryResource>
    void launch_write(const MPI_Comm& parent_comm, const Buffer<T, MemoryResource>& input,
                      const std::string& path)
    {
        ERRCHK_MPI_EXPR_DESC(!in_progress, "Previous IO operation was still in progress.");
        in_progress = true;

        // Duplicate the parent communicator
        comm = MPICommWrapper(parent_comm);

        // Migrate input to the staging buffer
        migrate(input, staging_buffer);

        ERRCHK_MPI_API(MPI_File_open(comm.value(), path.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY,
                                     info, file.get()));
        ERRCHK_MPI_API(MPI_File_set_view(file.value(), 0, get_mpi_dtype<T>(), *global_subarray,
                                         "native", info));
        ERRCHK_MPI_API(MPI_File_iwrite_all(file.value(), staging_buffer.data(), 1, *local_subarray,
                                           req.get()));
    }

    void wait_write()
    {
        ERRCHK_MPI_EXPR_DESC(file.value() != MPI_FILE_NULL,
                             "Function called but there was no write operation in progress. "
                             "Function launch_write must be called before wait_write.");
        ERRCHK_MPI_EXPR_DESC(!req.complete(),
                             "Function called but there was no write operation in progress. "
                             "Function launch_write must be called before wait_write.");
        req.wait();
        ERRCHK_MPI_API(MPI_File_close(file.get()));
        ERRCHK_MPI(file.value() == MPI_FILE_NULL);

        // Ensure all processess have written their segment to disk.
        // Not strictly needed here and comes with a slight overhead
        // However, will cause hard-to-trace issues if the reader
        // tries to access the data without barrier.
        ERRCHK_MPI_API(MPI_Barrier(comm.value()));
        in_progress = false;
    }
};
