#pragma once

#include <mpi.h>

#include "buffer.h"
#include "mpi_utils.h"

// TODO
// 1) move the data from the input buffer and return in iwrite
// 2) remove redundancy
// 3) ensure correctness with arbitrary call patterns
//
// MPI IO hints: pg. 648 in MPI 4.1 specification

// Note: the same parent_comm must be used throughout:
// this is important especially for the barrier: required
// to be sure that all processes have written their result out to disk
template <typename T> class IOTask {

  private:
    Shape file_dims;
    Index file_offset;

    Shape mesh_dims;
    Shape mesh_subdims;
    Index mesh_offset;

    Buffer<T> staging_buffer; // Buffer used for IO

    MPI_Request req;
    MPI_File file;

  public:
    IOTask(const Shape& in_file_dims, const Index& in_file_offset, const Shape& in_mesh_dims,
           const Shape& in_mesh_subdims, const Index& in_mesh_offset)
        : file_dims(in_file_dims),
          file_offset(in_file_offset),
          mesh_dims(in_mesh_dims),
          mesh_subdims(in_mesh_subdims),
          mesh_offset(in_mesh_offset),
          staging_buffer(prod(in_mesh_dims)),
          req(MPI_REQUEST_NULL),
          file(MPI_FILE_NULL) {};

    void read(const MPI_Comm cart_comm, const std::string& path, T* data)
    {
        MPI_Info info = MPI_INFO_NULL;
        ERRCHK_MPI_API(MPI_Info_create(&info));
        ERRCHK_MPI_API(MPI_Info_set(info, "blocksize", "4096"));
        ERRCHK_MPI_API(MPI_Info_set(info, "striping_factor", "4"));
        // ERRCHK_MPI_API(MPI_Info_set(info, "striping_unit", "...")); // Size of stripe chunks
        // ERRCHK_MPI_API(MPI_Info_set(info, "cb_buffer_size", "...")); // Collective buffer
        // size ERRCHK_MPI_API(MPI_Info_set(info, "romio_ds_read", "...")); // Data sieving
        // ERRCHK_MPI_API(MPI_Info_set(info, "romio_ds_write", "...")); // Data sieving
        // ERRCHK_MPI_API(MPI_Info_set(info, "romio_cb_read", "...")); // Collective buffering
        // ERRCHK_MPI_API(MPI_Info_set(info, "romio_cb_write", "...")); // Collective buffering
        // ERRCHK_MPI_API(MPI_Info_set(info, "romio_no_indep_rw", "...")); // Enable/disable
        // independent rw

        MPI_Datatype global_subarray = create_subarray(file_dims, mesh_subdims, file_offset,
                                                       get_mpi_dtype<T>());
        MPI_Datatype local_subarray  = create_subarray(mesh_dims, mesh_subdims, mesh_offset,
                                                       get_mpi_dtype<T>());

        ERRCHK_MPI_EXPR_DESC(file == MPI_FILE_NULL, "Previous IO operation was still in progress.");
        ERRCHK_MPI_API(MPI_File_open(cart_comm, path.c_str(), MPI_MODE_RDONLY, info, &file));
        ERRCHK_MPI_API(
            MPI_File_set_view(file, 0, get_mpi_dtype<T>(), global_subarray, "native", info));

        MPI_Offset bytes;
        ERRCHK_MPI_API(MPI_File_get_size(file, &bytes));
        ERRCHK_MPI_EXPR_DESC(as<uint64_t>(bytes) == prod(file_dims) * sizeof(T),
                             "Tried to read a file that had unexpected file size. Ensure that the "
                             "file read/written using the same grid dimensions.");

        MPI_Status status;
        status.MPI_ERROR = MPI_SUCCESS;
        ERRCHK_MPI_API(MPI_File_read_all(file, data, 1, local_subarray, &status));
        ERRCHK_MPI_API(status.MPI_ERROR);
        ERRCHK_MPI_API(MPI_File_close(&file));
        ERRCHK_MPI(file == MPI_FILE_NULL);

        ERRCHK_MPI_API(MPI_Type_free(&local_subarray));
        ERRCHK_MPI_API(MPI_Type_free(&global_subarray));
        ERRCHK_MPI_API(MPI_Info_free(&info));
    };

    void write(const MPI_Comm cart_comm, const T* data, const std::string& path)
    {
        MPI_Info info = MPI_INFO_NULL;
        ERRCHK_MPI_API(MPI_Info_create(&info));
        ERRCHK_MPI_API(MPI_Info_set(info, "blocksize", "4096"));
        ERRCHK_MPI_API(MPI_Info_set(info, "striping_factor", "4"));
        // ERRCHK_MPI_API(MPI_Info_set(info, "striping_unit", "...")); // Size of stripe chunks
        // ERRCHK_MPI_API(MPI_Info_set(info, "cb_buffer_size", "...")); // Collective buffer
        // size ERRCHK_MPI_API(MPI_Info_set(info, "romio_ds_read", "...")); // Data sieving
        // ERRCHK_MPI_API(MPI_Info_set(info, "romio_ds_write", "...")); // Data sieving
        // ERRCHK_MPI_API(MPI_Info_set(info, "romio_cb_read", "...")); // Collective buffering
        // ERRCHK_MPI_API(MPI_Info_set(info, "romio_cb_write", "...")); // Collective buffering
        // ERRCHK_MPI_API(MPI_Info_set(info, "romio_no_indep_rw", "...")); // Enable/disable
        // independent rw

        MPI_Datatype global_subarray = create_subarray(file_dims, mesh_subdims, file_offset,
                                                       get_mpi_dtype<T>());
        MPI_Datatype local_subarray  = create_subarray(mesh_dims, mesh_subdims, mesh_offset,
                                                       get_mpi_dtype<T>());

        ERRCHK_MPI_EXPR_DESC(file == MPI_FILE_NULL, "Previous IO operation was still in progress.");
        ERRCHK_MPI_API(
            MPI_File_open(cart_comm, path.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY, info, &file));
        ERRCHK_MPI_API(
            MPI_File_set_view(file, 0, get_mpi_dtype<T>(), global_subarray, "native", info));

        MPI_Status status;
        status.MPI_ERROR = MPI_SUCCESS;
        ERRCHK_MPI_API(MPI_File_write_all(file, data, 1, local_subarray, &status));
        ERRCHK_MPI_API(status.MPI_ERROR);
        ERRCHK_MPI_API(MPI_File_close(&file));
        ERRCHK_MPI(file == MPI_FILE_NULL);

        ERRCHK_MPI_API(MPI_Type_free(&local_subarray));
        ERRCHK_MPI_API(MPI_Type_free(&global_subarray));
        ERRCHK_MPI_API(MPI_Info_free(&info));
    }

    void launch_write(const MPI_Comm cart_comm, const T* data, const std::string& path)
    {
        MPI_Info info = MPI_INFO_NULL;
        ERRCHK_MPI_API(MPI_Info_create(&info));
        ERRCHK_MPI_API(MPI_Info_set(info, "blocksize", "4096"));
        ERRCHK_MPI_API(MPI_Info_set(info, "striping_factor", "4"));
        // ERRCHK_MPI_API(MPI_Info_set(info, "striping_unit", "...")); // Size of stripe chunks
        // ERRCHK_MPI_API(MPI_Info_set(info, "cb_buffer_size", "...")); // Collective buffer
        // size ERRCHK_MPI_API(MPI_Info_set(info, "romio_ds_read", "...")); // Data sieving
        // ERRCHK_MPI_API(MPI_Info_set(info, "romio_ds_write", "...")); // Data sieving
        // ERRCHK_MPI_API(MPI_Info_set(info, "romio_cb_read", "...")); // Collective buffering
        // ERRCHK_MPI_API(MPI_Info_set(info, "romio_cb_write", "...")); // Collective buffering
        // ERRCHK_MPI_API(MPI_Info_set(info, "romio_no_indep_rw", "...")); // Enable/disable
        // independent rw

        MPI_Datatype global_subarray = create_subarray(file_dims, mesh_subdims, file_offset,
                                                       get_mpi_dtype<T>());
        MPI_Datatype local_subarray  = create_subarray(mesh_dims, mesh_subdims, mesh_offset,
                                                       get_mpi_dtype<T>());

        ERRCHK_MPI(file == MPI_FILE_NULL);

        // TODO: copy input to the staging buffer

        ERRCHK_MPI_API(
            MPI_File_open(cart_comm, path.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY, info, &file));
        ERRCHK_MPI_API(
            MPI_File_set_view(file, 0, get_mpi_dtype<T>(), global_subarray, "native", info));
        ERRCHK_MPI_API(MPI_File_iwrite_all(file, data, 1, local_subarray, &req));
        ERRCHK_MPI_API(MPI_Type_free(&local_subarray));
        ERRCHK_MPI_API(MPI_Type_free(&global_subarray));
        ERRCHK_MPI_API(MPI_Info_free(&info));
    };

    void wait_write()
    {
        ERRCHK_MPI_EXPR_DESC(file != MPI_FILE_NULL,
                             "Function called but there was no write operation in progress. "
                             "Function launch_write must be called before wait_write.");
        ERRCHK_MPI_EXPR_DESC(req != MPI_REQUEST_NULL,
                             "Function called but there was no write operation in progress. "
                             "Function launch_write must be called before wait_write.");
        wait_and_destroy_request(req);
        ERRCHK_MPI_API(MPI_File_close(&file));
        ERRCHK_MPI(file == MPI_FILE_NULL);
    };

    /** Ensures that all processes have written their segment to disk. */
    void barrier(const MPI_Comm cart_comm) { MPI_Barrier(cart_comm); }
};
