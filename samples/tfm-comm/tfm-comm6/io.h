#pragma once

#include <mpi.h>

#include "mpi_utils.h"

// TODO
// 1) move the data from the input buffer and return in iwrite
// 2) remove redundancy
// 3) ensure correctness with arbitrary call patterns

template <typename T> struct IOTask {

    Shape file_dims;
    Index file_offset;

    Shape mesh_dims;
    Shape mesh_subdims;
    Index mesh_offset;

    std::unique_ptr<Buffer<T>> buffer; // Buffer used for IO

    MPI_Request req;
    MPI_File file;

    IOTask(const Shape& file_dims, const Index& file_offset, const Shape& mesh_dims,
           const Shape& mesh_subdims, const Index& mesh_offset)
        : file_dims(file_dims),
          file_offset(file_offset),
          mesh_dims(mesh_dims),
          mesh_subdims(mesh_subdims),
          mesh_offset(mesh_offset),
          buffer(std::make_unique<Buffer<T>>(prod(mesh_dims))),
          req(MPI_REQUEST_NULL),
          file(MPI_FILE_NULL) {};

    void read(const MPI_Comm cart_comm, const std::string& path, T* data)
    {
        MPI_Info info;
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
                                                       get_dtype<T>());
        MPI_Datatype local_subarray  = create_subarray(mesh_dims, mesh_subdims, mesh_offset,
                                                       get_dtype<T>());

        ERRCHK(file == MPI_FILE_NULL);
        ERRCHK_MPI_API(MPI_File_open(cart_comm, path.c_str(), MPI_MODE_RDONLY, info, &file));
        ERRCHK_MPI_API(MPI_File_set_view(file, 0, get_dtype<T>(), global_subarray, "native", info));
        MPI_Status status = {.MPI_ERROR = MPI_SUCCESS};
        ERRCHK_MPI_API(MPI_File_read_all(file, data, 1, local_subarray, &status));
        ERRCHK_MPI_API(status.MPI_ERROR);
        ERRCHK_MPI_API(MPI_File_close(&file));
        ERRCHK(file == MPI_FILE_NULL);

        ERRCHK_MPI_API(MPI_Type_free(&local_subarray));
        ERRCHK_MPI_API(MPI_Type_free(&global_subarray));
        ERRCHK_MPI_API(MPI_Info_free(&info));
    };

    void write(const MPI_Comm cart_comm, const std::string& path, const T* data)
    {
        MPI_Info info;
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
                                                       get_dtype<T>());
        MPI_Datatype local_subarray  = create_subarray(mesh_dims, mesh_subdims, mesh_offset,
                                                       get_dtype<T>());

        ERRCHK(file == MPI_FILE_NULL);
        ERRCHK_MPI_API(
            MPI_File_open(cart_comm, path.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY, info, &file));
        ERRCHK_MPI_API(MPI_File_set_view(file, 0, get_dtype<T>(), global_subarray, "native", info));
        ERRCHK_MPI_API(MPI_File_write_all(file, data, 1, local_subarray, MPI_STATUS_IGNORE));
        ERRCHK_MPI_API(MPI_File_close(&file));
        ERRCHK(file == MPI_FILE_NULL);

        ERRCHK_MPI_API(MPI_Type_free(&local_subarray));
        ERRCHK_MPI_API(MPI_Type_free(&global_subarray));
        ERRCHK_MPI_API(MPI_Info_free(&info));
    }

    void launch_write(const MPI_Comm cart_comm, const std::string& path, const T* data)
    {
        MPI_Info info;
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
                                                       get_dtype<T>());
        MPI_Datatype local_subarray  = create_subarray(mesh_dims, mesh_subdims, mesh_offset,
                                                       get_dtype<T>());

        ERRCHK(file == MPI_FILE_NULL);
        ERRCHK_MPI_API(
            MPI_File_open(cart_comm, path.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY, info, &file));
        ERRCHK_MPI_API(MPI_File_set_view(file, 0, get_dtype<T>(), global_subarray, "native", info));
        ERRCHK_MPI_API(MPI_File_iwrite_all(file, data, 1, local_subarray, &req));
        ERRCHK_MPI_API(MPI_Type_free(&local_subarray));
        ERRCHK_MPI_API(MPI_Type_free(&global_subarray));
        ERRCHK_MPI_API(MPI_Info_free(&info));
    };

    void wait_write()
    {
        wait_request(req);
        ERRCHK_MPI_API(MPI_File_close(&file));
        ERRCHK(file == MPI_FILE_NULL);
    };
};
