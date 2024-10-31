#pragma once

#include <mpi.h>

#include "mpi_utils.h"

// TODO
// 1) move the data from the input buffer and return in iwrite
// 2) remove redundancy
// 3) ensure correctness with arbitrary call patterns
//
// MPI IO hints: pg. 648 in MPI 4.1 specification

template <typename T> class IOTask {

  private:
    Shape m_file_dims;
    Index m_file_offset;

    Shape m_mesh_dims;
    Shape m_mesh_subdims;
    Index m_mesh_offset;

    std::unique_ptr<Buffer<T>> m_staging_buffer; // Buffer used for IO

    MPI_Request m_req;
    MPI_File m_file;

  public:
    IOTask(const Shape& file_dims, const Index& file_offset, const Shape& mesh_dims,
           const Shape& mesh_subdims, const Index& mesh_offset)
        : m_file_dims(file_dims),
          m_file_offset(file_offset),
          m_mesh_dims(mesh_dims),
          m_mesh_subdims(mesh_subdims),
          m_mesh_offset(mesh_offset),
          m_staging_buffer(std::make_unique<Buffer<T>>(prod(mesh_dims))),
          m_req(MPI_REQUEST_NULL),
          m_file(MPI_FILE_NULL) {};

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

        MPI_Datatype global_subarray = create_subarray(m_file_dims, m_mesh_subdims, m_file_offset,
                                                       get_dtype<T>());
        MPI_Datatype local_subarray  = create_subarray(m_mesh_dims, m_mesh_subdims, m_mesh_offset,
                                                       get_dtype<T>());

        ERRCHK_MPI_EXPR_DESC(m_file == MPI_FILE_NULL,
                             "Previous IO operation was still in progress.");
        ERRCHK_MPI_API(MPI_File_open(cart_comm, path.c_str(), MPI_MODE_RDONLY, info, &m_file));
        ERRCHK_MPI_API(
            MPI_File_set_view(m_file, 0, get_dtype<T>(), global_subarray, "native", info));

        MPI_Offset bytes;
        ERRCHK_MPI_API(MPI_File_get_size(m_file, &bytes));
        ERRCHK_MPI_EXPR_DESC(as<uint64_t>(bytes) == prod(m_file_dims) * sizeof(T),
                             "Tried to read a file that had unexpected file size. Ensure that the "
                             "file read/written using the same grid dimensions.");

        MPI_Status status;
        status.MPI_ERROR = MPI_SUCCESS;
        ERRCHK_MPI_API(MPI_File_read_all(m_file, data, 1, local_subarray, &status));
        ERRCHK_MPI_API(status.MPI_ERROR);
        ERRCHK_MPI_API(MPI_File_close(&m_file));
        ERRCHK_MPI(m_file == MPI_FILE_NULL);

        ERRCHK_MPI_API(MPI_Type_free(&local_subarray));
        ERRCHK_MPI_API(MPI_Type_free(&global_subarray));
        ERRCHK_MPI_API(MPI_Info_free(&info));
    };

    void write(const MPI_Comm cart_comm, const std::string& path, const T* data)
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

        MPI_Datatype global_subarray = create_subarray(m_file_dims, m_mesh_subdims, m_file_offset,
                                                       get_dtype<T>());
        MPI_Datatype local_subarray  = create_subarray(m_mesh_dims, m_mesh_subdims, m_mesh_offset,
                                                       get_dtype<T>());

        ERRCHK_MPI_EXPR_DESC(m_file == MPI_FILE_NULL,
                             "Previous IO operation was still in progress.");
        ERRCHK_MPI_API(MPI_File_open(cart_comm, path.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY,
                                     info, &m_file));
        ERRCHK_MPI_API(
            MPI_File_set_view(m_file, 0, get_dtype<T>(), global_subarray, "native", info));

        MPI_Status status;
        status.MPI_ERROR = MPI_SUCCESS;
        ERRCHK_MPI_API(MPI_File_write_all(m_file, data, 1, local_subarray, &status));
        ERRCHK_MPI_API(status.MPI_ERROR);
        ERRCHK_MPI_API(MPI_File_close(&m_file));
        ERRCHK_MPI(m_file == MPI_FILE_NULL);

        ERRCHK_MPI_API(MPI_Type_free(&local_subarray));
        ERRCHK_MPI_API(MPI_Type_free(&global_subarray));
        ERRCHK_MPI_API(MPI_Info_free(&info));
    }

    void launch_write(const MPI_Comm cart_comm, const std::string& path, const T* data)
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

        MPI_Datatype global_subarray = create_subarray(m_file_dims, m_mesh_subdims, m_file_offset,
                                                       get_dtype<T>());
        MPI_Datatype local_subarray  = create_subarray(m_mesh_dims, m_mesh_subdims, m_mesh_offset,
                                                       get_dtype<T>());

        ERRCHK_MPI(m_file == MPI_FILE_NULL);

        // TODO: copy input to the staging buffer

        ERRCHK_MPI_API(MPI_File_open(cart_comm, path.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY,
                                     info, &m_file));
        ERRCHK_MPI_API(
            MPI_File_set_view(m_file, 0, get_dtype<T>(), global_subarray, "native", info));
        ERRCHK_MPI_API(MPI_File_iwrite_all(m_file, data, 1, local_subarray, &m_req));
        ERRCHK_MPI_API(MPI_Type_free(&local_subarray));
        ERRCHK_MPI_API(MPI_Type_free(&global_subarray));
        ERRCHK_MPI_API(MPI_Info_free(&info));
    };

    void wait_write()
    {
        ERRCHK_MPI_EXPR_DESC(m_file != MPI_FILE_NULL,
                             "Function called but there was no write operation in progress. "
                             "Function launch_write must be called before wait_write.");
        ERRCHK_MPI_EXPR_DESC(m_req != MPI_REQUEST_NULL,
                             "Function called but there was no write operation in progress. "
                             "Function launch_write must be called before wait_write.");
        wait_request(m_req);
        ERRCHK_MPI_API(MPI_File_close(&m_file));
        ERRCHK_MPI(m_file == MPI_FILE_NULL);
    };
};
