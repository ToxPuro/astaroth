#pragma once
#include <memory>

#include <mpi.h>

using mpi_request_ptr_t  = std::unique_ptr<MPI_Request, void (*)(MPI_Request*)>;
using mpi_comm_ptr_t     = std::unique_ptr<MPI_Comm, void (*)(MPI_Comm*)>;
using mpi_file_ptr_t     = std::unique_ptr<MPI_File, void (*)(MPI_File*)>;
using mpi_datatype_ptr_t = std::unique_ptr<MPI_Datatype, void (*)(MPI_Datatype*)>;
using mpi_info_ptr_t     = std::unique_ptr<MPI_Info, void (*)(MPI_Info*)>;

static inline mpi_request_ptr_t
mpi_request_make_unique()
{
    return mpi_request_ptr_t{[]() {
                                 PRINT_LOG("new");
                                 MPI_Request* ptr = new MPI_Request;
                                 *ptr             = MPI_REQUEST_NULL;
                                 return ptr;
                             }(),
                             [](MPI_Request* ptr) {
                                 PRINT_LOG("delete");
                                 ERRCHK_MPI(*ptr == MPI_REQUEST_NULL);
                                 delete ptr;
                             }};
}

static inline mpi_comm_ptr_t
mpi_comm_make_unique()
{
    return mpi_comm_ptr_t{[]() {
                              PRINT_LOG("new");
                              MPI_Comm* ptr = new MPI_Comm;
                              *ptr          = MPI_COMM_NULL;
                              return ptr;
                          }(),
                          [](MPI_Comm* ptr) {
                              PRINT_LOG("delete");
                              if (*ptr != MPI_COMM_NULL)
                                  ERRCHK_MPI_API(MPI_Comm_free(ptr));
                              delete ptr;
                          }};
}

static inline mpi_file_ptr_t
mpi_file_make_unique()
{
    return mpi_file_ptr_t{[]() {
                              PRINT_LOG("new");
                              MPI_File* ptr = new MPI_File;
                              *ptr          = MPI_FILE_NULL;
                              return ptr;
                          }(),
                          [](MPI_File* ptr) {
                              PRINT_LOG("delete");
                              if (*ptr != MPI_FILE_NULL)
                                  ERRCHK_MPI_API(MPI_File_close(ptr));
                              delete ptr;
                          }};
}

static inline mpi_datatype_ptr_t
mpi_datatype_make_unique()
{
    return mpi_datatype_ptr_t{[]() {
                                  PRINT_LOG("new");
                                  MPI_Datatype* ptr = new MPI_Datatype;
                                  *ptr              = MPI_DATATYPE_NULL;
                                  return ptr;
                              }(),
                              [](MPI_Datatype* ptr) {
                                  PRINT_LOG("delete");
                                  if (*ptr != MPI_DATATYPE_NULL)
                                      ERRCHK_MPI_API(MPI_Type_free(ptr));
                                  delete ptr;
                              }};
}

static inline mpi_info_ptr_t
mpi_info_make_unique()
{
    return mpi_info_ptr_t{[]() {
                              PRINT_LOG("new");
                              MPI_Info* ptr = new MPI_Info;
                              *ptr          = MPI_INFO_NULL;
                              return ptr;
                          }(),
                          [](MPI_Info* ptr) {
                              PRINT_LOG("delete");
                              if (*ptr != MPI_INFO_NULL)
                                  ERRCHK_MPI_API(MPI_Type_free(ptr));
                              delete ptr;
                          }};
}
