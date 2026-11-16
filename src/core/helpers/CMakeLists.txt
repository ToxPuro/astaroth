## Astaroth Core
    # if (MPI_ENABLED)
if(MPI_ENABLED)
  add_library(astaroth_helpers OBJECT helpers.cc device.cc buffer.cc logging.c mpi.cc)
else()
  add_library(astaroth_helpers OBJECT helpers.cc device.cc buffer.cc logging.c nompi.cc)
endif()

set_property(TARGET astaroth_helpers PROPERTY POSITION_INDEPENDENT_CODE ON)
target_include_directories(astaroth_helpers PUBLIC ${AC_BASE_PATH}/include ${AC_BASE_PATH}/src ${AC_BASE_PATH}/src/core/kernels ${AC_BASE_PATH}/acc-runtime/api)
ac_link_mpi(astaroth_helpers)
target_link_libraries(astaroth_helpers PRIVATE acc-runtime-headers)
