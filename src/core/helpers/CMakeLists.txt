## Astaroth Core
add_library(astaroth_helpers STATIC helpers.cc device.cc buffer.cc logging.c)
target_include_directories(astaroth_helpers PUBLIC ${AC_BASE_PATH}/include ${AC_BASE_PATH}/src/core/kernels ${AC_BASE_PATH}/acc-runtime/api)
if (MPI_ENABLED)
    set(MPI_CXX_SKIP_MPICXX ON) # Skip MPI C++ bindings
    find_package(MPI REQUIRED COMPONENTS C)
    target_link_libraries(astaroth_helpers PRIVATE MPI::MPI_C)
    target_include_directories(astaroth_helpers PRIVATE ${MPI_C_INCLUDE_DIRS})
endif()
#target_link_libraries(astaroth_helpers PRIVATE acc-runtime-headers )
