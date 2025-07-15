add_library(ac_update_params OBJECT update_params.cc)
target_include_directories(ac_update_params PUBLIC ${AC_BASE_PATH}/include ${AC_BASE_PATH}/src/core/kernels)
target_link_libraries(ac_update_params PRIVATE acc-runtime-headers)
if (MPI_ENABLED)
    set(MPI_CXX_SKIP_MPICXX ON) # Skip MPI C++ bindings
    find_package(MPI REQUIRED COMPONENTS C)
    target_link_libraries(ac_update_params PUBLIC MPI::MPI_C)
endif()
