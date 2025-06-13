if (NOT DEFINED DSL_MODULE_DIR)
    get_filename_component(resolved_module_dir acc-runtime/samples/mhd_modular REALPATH BASE_DIR ${AC_BASE_PATH})
    set(DSL_MODULE_DIR ${resolved_module_dir}) # Default
else()
    get_filename_component(resolved_module_dir "${DSL_MODULE_DIR}" REALPATH
                           BASE_DIR ${CMAKE_BINARY_DIR})
    set(DSL_MODULE_DIR ${resolved_module_dir})
endif()

add_library(acc_runtime_compiler compiler.cpp ../src/core/update_params.cc ../src/core/decomposition.cc)
target_link_libraries(acc_runtime_compiler PRIVATE acc-runtime-headers)
target_include_directories(acc_runtime_compiler PUBLIC ${AC_BASE_PATH}/include)
target_compile_definitions(acc_runtime_compiler PRIVATE DSL_MODULE_DIR="${DSL_MODULE_DIR}")
add_definitions(-DAC_CPU_CODE=1)
if (MPI_ENABLED)
    set(MPI_CXX_SKIP_MPICXX ON) # Skip MPI C++ bindings
    find_package(MPI REQUIRED COMPONENTS C)
    target_link_libraries(astaroth_helpers PRIVATE MPI::MPI_C)
endif()
