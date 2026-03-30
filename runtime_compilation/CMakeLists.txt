if (NOT DEFINED DSL_MODULE_DIR)
    get_filename_component(resolved_module_dir acc-runtime/samples/mhd_modular REALPATH BASE_DIR ${AC_BASE_PATH})
    set(DSL_MODULE_DIR ${resolved_module_dir}) # Default
else()
    get_filename_component(resolved_module_dir "${DSL_MODULE_DIR}" REALPATH
                           BASE_DIR ${CMAKE_BINARY_DIR})
    set(DSL_MODULE_DIR ${resolved_module_dir})
endif()

if (NOT DEFINED AC_STENCIL_ORDER)
  set(AC_STENCIL_ORDER -1)
endif()

if(RUNTIME_COMPILATION)
  add_library(acc_runtime_compiler OBJECT compiler.cpp)
else()
  add_library(acc_runtime_compiler OBJECT nocompiler.cpp)
endif()
set_property(TARGET acc_runtime_compiler PROPERTY POSITION_INDEPENDENT_CODE ON)

target_compile_definitions(acc_runtime_compiler PRIVATE -DAC_STENCIL_ORDER=${AC_STENCIL_ORDER})
target_link_libraries(acc_runtime_compiler PRIVATE acc-runtime-headers)
target_include_directories(acc_runtime_compiler PUBLIC ${AC_BASE_PATH}/include)
target_compile_definitions(acc_runtime_compiler PRIVATE DSL_MODULE_DIR="${DSL_MODULE_DIR}")
target_compile_definitions(acc_runtime_compiler PRIVATE -DRUNTIME_CMAKE_OPTIONS="${RUNTIME_CMAKE_OPTIONS}")
ac_link_mpi(acc_runtime_compiler)
add_definitions(-DAC_CPU_CODE=1)

