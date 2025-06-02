if (NOT DEFINED DSL_MODULE_DIR)
    set(DSL_MODULE_DIR ${PROJECT_SOURCE_DIR}/samples/mhd_modular) # Default
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
