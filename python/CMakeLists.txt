## Project settings
project(pyastaroth
        LANGUAGES CXX)

## Dependencies
litgen_find_nanobind()
litgen_find_mpi4py()

set(pydef_dir "${CMAKE_CURRENT_SOURCE_DIR}/pydef")
set(stubs_dir "${CMAKE_CURRENT_SOURCE_DIR}/stubs")

set(pyastaroth_header "${CMAKE_SOURCE_DIR}/include/astaroth.h")
set(pyastaroth_pydef "${pydef_dir}/pyastaroth-impl.cc")
set(pyastaroth_stubs_pyi "${stubs_dir}/pyastaroth/__init__.pyi")
set(pyastaroth_include_directories "${CMAKE_SOURCE_DIR}/include"
                                   "${CMAKE_BINARY_DIR}/acc-runtime/api"
                                   "${CMAKE_SOURCE_DIR}/acc-runtime/api")

litgen_generate_bindings(pyastaroth_bindings
                         ${pyastaroth_pydef}
                         ${pyastaroth_stubs_pyi}
                         ${CMAKE_SOURCE_DIR}
                         ${pyastaroth_header}
                         "${pyastaroth_include_directories}")
add_dependencies(pyastaroth_bindings acc-runtime-headers)

nanobind_add_module(pyastaroth_impl ${pyastaroth_pydef})

litgen_setup_module(astaroth_core
                    pyastaroth_impl
                    pyastaroth/impl
                    ${stubs_dir})

add_dependencies(pyastaroth_impl pyastaroth_bindings)
target_include_directories(pyastaroth_impl PUBLIC
                           ${AC_BASE_PATH}/include
                           ${AC_BASE_PATH}/acc-runtime/api
                           ${mpi4py_INCLUDE_DIR})
target_link_libraries(pyastaroth_impl PUBLIC acc-runtime-headers)
ac_link_mpi(pyastaroth_impl)

set(pyastaroth_utils_header "${CMAKE_SOURCE_DIR}/include/astaroth_utils.h")
set(pyastaroth_utils_pydef     "${pydef_dir}/pyastaroth-utils-impl.cc")
set(pyastaroth_utils_stubs_pyi "${stubs_dir}/pyastaroth/__init__.pyi")
set(pyastaroth_utils_include_directories "${CMAKE_SOURCE_DIR}/include"
                                         "${CMAKE_BINARY_DIR}/acc-runtime/api"
                                         "${CMAKE_SOURCE_DIR}/acc-runtime/api")

litgen_generate_bindings(pyastaroth_utils_bindings
                         ${pyastaroth_utils_pydef}
                         ${pyastaroth_utils_stubs_pyi}
                         ${CMAKE_SOURCE_DIR}
                         ${pyastaroth_utils_header}
                         "${pyastaroth_utils_include_directories}")
add_dependencies(pyastaroth_utils_bindings acc-runtime-headers)

nanobind_add_module(pyastaroth_utils_impl ${pyastaroth_utils_pydef})

litgen_setup_module(astaroth_utils
                    pyastaroth_utils_impl
                    pyastaroth/impl
                    ${stubs_dir})

add_dependencies(pyastaroth_utils_impl pyastaroth_utils_bindings)
target_include_directories(pyastaroth_utils_impl PUBLIC ${AC_BASE_PATH}/include)
target_link_libraries(pyastaroth_utils_impl PUBLIC astaroth_utils)
