## Project settings
project(pyastaroth
        LANGUAGES CXX)

## Dependencies
litgen_find_nanobind()

set(litgen_generator "${CMAKE_CURRENT_SOURCE_DIR}/litgen_generate_bindings.py")
set(pyastaroth_pydef "pydef/pyastaroth-impl.cc")
set(pyastaroth_stubs_pyi "stubs/pyastaroth/__init__.pyi")
set(pyastaroth_include_directories
     "${CMAKE_SOURCE_DIR}/include"
     "${CMAKE_BINARY_DIR}/acc-runtime/api"
     "${CMAKE_SOURCE_DIR}/acc-runtime/api")

add_custom_target(pyastaroth_bindings
                  COMMAND ${litgen_generator} ${pyastaroth_pydef} ${pyastaroth_stubs_pyi} ${CMAKE_SOURCE_DIR} ${pyastaroth_include_directories}
                  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                  COMMENT "Generating C++ polyfil for Python bindings using litgen")
add_dependencies(pyastaroth_bindings acc-runtime-headers)

nanobind_add_module(pyastaroth_impl ${pyastaroth_pydef})

litgen_setup_module(astaroth_core
                    pyastaroth_impl
                    pyastaroth
                    ${CMAKE_CURRENT_SOURCE_DIR}/stubs)

add_dependencies(pyastaroth_impl pyastaroth_bindings)
target_include_directories(pyastaroth_impl PUBLIC ${AC_BASE_PATH}/include ${AC_BASE_PATH}/acc-runtime/api)
target_link_libraries(pyastaroth_impl PUBLIC acc-runtime-headers astaroth_utils)
