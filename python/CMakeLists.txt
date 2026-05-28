## Project settings
project(pyastaroth LANGUAGES CXX)

## Dependencies
litgen_find_nanobind()
litgen_find_mpi4py()

set(pydef_dir "${CMAKE_CURRENT_SOURCE_DIR}/pydef")
set(stubs_dir "${CMAKE_CURRENT_SOURCE_DIR}/stubs")

set(pyastaroth_pydef               "${pydef_dir}/pyastaroth-core-impl.cc")
set(pyastaroth_include_directories "${CMAKE_SOURCE_DIR}/include"
                                   "${CMAKE_BINARY_DIR}/acc-runtime/api"
                                   "${CMAKE_SOURCE_DIR}/acc-runtime/api")

litgen_generate_bindings(
    pyastaroth_core_bindings
    PYDEF_FILE
        ${pyastaroth_pydef}
    STUBS_FILE
        "${stubs_dir}/pyastaroth/impl/core.pyi"
    BASE_DIR
        ${CMAKE_SOURCE_DIR}
    HEADER_FILE
        "${CMAKE_SOURCE_DIR}/include/astaroth.h"
    INCLUDE_DIRECTORIES
        "${pyastaroth_include_directories}"
    AMALGAMATE
)
add_dependencies(pyastaroth_core_bindings acc-runtime-headers)

nanobind_add_module(
    core
    NB_SUPPRESS_WARNINGS
    ${pyastaroth_pydef}
)
litgen_setup_module(
    astaroth_core
    core
    pyastaroth/impl
    ${stubs_dir}
)

add_dependencies(core pyastaroth_core_bindings)
target_include_directories(
    core
    PUBLIC
        ${AC_BASE_PATH}/include
        ${AC_BASE_PATH}/acc-runtime/api
        ${mpi4py_INCLUDE_DIR}
)
target_link_libraries(core PUBLIC astaroth_core acc-runtime-driver)
ac_link_mpi(core)

set(pyastaroth_grid_pydef               "${pydef_dir}/pyastaroth-grid-impl.cc")
set(pyastaroth_grid_include_directories "${CMAKE_SOURCE_DIR}/include"
                                        "${CMAKE_BINARY_DIR}/acc-runtime/api"
                                        "${CMAKE_SOURCE_DIR}/acc-runtime/api")

litgen_generate_bindings(
    pyastaroth_grid_bindings
    PYDEF_FILE
        ${pyastaroth_grid_pydef}
    STUBS_FILE
        "${stubs_dir}/pyastaroth/impl/grid.pyi"
    BASE_DIR
        ${CMAKE_SOURCE_DIR}
    HEADER_FILE
        "${CMAKE_SOURCE_DIR}/include/astaroth_grid.h"
)
add_dependencies(pyastaroth_grid_bindings acc-runtime-headers)

nanobind_add_module(
    grid
    NB_SUPPRESS_WARNINGS
    ${pyastaroth_grid_pydef}
)
litgen_setup_module(
    astaroth_core
    grid
    pyastaroth/impl
    ${stubs_dir}
)

add_dependencies(grid pyastaroth_grid_bindings)
target_include_directories(
    grid
    PUBLIC
        ${AC_BASE_PATH}/include
        ${mpi4py_INCLUDE_DIR}
)
ac_link_mpi(grid)

set(pyastaroth_node_pydef               "${pydef_dir}/pyastaroth-node-impl.cc")
set(pyastaroth_node_include_directories "${CMAKE_SOURCE_DIR}/include"
                                        "${CMAKE_BINARY_DIR}/acc-runtime/api"
                                        "${CMAKE_SOURCE_DIR}/acc-runtime/api")

litgen_generate_bindings(
    pyastaroth_node_bindings
    PYDEF_FILE
        ${pyastaroth_node_pydef}
    STUBS_FILE
        "${stubs_dir}/pyastaroth/impl/node.pyi"
    BASE_DIR
        ${CMAKE_SOURCE_DIR}
    HEADER_FILE
        "${CMAKE_SOURCE_DIR}/include/astaroth_node.h"
)
add_dependencies(pyastaroth_node_bindings acc-runtime-headers)

nanobind_add_module(
    node
    NB_SUPPRESS_WARNINGS
    ${pyastaroth_node_pydef}
)
litgen_setup_module(
    astaroth_core
    node
    pyastaroth/impl
    ${stubs_dir}
)

add_dependencies(node pyastaroth_node_bindings)
target_include_directories(
    node
    PUBLIC
        ${AC_BASE_PATH}/include
)

# set(pyastaroth_device_pydef               "${pydef_dir}/pyastaroth-device-impl.cc")
# set(pyastaroth_device_include_directories "${CMAKE_SOURCE_DIR}/include"
#                                           "${CMAKE_BINARY_DIR}/acc-runtime/api"
#                                           "${CMAKE_SOURCE_DIR}/acc-runtime/api")
#
# litgen_generate_bindings(
#     pyastaroth_device_bindings
#     PYDEF_FILE
#         ${pyastaroth_device_pydef}
#     STUBS_FILE
#         "${stubs_dir}/pyastaroth/impl/device.pyi"
#     BASE_DIR
#         ${CMAKE_SOURCE_DIR}
#     HEADER_FILE
#         "${CMAKE_SOURCE_DIR}/include/astaroth_device.h"
# )
# add_dependencies(pyastaroth_device_bindings acc-runtime-headers)
#
# nanobind_add_module(
#     device
#     NB_SUPPRESS_WARNINGS
#     ${pyastaroth_device_pydef}
# )
# litgen_setup_module(
#     astaroth_core
#     device
#     pyastaroth/impl
#     ${stubs_dir}
# )
#
# add_dependencies(device pyastaroth_device_bindings)
# target_include_directories(
#     device
#     PUBLIC
#         ${AC_BASE_PATH}/include
# )

set(pyastaroth_utils_pydef               "${pydef_dir}/pyastaroth-utils-impl.cc")
set(pyastaroth_utils_include_directories "${CMAKE_SOURCE_DIR}/include"
                                         "${CMAKE_BINARY_DIR}/acc-runtime/api"
                                         "${CMAKE_SOURCE_DIR}/acc-runtime/api")

litgen_generate_bindings(
    pyastaroth_utils_bindings
    PYDEF_FILE
        ${pyastaroth_utils_pydef}
    STUBS_FILE
        "${stubs_dir}/pyastaroth/impl/utils.pyi"
    BASE_DIR
        ${CMAKE_SOURCE_DIR}
    HEADER_FILE
        "${CMAKE_SOURCE_DIR}/include/astaroth_utils.h"
)
add_dependencies(pyastaroth_utils_bindings acc-runtime-headers)

nanobind_add_module(
    utils
    NB_SUPPRESS_WARNINGS
    ${pyastaroth_utils_pydef}
)
litgen_setup_module(
    astaroth_utils
    utils
    pyastaroth/impl
    ${stubs_dir}
)

add_dependencies(utils pyastaroth_utils_bindings)
target_include_directories(utils PUBLIC ${AC_BASE_PATH}/include)
