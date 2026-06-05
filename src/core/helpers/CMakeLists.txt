## Astaroth Core
if(RUNTIME_COMPILATION)
	add_library(astaroth_helpers OBJECT helpers.cc device.cc buffer.cc logging.c)
else()
	add_library(astaroth_helpers OBJECT helpers.cc device.cc buffer.cc logging.c)
endif()

set_property(TARGET astaroth_helpers PROPERTY POSITION_INDEPENDENT_CODE ON)
target_include_directories(astaroth_helpers PUBLIC ${AC_BASE_PATH}/include ${AC_BASE_PATH}/src/core/kernels ${AC_BASE_PATH}/acc-runtime/api)
target_link_libraries(astaroth_helpers PUBLIC acc-runtime-headers)
ac_link_mpi(astaroth_helpers)
target_link_libraries(astaroth_helpers PRIVATE acc-runtime-headers)
