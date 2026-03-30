add_library(ac_decomposition OBJECT decomposition.cc)
set_property(TARGET ac_decomposition PROPERTY POSITION_INDEPENDENT_CODE ON)
target_include_directories(ac_decomposition PUBLIC ${AC_BASE_PATH}/include ${AC_BASE_PATH}/src/core/kernels)
target_link_libraries(ac_decomposition PRIVATE acc-runtime-headers)
ac_link_mpi(ac_decomposition)

