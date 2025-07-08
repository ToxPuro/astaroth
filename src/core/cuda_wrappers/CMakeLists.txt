add_library(ac_cuda_wrappers STATIC cuda.cc)
target_link_libraries(ac_cuda_wrappers PRIVATE acc-runtime-headers)
target_include_directories(ac_cuda_wrappers PRIVATE ${AC_BASE_PATH}/include)
target_include_directories(ac_cuda_wrappers PRIVATE ${AC_BASE_PATH}/acc-runtime/api)
