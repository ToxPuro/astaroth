## Astaroth Kernels
if (USE_HIP)
    set_source_files_properties(kernels.cu PROPERTIES LANGUAGE CXX)
endif()

add_library(kernels STATIC kernels.cu)
target_link_libraries(kernels PUBLIC acc-runtime-headers)
target_compile_features(kernels PRIVATE cxx_std_17)
