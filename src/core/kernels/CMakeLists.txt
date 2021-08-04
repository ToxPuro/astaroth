## Astaroth Kernels
if (USE_HIP)
    set_source_files_properties(kernels.cu PROPERTIES LANGUAGE CXX)
endif()

add_library(kernels STATIC kernels.cu)
add_dependencies(kernels acc-runtime-headers)
target_compile_features(kernels PRIVATE cxx_std_17)
