## Astaroth Kernels
add_library(astaroth_kernels STATIC kernels.cc)
# tell cmake that kernels.cpp is really a cuda file
if(NOT USE_HIP)
    set_source_files_properties(kernels.cc PROPERTIES LANGUAGE CUDA)
endif()
add_dependencies(astaroth_kernels dsl_headers)
target_compile_features(astaroth_kernels PRIVATE cxx_std_11)
