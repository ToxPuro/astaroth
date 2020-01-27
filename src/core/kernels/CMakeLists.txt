## Astaroth Kernels
add_library(astaroth_kernels STATIC kernels.cu)
add_dependencies(astaroth_kernels dsl_headers)
target_compile_features(astaroth_kernels PRIVATE cxx_std_11)
set_property(TARGET astaroth_kernels PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET astaroth_kernels PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
