## Astaroth Kernels
add_library(kernels STATIC kernels.cu)
add_dependencies(kernels acc-runtime)
target_compile_features(kernels PRIVATE cxx_std_17)