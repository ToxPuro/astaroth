find_package(CUDAToolkit)

## Astaroth Core
add_library(astaroth_core STATIC device.cc grid.cc task.cc node.cc astaroth.cc astaroth_fortran.cc debug.cc)
target_link_libraries(astaroth_core kernels CUDA::cudart CUDA::cuda_driver)

## Options
if (MPI_ENABLED)
    target_link_libraries(astaroth_core MPI::MPI_C)
endif()
