if(NOT USE_HIP)
    find_package(CUDAToolkit)
else()
    # todo whatver is needed for HIP
endif()
## Astaroth Core
add_library(astaroth_core STATIC device.cc grid.cc task.cc node.cc astaroth.cc astaroth_fortran.cc)
target_link_libraries(astaroth_core astaroth_kernels)
if(NOT USE_HIP)
    target_link_libraries(astaroth_core CUDA::cudart CUDA::cuda_driver)
endif()

## Options
if (MPI_ENABLED)
    target_link_libraries(astaroth_core MPI::MPI_C)
endif()
