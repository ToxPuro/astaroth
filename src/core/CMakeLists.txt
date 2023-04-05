## Astaroth Core
add_library(astaroth_core device.cc grid.cc task.cc node.cc astaroth.cc astaroth_fortran.cc debug.cc)

if (USE_HIP)
    find_package(hip)
    find_package(rocblas)
    target_link_libraries(astaroth_core kernels hip::host roc::rocblas)
else()
    find_package(CUDAToolkit)
    target_link_libraries(astaroth_core kernels CUDA::cudart CUDA::cuda_driver CUDA::cublas)
endif()

if (USE_PERFSTUBS)
    find_package(Threads)
    # Would like to define scope here, but the other target_link_libraries are plain...
    target_link_libraries(astaroth_core perfstubs Threads::Threads)
endif()

## Options
if (MPI_ENABLED)
    target_link_libraries(astaroth_core MPI::MPI_C)
endif()
