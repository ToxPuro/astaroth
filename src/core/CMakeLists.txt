## Astaroth Core
add_library(astaroth_core SHARED device.cc grid.cc task.cc node.cc astaroth.cc astaroth_fortran.cc debug.cc)

if (USE_HIP)
    find_package(hip)
    target_link_libraries(astaroth_core kernels hip::host)
else()
    find_package(CUDAToolkit)
    target_link_libraries(astaroth_core kernels CUDA::cudart CUDA::cuda_driver)
endif()

## Options
if (MPI_ENABLED)
    target_link_libraries(astaroth_core MPI::MPI_C)
endif()
