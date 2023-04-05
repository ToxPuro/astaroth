## Astaroth Core
add_library(astaroth_core device.cc grid.cc task.cc node.cc astaroth.cc astaroth_fortran.cc debug.cc extrema.cu)

if (USE_HIP)
    find_package(hip)
    #TODO: link against HIP thrust
    target_link_libraries(astaroth_core kernels hip::host)
else()
    find_package(CUDAToolkit)
    find_package(Thrust REQUIRED CONFIG)
    thrust_create_target(Thrust)
    target_link_libraries(astaroth_core kernels CUDA::cudart CUDA::cuda_driver Thrust)
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
