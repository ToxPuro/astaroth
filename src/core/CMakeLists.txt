## Astaroth Core

if (USE_HIP)
    set_source_files_properties(extrema.cu PROPERTIES LANGUAGE CXX)
    add_library(astaroth_core
	device.cc grid.cc task.cc node.cc astaroth.cc astaroth_fortran.cc debug.cc extrema.cu
    )
    find_package(hip)
    find_package(rocthrust)

    #link_directories($ENV{MPICH_DIR}/../../../gtl/lib)

    #link_directories($ENV{MPICH_DIR}/../../../gtl/lib)
    #libmpi_gtl_hsa.so.0 => /opt/cray/pe/lib64/libmpi_gtl_hsa.so.0
    #target_link_directories(astaroth_core PUBLIC /opt/cray/pe/mpich/8.1.23/gtl/lib)
    target_link_directories(astaroth_core PUBLIC /opt/cray/pe/lib64)
    target_link_libraries(astaroth_core
	    PUBLIC
	    kernels
	    hip::host
	    roc::rocthrust
	    roc::rocprim_hip
	    mpi_gtl_hsa
	    )
    message("AMD GPU targets: ${AMDGPU_TARGETS}\n")
    foreach(amdgpu_target ${AMDGPU_TARGETS})
    target_link_libraries(astaroth_core
        INTERFACE
            --cuda-gpu-arch=${amdgpu_target}
    )
    endforeach()
else()
    add_library(astaroth_core
	device.cc grid.cc task.cc node.cc astaroth.cc astaroth_fortran.cc debug.cc extrema.cu
    )
    find_package(CUDAToolkit)
    find_package(Thrust REQUIRED CONFIG)
    thrust_create_target(Thrust)
    target_link_libraries(astaroth_core PUBLIC kernels CUDA::cudart CUDA::cuda_driver Thrust)
endif()

if (USE_PERFSTUBS)
    find_package(Threads)
    # Would like to define scope here, but the other target_link_libraries are plain...
    target_link_libraries(astaroth_core PUBLIC perfstubs Threads::Threads)
endif()

## Options
if (MPI_ENABLED)
	target_link_libraries(astaroth_core PUBLIC MPI::MPI_C)
endif()
