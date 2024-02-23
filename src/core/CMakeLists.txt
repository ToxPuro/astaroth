## Astaroth Core
add_library(astaroth_core device.cc grid.cc task.cc node.cc astaroth.cc astaroth_fortran.cc debug.cc)

if (USE_HIP)
    find_package(hip)
    target_link_libraries(astaroth_core
	PUBLIC
        kernels
        hip::host
    )

    if(DEFINED ENV{CRAYPE_VERSION})
        # Workaround due to hip/cmake/cray PE issue with GPU RDMA
        # If we're on a Cray machine, link against the GTL library
        # WARNING: this path may break, works on LUMI in June of 2023
        target_link_directories(astaroth_core
            PUBLIC
	    /opt/cray/pe/lib64
	#/opt/rocm-5.2.3/llvm/lib
        )
        target_link_libraries(astaroth_core
            PUBLIC
            mpi_gtl_hsa
        )
    endif()

    message("AMD GPU targets: ${AMDGPU_TARGETS}\n")
    foreach(amdgpu_target ${AMDGPU_TARGETS})
	target_link_libraries(astaroth_core
	    INTERFACE
	      --cuda-gpu-arch=${amdgpu_target}
	)
    endforeach()

else()
    find_package(CUDAToolkit)
    target_link_libraries(astaroth_core
	PUBLIC
	kernels CUDA::cudart CUDA::cuda_driver)
endif()

if (USE_PERFSTUBS)
    find_package(Threads)
    target_link_libraries(astaroth_core
	PUBLIC
        perfstubs
	Threads::Threads
    )
endif()

## Definitions
if (MULTIGPU_ENABLED)
    add_definitions(-DAC_MULTIGPU_ENABLED=1)
else ()
    add_definitions(-DAC_MULTIGPU_ENABLED=0)
endif ()

if (PACKED_DATA_TRANSFERS)
    add_definitions(-DPACKED_DATA_TRANSFERS=1)
else ()
    add_definitions(-DPACKED_DATA_TRANSFERS=0)
endif ()

## Options
if (MPI_ENABLED)
    target_link_libraries(astaroth_core
	PUBLIC
	MPI::MPI_C
    )
endif()


