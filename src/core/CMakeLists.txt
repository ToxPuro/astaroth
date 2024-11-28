## Astaroth Core
add_library(astaroth_core device.cc grid.cc task.cc node.cc astaroth.cc astaroth_fortran.cc debug.cc decomposition.cc)
target_link_libraries(astaroth_core PUBLIC kernels)
target_include_directories(astaroth_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../include)

if (MPI_ENABLED)
    find_package(MPI REQUIRED COMPONENTS C)
    target_link_libraries(astaroth_core PUBLIC MPI::MPI_C)
endif()

if (USE_HIP)
    # May need to link to hip::host at some point
else()
    target_link_libraries(astaroth_core PUBLIC CUDA::cudart_static)
endif()

# JP: disabled due to being error-prone, verbose, and not using modern CMake practices
# JP: Should consider enabling something here, e.g., CRAYPE GTL if still needed
# JP: on LUMI (the workaround was done 1.5 years ago and there have been several LUMI
# JP: updates since)
# if (USE_HIP)
#     find_package(hip)
#     target_link_libraries(astaroth_core
# 	PUBLIC
#         kernels
#         hip::host
#     )

#     if(DEFINED ENV{CRAYPE_VERSION})
#         # Workaround due to hip/cmake/cray PE issue with GPU RDMA
#         # If we're on a Cray machine, link against the GTL library
#         # WARNING: this path may break, works on LUMI in June of 2023
#         target_link_directories(astaroth_core
#             PUBLIC
# 	    /opt/cray/pe/lib64
# 	#/opt/rocm-5.2.3/llvm/lib
#         )
#         target_link_libraries(astaroth_core
#             PUBLIC
#             mpi_gtl_hsa
#         )
#     endif()

# JP: disabled as this caused issues on Triton. CMake should resolve these automatically.
#    message("AMD GPU targets: ${AMDGPU_TARGETS}\n")
#    foreach(amdgpu_target ${AMDGPU_TARGETS})
#	target_link_libraries(astaroth_core
#	    INTERFACE
#	      --cuda-gpu-arch=${amdgpu_target}
#	)
#    endforeach()

# else()
#     find_package(CUDAToolkit)
#     target_link_libraries(astaroth_core
# 	PUBLIC
# 	kernels CUDA::cudart CUDA::cuda_driver)
# endif()

# if (USE_PERFSTUBS)
#     find_package(Threads)
#     target_link_libraries(astaroth_core
# 	PUBLIC
#         perfstubs
# 	Threads::Threads
#     )
# else()
#     find_package(Threads REQUIRED)
#     target_link_libraries(astaroth_core PRIVATE Threads::Threads)
# endif()

## Options
# if (MPI_ENABLED)
#     target_link_libraries(astaroth_core
# 	PUBLIC
# 	MPI::MPI_C
#     )
# endif()
