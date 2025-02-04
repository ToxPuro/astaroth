## Packages
set(MPI_CXX_SKIP_MPICXX ON) # Skip MPI C++ bindings
find_package(MPI REQUIRED COMPONENTS C)

## Library
add_library(acm SHARED acm.cc acm_error.c datatypes.cc halo_exchange.cc packet.cc mpi_utils.cc buffer_exchange.cc reduce.cu transform.cu pack.cu pack.cc transform.cc ndbuffer.cc buffer.cc decomp.cc partition.cc segment.cc math_utils.cc ntuple.cc pointer.cc allocator.cc type_conversion.cc errchk_print.c)
target_link_libraries(acm PUBLIC MPI::MPI_C)
target_include_directories(acm PUBLIC ${PROJECT_SOURCE_DIR})

## CUDA & HIP
find_package(CUDAToolkit QUIET)
find_package(HIP QUIET)
if(NOT HOST_ONLY_MODE AND CUDAToolkit_FOUND)
    message(STATUS "CUDA found")
    enable_language(CUDA)
    target_compile_definitions(acm PUBLIC ACM_DEVICE_ENABLED)
    target_compile_definitions(acm PUBLIC ACM_CUDA_ENABLED)
    include_directories(SYSTEM ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}) # Silence warnings originating from CUDA headers
    target_link_libraries(acm PUBLIC CUDA::cudart)
    

    set(CMAKE_CUDA_ARCHITECTURES "60;61;70;80")
    message(STATUS "Enabled CUDA architectures (ACM): " ${CMAKE_CUDA_ARCHITECTURES})
    set_source_files_properties(reduce.cu transform.cu pack.cu PROPERTIES LANGUAGE CUDA)
    set_property(TARGET acm PROPERTY CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})
elseif(NOT HOST_ONLY_MODE AND HIP_FOUND)
    message(STATUS "HIP found")
    enable_language(HIP)
    target_compile_definitions(acm PUBLIC ACM_DEVICE_ENABLED)
    target_compile_definitions(acm PUBLIC ACM_HIP_ENABLED)
    set(CMAKE_HIP_ARCHITECTURES "gfx90a;gfx908")
    include_directories(SYSTEM ${HIP_INCLUDE_DIRS}) # Silence warnings originating from HIP headers
    target_link_libraries(acm PUBLIC hip::host)

    set(CMAKE_HIP_ARCHITECTURES "gfx90a;gfx908")
    message(STATUS "Enabled HIP architectures (ACM): " ${CMAKE_HIP_ARCHITECTURES})
    set_source_files_properties(reduce.cu transform.cu pack.cu PROPERTIES LANGUAGE HIP)
    set_property(TARGET acm PROPERTY HIP_ARCHITECTURES ${CMAKE_HIP_ARCHITECTURES})
elseif(HOST_ONLY_MODE)
    target_compile_definitions(acm PUBLIC ACM_HOST_ONLY_MODE_ENABLED)
else()
    message(WARNING "Did not find CUDA or HIP, compiling in host-only mode")
    # add_compile_options(-DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP)
endif()
