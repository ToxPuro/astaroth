
find_package(CUDAToolkit QUIET)
find_package(HIP QUIET)
add_library(device-libraries INTERFACE)
if(NOT HOST_ONLY_MODE AND CUDAToolkit_FOUND)
    message(STATUS "CUDA found")
    enable_language(CUDA)
    add_compile_options(-DDEVICE_ENABLED)
    add_compile_options(-DCUDA_ENABLED)
    set(CMAKE_CUDA_ARCHITECTURES "60;61;70;80")
    include_directories(SYSTEM ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}) # Silence warnings originating from CUDA headers
    
    target_link_libraries(device-libraries INTERFACE CUDA::cudart)
elseif(NOT HOST_ONLY_MODE AND HIP_FOUND)
    message(STATUS "HIP found")
    enable_language(HIP)
    add_compile_options(-DDEVICE_ENABLED)
    add_compile_options(-DHIP_ENABLED)
    set(CMAKE_HIP_ARCHITECTURES "gfx90a;gfx908")
    include_directories(SYSTEM ${HIP_INCLUDE_DIRS}) # Silence warnings originating from HIP headers

    message(WARNING ${HIP_INCLUDE_DIRS})

    set_source_files_properties(pack.cu PROPERTIES LANGUAGE HIP)
    target_link_libraries(device-libraries INTERFACE hip::host)
else()
    message(WARNING "Did not find CUDA or HIP, compiling in host-only mode")
    add_compile_options(-DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP)
    # add_library(device-module STATIC pack.cc)
endif()

## Packages
find_package(MPI REQUIRED COMPONENTS C)
add_library(acm SHARED acm.cc acm_error.c halo_exchange.cc packet.cc mpi_utils.cc buffer_exchange.cc pack.cu pack.cc ndbuffer.cc buffer.cc decomp.cc partition.cc segment.cc math_utils.cc vector.cc static_array.cc array.cc type_conversion.cc errchk_print.c)
target_link_libraries(acm PUBLIC MPI::MPI_C device-libraries)
target_include_directories(acm PUBLIC ${PROJECT_SOURCE_DIR})

