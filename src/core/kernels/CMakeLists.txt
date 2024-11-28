## Astaroth Kernels
add_library(kernels STATIC kernels.cu)
target_link_libraries(kernels PUBLIC acc-runtime-headers)
target_include_directories(kernels PUBLIC ${CMAKE_SOURCE_DIR}/include)

if (USE_HIP)
    set_source_files_properties(kernels.cu PROPERTIES LANGUAGE HIP)
    set_property(TARGET kernels PROPERTY HIP_ARCHITECTURES ${CMAKE_HIP_ARCHITECTURES})
else()
    set_property(TARGET kernels PROPERTY CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})
endif()
