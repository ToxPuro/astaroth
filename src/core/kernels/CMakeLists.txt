## Astaroth Kernels
if (USE_HIP)
    set_source_files_properties(kernels.cu PROPERTIES LANGUAGE HIP)
endif()

add_library(kernels kernels.cu)
target_link_libraries(kernels PUBLIC acc-runtime-headers)
target_compile_features(kernels PRIVATE cxx_std_17)
target_include_directories(kernels PUBLIC ${CMAKE_SOURCE_DIR}/include)
