

if (USE_HIP)
    find_package(HIP REQUIRED)
    add_executable(microbenchmark-nn microbenchmark-nn.cu array.cu backend-miopen.cu)

    set_source_files_properties(microbenchmark-nn.cu PROPERTIES LANGUAGE HIP)
    set_source_files_properties(backend-miopen.cu PROPERTIES LANGUAGE HIP)
    set_source_files_properties(array.cu PROPERTIES LANGUAGE HIP)

    # Roctracer
    include_directories(${ROCM_PATH}/roctracer/include)
    link_directories(${ROCM_PATH}/roctracer/lib)
    target_link_libraries(microbenchmark-nn ${ROCTRACER})

    include_directories(${ROCM_PATH}/include/)
    link_directories(${ROCM_PATH}/lib/)
    target_link_libraries(microbenchmark-nn MIOpen)
else ()
    add_executable(microbenchmark-nn microbenchmark-nn.cu array.cu backend-cudnn.cu)
    target_link_libraries(microbenchmark-nn cudnn)
endif()

# Include directories
target_include_directories(microbenchmark-nn PRIVATE ${CMAKE_SOURCE_DIR}/acc-runtime/api)
