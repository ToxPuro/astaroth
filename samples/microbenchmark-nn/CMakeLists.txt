

if (USE_HIP)
    add_executable(microbenchmark-nn backend-miopen.cu array.cu)

    set_source_files_properties(microbenchmark-nn.cu PROPERTIES LANGUAGE CXX)
    set_source_files_properties(backend-miopen.cu PROPERTIES LANGUAGE CXX)
    set_source_files_properties(array.cu PROPERTIES LANGUAGE CXX)

    # Roctracer
    include_directories(/opt/rocm/roctracer/include)
    link_directories(/opt/rocm/roctracer/lib)
    target_link_libraries(microbenchmark-nn roctracer64)

    include_directories(/opt/rocm/include/)
    include_directories(/opt/rocm/lib/)
    target_link_libraries(microbenchmark-nn MIOpen)
else ()
    add_executable(microbenchmark-nn microbenchmark-nn.cu array.cu)
    target_link_libraries(microbenchmark-nn cudnn)
endif()

# Include directories
target_include_directories(microbenchmark-nn PRIVATE ${CMAKE_SOURCE_DIR}/acc-runtime/api)