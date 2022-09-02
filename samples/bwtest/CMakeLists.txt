add_executable(bwtest bwtest.cu)
add_executable(bwtest-stencil bwtest-stencil.cu)
add_executable(bwtest-benchmark bwtest-benchmark.cu)

if (USE_HIP)
set_source_files_properties(bwtest.cu bwtest-stencil.cu bwtest-benchmark.cu PROPERTIES LANGUAGE CXX)
endif()

target_link_libraries(bwtest acc-runtime-headers)
target_link_libraries(bwtest-stencil acc-runtime-headers)
target_link_libraries(bwtest-benchmark acc-runtime-headers)
