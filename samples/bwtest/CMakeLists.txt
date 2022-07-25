add_executable(bwtest main.cu)
add_executable(bwtest-stencil main2.cu)

if (USE_HIP)
set_source_files_properties(main.cu main2.cu PROPERTIES LANGUAGE CXX)
endif()

target_link_libraries(bwtest acc-runtime-headers)
target_link_libraries(bwtest-stencil acc-runtime-headers)