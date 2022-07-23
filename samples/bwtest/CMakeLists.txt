add_executable(bwtest main.cu)

if (USE_HIP)
set_source_files_properties(main.cu PROPERTIES LANGUAGE CXX)
endif()

target_link_libraries(bwtest acc-runtime-headers)