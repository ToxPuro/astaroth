add_executable(bwtest main.c)
target_link_libraries(bwtest acc-runtime-headers)
target_compile_options(bwtest PRIVATE -O3)
