add_executable(microbenchmark microbenchmark.cu)

# Include directories
target_include_directories(microbenchmark PRIVATE ${CMAKE_SOURCE_DIR}/acc-runtime/api)
target_include_directories(microbenchmark PRIVATE ${CMAKE_SOURCE_DIR}/include)

if (USE_HIP)
    set_source_files_properties(microbenchmark.cu PROPERTIES LANGUAGE HIP)

    #ROC_TRACER_PATH comes from the first CMakeLists.txt
    # Roctracer
    include_directories(${ROC_TRACER_PATH})
    link_directories(${ROC_TRACER_PATH})
    target_link_libraries(microbenchmark roctracer64)
endif()

# Default values
if(USE_SMEM)
    set(USE_SMEM 1)
else()
    set(USE_SMEM 0)
endif()

if(NOT DEFINED MAX_THREADS_PER_BLOCK)
    set(MAX_THREADS_PER_BLOCK 0)
endif()

## Options
# cmake -DUSE_SMEM=1 .. to use smem in bwtest-benchmark
target_compile_options(microbenchmark PRIVATE -DUSE_SMEM=${USE_SMEM})
# cmake -DMAX_THREADS_PER_BLOCK=192 .. to set launch bounds in bwtest-benchmark
target_compile_options(microbenchmark PRIVATE -DMAX_THREADS_PER_BLOCK=${MAX_THREADS_PER_BLOCK})

if (NOT DEFINED MB_IMPLEMENTATION)
    set(MB_IMPLEMENTATION 1)
endif()
target_compile_options(microbenchmark PRIVATE -DMB_IMPLEMENTATION=${MB_IMPLEMENTATION})

