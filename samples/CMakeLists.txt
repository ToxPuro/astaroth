if (NOT BUILD_TESTS)
    add_subdirectory(ctest)
    add_subdirectory(cpptest)

    #TP: we should make a wrapper functions for the CUDA calls
    add_subdirectory(benchmark-device)
    add_subdirectory(benchmark)

    add_subdirectory(mpi-io)
    add_subdirectory(mpi-io-multithreaded)
    add_subdirectory(pc-varfile-import)

    #add_subdirectory(stress)
    #add_subdirectory(standalone)
    #add_subdirectory(standalone_mpi)
    #add_subdirectory(ctest)
    #add_subdirectory(cpptest)
    #add_subdirectory(mpitest)
    #add_subdirectory(taskgraph_example)
    #add_subdirectory(taskgraph_print)
    #add_subdirectory(taskgraph_trace)
    #add_subdirectory(taskgraph_test)
    #add_subdirectory(mpi_fullgriderror_test)
    #add_subdirectory(benchmark)
    #add_subdirectory(benchmark-device)
    #add_subdirectory(boundcond_test)

    #add_subdirectory(mpi-io)
    #add_subdirectory(genbenchmarkscripts)
    #add_subdirectory(mpi_reduce_bench)
    #add_subdirectory(fortrantest)

    if (NOT RUNTIME_COMPILATION)
        add_subdirectory(benchmark-node)
    endif()

    if (NOT CPU_BUILD)
        add_subdirectory(microbenchmark)
    endif()

    #if (MPI_ENABLED)
    #  add_subdirectory(bwtest)
    #endif()
endif()

if (BUILD_MLLIB_SAMPLES)
    # Workaround for finding miopen on Triton
    link_directories(/opt/rocm/miopen/lib/)
    add_subdirectory(microbenchmark-nn)
endif ()

if (BUILD_MHD_SAMPLES)
    # add_subdirectory(standalone)
    # add_subdirectory(standalone_mpi)
    add_subdirectory(mpitest)
    add_subdirectory(devicetest)
    #add_subdirectory(stencil-loader)
endif()
