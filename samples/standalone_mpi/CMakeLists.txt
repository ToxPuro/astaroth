add_executable(ac_run_mpi)
if(AC_RUN_MPI_SOMA_INTEGRATION)
    #set(MPI_CXX_SKIP_MPICXX ON)
    #set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -DOMPI_SKIP_MPICXX")
    find_package(MPI REQUIRED COMPONENTS C CXX)

    find_package(soma)
    target_link_libraries(
    ac_run_mpi
        soma-client
    )
    #target_compile_definitions(ac_run_mpi PUBLIC AC_SOMA_INTEGRATION=1 OMPI_SKIP_MPICXX=1)
    target_compile_definitions(ac_run_mpi PUBLIC AC_SOMA_INTEGRATION=1)
    target_sources(ac_run_mpi
    PRIVATE
        simulation_soma_integration.cc
    )
endif()

target_sources(ac_run_mpi
PRIVATE
	main.cc
	host_memory.cc
	host_forcing.cc
	config_loader.cc
	simulation_rng.cc
)

target_include_directories(
ac_run_mpi
PRIVATE
	include
)

target_link_libraries(
ac_run_mpi
	astaroth_utils
	astaroth_core
)

