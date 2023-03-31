add_executable(
ac_run_mpi
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

if(AC_RUN_MPI_SOMA_INTEGRATION)
     find_package(soma)
     target_link_libraries(
     ac_run_mpi
             soma-client
     )
     target_compile_definitions(ac_run_mpi -DAC_SOMA_INTEGRATION=1)
	    
endif()
