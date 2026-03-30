add_executable(
ac_run_mpi
	main.cc
	simulation_rng.cc
)

set_property(TARGET ac_run_mpi PROPERTY POSITION_INDEPENDENT_CODE ON)

target_include_directories(
ac_run_mpi
PRIVATE
	include
	include/rapidcsv
)

target_link_libraries(
ac_run_mpi
	astaroth_utils
	astaroth_core
)
