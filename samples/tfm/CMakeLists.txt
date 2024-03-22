## TFM
add_executable(tfm tfm.c)
target_link_libraries(tfm astaroth_core astaroth_utils)

add_executable(tfm_mpi tfm_mpi.c)
target_link_libraries(tfm_mpi astaroth_core astaroth_utils)

add_executable(tfm_benchmark tfm_benchmark.c)
target_link_libraries(tfm_benchmark astaroth_core astaroth_utils)

add_executable(tfm_pipeline tfm_pipeline.c)
target_link_libraries(tfm_pipeline astaroth_core astaroth_utils)
