## TFM
add_executable(tfm tfm.c)
target_link_libraries(tfm astaroth_core astaroth_utils)

add_executable(tfm_mpi tfm.c)
target_link_libraries(tfm_mpi astaroth_core astaroth_utils)
