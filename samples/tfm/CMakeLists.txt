## TFM
add_executable(tfm tfm.c)
target_link_libraries(tfm astaroth_core astaroth_utils)
target_include_directories(tfm PRIVATE ${CMAKE_SOURCE_DIR})

# add_executable(tfm_mpi tfm_mpi.c)
# target_link_libraries(tfm_mpi astaroth_core astaroth_utils)
# target_include_directories(tfm_mpi PRIVATE ${CMAKE_SOURCE_DIR})

add_executable(tfm_benchmark tfm_benchmark.c)
target_link_libraries(tfm_benchmark astaroth_core astaroth_utils)
target_include_directories(tfm_benchmark PRIVATE ${CMAKE_SOURCE_DIR})

add_executable(tfm_pipeline tfm_pipeline.c)
target_link_libraries(tfm_pipeline astaroth_core astaroth_utils)
target_include_directories(tfm_pipeline PRIVATE ${CMAKE_SOURCE_DIR})

# add_subdirectory(${CMAKE_SOURCE_DIR}/3rdparty/inih 3rdparty/inih)
add_executable(tfm_standalone tfm_standalone.c)
target_link_libraries(tfm_standalone astaroth_core astaroth_utils astaroth_forcing inih)
target_include_directories(tfm_standalone PRIVATE ${CMAKE_SOURCE_DIR}/3rdparty/inih)
target_include_directories(tfm_standalone PRIVATE ${CMAKE_SOURCE_DIR})
target_compile_definitions(tfm_standalone PRIVATE AC_DEFAULT_TFM_CONFIG="${DSL_MODULE_DIR}/mhd.ini")
