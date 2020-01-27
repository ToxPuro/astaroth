## Astaroth Core
find_package(CUDA)
find_package(MPI)

add_library(astaroth_core STATIC device.cc node.cc astaroth.cc)
target_link_libraries(astaroth_core astaroth_utils astaroth_kernels MPI::MPI_CXX ${CUDA_LIBRARIES})
#target_include_directories(astaroth_core)
