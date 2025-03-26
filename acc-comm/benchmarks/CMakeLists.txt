## Packages
set(MPI_CXX_SKIP_MPICXX ON) # Skip MPI C++ bindings
find_package(MPI REQUIRED COMPONENTS C)

## Benchmarks
add_executable(bm_pack bm_pack.cc bm.cc)
target_link_libraries(bm_pack PRIVATE MPI::MPI_C acm)

add_executable(bm_scaling bm_scaling.cc bm.cc)
target_link_libraries(bm_scaling PRIVATE MPI::MPI_C roctracer64 acm)

add_executable(bm_transfer bm_transfer.cc bm.cc)
target_link_libraries(bm_transfer PRIVATE MPI::MPI_C acm)
