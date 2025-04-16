## Packages
set(MPI_CXX_SKIP_MPICXX ON) # Skip MPI C++ bindings
find_package(MPI REQUIRED COMPONENTS C)

## Benchmarks
add_executable(bm_pack bm_pack.cc bm.cc)
target_link_libraries(bm_pack PRIVATE acm MPI::MPI_C)

add_executable(bm_rank_reordering bm_rank_reordering.cc bm.cc)
target_link_libraries(bm_rank_reordering PRIVATE acm MPI::MPI_C)

add_executable(bm_scaling bm_scaling.cc bm.cc)
target_link_libraries(bm_scaling PRIVATE acm MPI::MPI_C)

