#include "halo_exchange.h"
#include "halo_exchange_packed.h"

#include "partition.h"

void
test_halo_exchange(void)
{

    // // Partition the domain
    // auto segments{partition(local_mm, local_nn, rr)};

    // // Prune the segment containing the computational domain
    // for (size_t i{0}; i < segments.size(); ++i) {
    //     if (within_box(segments[i].offset, local_nn, rr)) {
    //         segments.erase(segments.begin() + as<long>(i));
    //         --i;
    //     }
    // }

    // NdArray<int, ac::mr::host_memory_resource> hin()
    WARNING_DESC("Not implemented");
}
