#pragma once

#include "astaroth_grid.h"

#include <memory>

#include "math_utils.h"

typedef struct Grid {
    Device device;
    AcMesh submesh;         // Submesh in host memory. Used as scratch space.
    uint3_64 decomposition; // For backwards compatibility. Should use AcDecompositionInfo.
    bool initialized;
    Volume nn;
    std::shared_ptr<AcTaskGraph> default_tasks;
    std::shared_ptr<AcTaskGraph> halo_exchange_tasks;
    std::shared_ptr<AcTaskGraph> periodic_bc_tasks;
    size_t mpi_tag_space_count;
    bool mpi_initialized;
    bool vertex_buffer_copied_from_user[NUM_VTXBUF_HANDLES]{};
    std::vector<KernelAnalysisInfo> kernel_analysis_info{};
} Grid;
