#include "acr_utils.h"

namespace acr {

auto
get(const AcMeshInfo& info, const AcIntParam& param)
{
    return info.int_params[param];
}

auto
get(const AcMeshInfo& info, const AcInt3Param& param)
{
    return info.int3_params[param];
}

auto
get(const AcMeshInfo& info, const AcRealParam& param)
{
    return info.real_params[param];
}

auto
get(const AcMeshInfo& info, const AcReal3Param& param)
{
    return info.real3_params[param];
}

void
set(const AcIntParam& param, const int value, AcMeshInfo& info)
{
    info.int_params[param] = value;
}

void
set(const AcInt3Param& param, const int3& value, AcMeshInfo& info)
{
    info.int3_params[param] = value;
}

void
set(const AcRealParam& param, const AcReal value, AcMeshInfo& info)
{
    info.real_params[param] = value;
}

void
set(const AcReal3Param& param, const AcReal3& value, AcMeshInfo& info)
{
    info.real3_params[param] = value;
}

Shape
get_global_nn(const AcMeshInfo& info)
{
    ERRCHK(acr::get(info, AC_global_nx) > 0);
    ERRCHK(acr::get(info, AC_global_ny) > 0);
    ERRCHK(acr::get(info, AC_global_nz) > 0);
    return Shape{as<uint64_t>(acr::get(info, AC_global_nx)),
                 as<uint64_t>(acr::get(info, AC_global_ny)),
                 as<uint64_t>(acr::get(info, AC_global_nz))};
}

Shape
get_local_nn(const AcMeshInfo& info)
{
    ERRCHK(acVerifyMeshInfo(info) == 0);
    return Shape{as<uint64_t>(acr::get(info, AC_nx)),
                 as<uint64_t>(acr::get(info, AC_ny)),
                 as<uint64_t>(acr::get(info, AC_nz))};
}

Shape
get_local_mm(const AcMeshInfo& info)
{
    ERRCHK(acVerifyMeshInfo(info) == 0);
    return Shape{as<uint64_t>(acr::get(info, AC_mx)),
                 as<uint64_t>(acr::get(info, AC_my)),
                 as<uint64_t>(acr::get(info, AC_mz))};
}

Dims
get_global_ss(const AcMeshInfo& info)
{
    ERRCHK(acr::get(info, AC_global_sx) > 0);
    ERRCHK(acr::get(info, AC_global_sy) > 0);
    ERRCHK(acr::get(info, AC_global_sz) > 0);
    return Dims{static_cast<AcReal>(acr::get(info, AC_global_sx)),
                static_cast<AcReal>(acr::get(info, AC_global_sy)),
                static_cast<AcReal>(acr::get(info, AC_global_sz))};
}

Index
get_global_nn_offset(const AcMeshInfo& info)
{
    ERRCHK(acVerifyMeshInfo(info) == 0);
    return Index{as<uint64_t>(acr::get(info, AC_multigpu_offset).x),
                 as<uint64_t>(acr::get(info, AC_multigpu_offset).y),
                 as<uint64_t>(acr::get(info, AC_multigpu_offset).z)};
}

Index
get_local_nn_offset()
{
    return Index{(STENCIL_WIDTH - 1) / 2, (STENCIL_HEIGHT - 1) / 2, (STENCIL_DEPTH - 1) / 2};
}

Index
get_local_rr()
{
    return get_local_nn_offset();
}

} // namespace acr
