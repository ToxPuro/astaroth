#include "acr_utils.h"

namespace acr {

int
get(const AcMeshInfo& info, const AcIntParam& param)
{
    return info.int_params[param];
}

int3
get(const AcMeshInfo& info, const AcInt3Param& param)
{
    return info.int3_params[param];
}

AcReal
get(const AcMeshInfo& info, const AcRealParam& param)
{
    return info.real_params[param];
}

AcReal3
get(const AcMeshInfo& info, const AcReal3Param& param)
{
    return info.real3_params[param];
}

// std::vector<ac::device_view<AcReal>>
// get(const VertexBufferArray& vba, const std::vector<Field>& fields, const BufferGroup& group)
// {
//     const auto count{vba.mx * vba.my * vba.mz};

//     std::vector<ac::device_view<AcReal>> output;

//     for (const auto& field : fields) {
//         switch (group) {
//         case BufferGroup::input:
//             output.push_back(ac::device_view<AcReal>{count, vba.in[field]});
//             break;
//         case BufferGroup::output:
//             output.push_back(ac::device_view<AcReal>{count, vba.out[field]});
//             break;
//         default:
//             ERRCHK(false);
//         }
//     }

//     return output;
// }

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

ac::shape
get_global_nn(const AcMeshInfo& info)
{
    ERRCHK(acr::get(info, AC_global_nx) > 0);
    ERRCHK(acr::get(info, AC_global_ny) > 0);
    ERRCHK(acr::get(info, AC_global_nz) > 0);
    return ac::shape{as<uint64_t>(acr::get(info, AC_global_nx)),
                     as<uint64_t>(acr::get(info, AC_global_ny)),
                     as<uint64_t>(acr::get(info, AC_global_nz))};
}

ac::shape
get_local_nn(const AcMeshInfo& info)
{
    ERRCHK(acVerifyMeshInfo(info) == 0);
    return ac::shape{as<uint64_t>(acr::get(info, AC_nx)),
                     as<uint64_t>(acr::get(info, AC_ny)),
                     as<uint64_t>(acr::get(info, AC_nz))};
}

ac::shape
get_local_mm(const AcMeshInfo& info)
{
    ERRCHK(acVerifyMeshInfo(info) == 0);
    return ac::shape{as<uint64_t>(acr::get(info, AC_mx)),
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

ac::index
get_global_nn_offset(const AcMeshInfo& info)
{
    ERRCHK(acVerifyMeshInfo(info) == 0);
    return ac::index{as<uint64_t>(acr::get(info, AC_multigpu_offset).x),
                     as<uint64_t>(acr::get(info, AC_multigpu_offset).y),
                     as<uint64_t>(acr::get(info, AC_multigpu_offset).z)};
}

ac::index
get_local_nn_offset()
{
    return ac::index{(STENCIL_WIDTH - 1) / 2, (STENCIL_HEIGHT - 1) / 2, (STENCIL_DEPTH - 1) / 2};
}

ac::index
get_local_rr()
{
    return get_local_nn_offset();
}

ac::device_view<AcReal>
make_ptr(const VertexBufferArray& vba, const Field& field, const BufferGroup& type)
{
    const size_t count{vba.mx * vba.my * vba.mz};

    switch (type) {
    case BufferGroup::input:
        return ac::device_view<AcReal>{count, vba.in[field]};
    case BufferGroup::output:
        return ac::device_view<AcReal>{count, vba.out[field]};
    default:
        ERRCHK(false);
        return ac::device_view<AcReal>{0, nullptr};
    }
}

ac::device_view<AcReal>
make_ptr(const VertexBufferArray& vba, const Profile& profile, const BufferGroup& type)
{
    const size_t count{vba.profiles.count};

    switch (type) {
    case BufferGroup::input:
        return ac::device_view<AcReal>{count, vba.profiles.in[profile]};
    case BufferGroup::output:
        return ac::device_view<AcReal>{count, vba.profiles.out[profile]};
    default:
        ERRCHK(false);
        return ac::device_view<AcReal>{0, nullptr};
    }
}

std::vector<ac::device_view<AcReal>>
get_ptrs(const VertexBufferArray& vba, const std::vector<Field>& fields, const BufferGroup& type)
{
    std::vector<ac::device_view<AcReal>> ptrs;

    for (const auto& field : fields)
        ptrs.push_back(make_ptr(vba, field, type));

    return ptrs;
}

std::vector<ac::device_view<AcReal>>
get_ptrs(const VertexBufferArray& vba, const std::vector<Profile>& profiles,
         const BufferGroup& type)
{
    std::vector<ac::device_view<AcReal>> ptrs;

    for (const auto& profile : profiles)
        ptrs.push_back(make_ptr(vba, profile, type));

    return ptrs;
}

std::vector<std::string>
get_strs(const std::vector<Field>& fields)
{
    std::vector<std::string> paths;

    for (const auto& field : fields)
        paths.push_back(std::string(field_names[field]));

    return paths;
}

} // namespace acr
