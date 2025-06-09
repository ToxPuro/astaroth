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

static int3
convert_to_int3(const ac::ntuple<uint64_t>& in)
{
    ERRCHK(in.size() == 3);
    return int3{as<int>(in[0]), as<int>(in[1]), as<int>(in[2])};
}

void set_global_nn(const ac::shape& global_nn, AcMeshInfo& info)
{
    ERRCHK(global_nn.size() == 3);
    acr::set(AC_global_nx, as<int>(global_nn[0]), info);
    acr::set(AC_global_ny, as<int>(global_nn[1]), info);
    acr::set(AC_global_nz, as<int>(global_nn[2]), info);

    // Backwards compatibility
    acr::set(AC_global_grid_n, convert_to_int3(global_nn), info);
}

void set_local_nn(const ac::shape& local_nn, AcMeshInfo& info){
    ERRCHK(local_nn.size() == 3);
    acr::set(AC_nx, as<int>(local_nn[0]), info);
    acr::set(AC_ny, as<int>(local_nn[1]), info);
    acr::set(AC_nz, as<int>(local_nn[2]), info);
}

void set_local_ss(const ac::shape& local_ss, AcMeshInfo& info)
{
    ERRCHK(local_ss.size() == 3);
    acr::set(AC_sx, static_cast<AcReal>(local_ss[0]), info);
    acr::set(AC_sy, static_cast<AcReal>(local_ss[1]), info);
    acr::set(AC_sz, static_cast<AcReal>(local_ss[2]), info);
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
        return ac::device_view<AcReal>{};
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
        return ac::device_view<AcReal>{};
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
