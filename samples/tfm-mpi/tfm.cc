#include <cmath>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <numeric>

#include "astaroth_headers.h" // To suppress unnecessary warnings

#include "astaroth_forcing.h"
#include "device_detail.h"

#include "acm/detail/errchk_mpi.h"
#include "acm/detail/mpi_utils.h"
#include "acm/detail/type_conversion.h"

#include "acm/detail/allocator.h"

#include "acm/detail/halo_exchange_packed.h"
#include "acm/detail/io.h"

#include "acm/detail/ndbuffer.h"

#include "stencil_loader.h"
#include "tfm_utils.h"

#include "acr_utils.h"
#include "device_utils.h"

#include <mpi.h>

#include <algorithm>
#include <random>

// #define AC_ENABLE_ASYNC_AVERAGES
#define AC_WRITE_SYNCHRONOUS_SNAPSHOTS
#define AC_WRITE_SYNCHRONOUS_PROFILES
#define AC_WRITE_SYNCHRONOUS_TIMESERIES
#define AC_WRITE_SYNCHRONOUS_SLICES

// #define AC_DISABLE_IO

using HaloExchangeTask = ac::comm::async_halo_exchange_task<AcReal, ac::mr::device_allocator>;
using IOTask           = ac::io::batched_async_write_task<AcReal, ac::mr::pinned_host_allocator>;

/** Concatenates the field name and ".mesh" of a vector of handles */
static auto
get_field_paths(const std::vector<Field>& fields, const size_t step)
{
    std::vector<std::string> paths;
    for (const auto& field : fields) {
        std::ostringstream oss;
        oss << field_names[static_cast<size_t>(field)] << "-step-" << std::setfill('0')
            << std::setw(12) << step << ".mesh";
        paths.push_back(oss.str());
    }
    return paths;
}

/** Apply a static cast to all elements of the input vector from type U to T */
template <typename T, typename U>
ac::ntuple<T>
static_cast_vec(const ac::ntuple<U>& in)
{
    ac::ntuple<T> out{ac::make_ntuple<T>(in.size(), 0)};
    for (size_t i{0}; i < in.size(); ++i)
        out[i] = static_cast<T>(in[i]);
    return out;
}

/** Appends two vectors and returns an appended vector */
template <typename T>
std::vector<T>
appended(const std::vector<T>& a, const std::vector<T>& b)
{
    std::vector<T> c{a};
    c.insert(c.end(), b.begin(), b.end());
    return c;
}

static int3
convert_to_int3(const ac::ntuple<uint64_t>& in)
{
    ERRCHK(in.size() == 3);
    return int3{as<int>(in[0]), as<int>(in[1]), as<int>(in[2])};
}

/**
 * Decompose the domain set in the input info and returns a complete local mesh info with all
 * parameters set (incl. multi-device offsets, and others)
 */
static AcMeshInfo
get_local_mesh_info(const MPI_Comm& cart_comm, const AcMeshInfo& info)
{
    // Calculate local dimensions
    ac::shape global_nn{acr::get_global_nn(info)};
    Dims global_ss{acr::get_global_ss(info)};

    const ac::shape decomp{ac::mpi::get_decomposition(cart_comm)};
    const ac::shape local_nn{global_nn / decomp};
    const Dims local_ss{global_ss / static_cast_vec<AcReal>(decomp)};

    const ac::index coords{ac::mpi::get_coords(cart_comm)};
    const ac::index global_nn_offset{coords * local_nn};

    // Fill AcMeshInfo
    AcMeshInfo local_info{info};

    acr::set(AC_nx, as<int>(local_nn[0]), local_info);
    acr::set(AC_ny, as<int>(local_nn[1]), local_info);
    acr::set(AC_nz, as<int>(local_nn[2]), local_info);

    local_info.int3_params[AC_multigpu_offset] = convert_to_int3(global_nn_offset);

    acr::set(AC_sx, static_cast<AcReal>(local_ss[0]), local_info);
    acr::set(AC_sy, static_cast<AcReal>(local_ss[1]), local_info);
    acr::set(AC_sz, static_cast<AcReal>(local_ss[2]), local_info);

    // Backwards compatibility
    acr::set(AC_global_grid_n, convert_to_int3(global_nn), local_info);

    ERRCHK(acHostUpdateLocalBuiltinParams(&local_info) == 0);
    ERRCHK(acHostUpdateMHDSpecificParams(&local_info) == 0);
    ERRCHK(acHostUpdateTFMSpecificGlobalParams(&local_info) == 0);

    // Others to ensure nothing is left uninitialized
    acr::set(AC_init_type, 0, local_info);
    // acr::set(AC_step_number, 0, local_info);
    acr::set(AC_dt, 0, local_info);
    acr::set(AC_dummy_real3, AcReal3{0, 0, 0}, local_info);

    // Special: exclude inner domain (used to fuse outer integration)
    acr::set(AC_exclude_inner, 0, local_info);

    ERRCHK(acVerifyMeshInfo(local_info) == 0);
    return local_info;
}

/**
 * Initialize and load test field profiles on the device based on AC_profile_amplitude and
 * AC_profile_wavenumber set in the info structure associated with the device.
 */
static int
init_tfm_profiles(const Device& device)
{
    PRINT_LOG_DEBUG("Enter");
    AcMeshInfo info{};
    ERRCHK_AC(acDeviceGetLocalConfig(device, &info));

    const auto dsz{acr::get(info, AC_dsz)};
    const long offset{-acr::get(info, AC_nz_min) + acr::get(info, AC_multigpu_offset).z};
    const size_t local_mz{as<size_t>(acr::get(info, AC_mz))};

    const AcReal amplitude{acr::get(info, AC_profile_amplitude)};
    const AcReal wavenumber{acr::get(info, AC_profile_wavenumber)};

    auto host_profile{std::make_unique<AcReal[]>(local_mz)};

    // All to zero
    acHostInitProfileToValue(0, local_mz, host_profile.get());
    for (size_t profile{0}; profile < NUM_PROFILES; ++profile) {
        ERRCHK_AC(acDeviceLoadProfile(device,
                                      host_profile.get(),
                                      local_mz,
                                      static_cast<Profile>(profile)));
    }

    // B1c (here B11) and B2c (here B21) to cosine
    acHostInitProfileToCosineWave(dsz,
                                  offset,
                                  amplitude,
                                  wavenumber,
                                  local_mz,
                                  host_profile.get());
    ERRCHK_AC(acDeviceLoadProfile(device, host_profile.get(), local_mz, PROFILE_B11mean_x));
    ERRCHK_AC(acDeviceLoadProfile(device, host_profile.get(), local_mz, PROFILE_B21mean_y));

    // B1s (here B12) and B2s (here B22)
    acHostInitProfileToSineWave(dsz,
                                offset,
                                amplitude,
                                wavenumber,
                                local_mz,
                                host_profile.get());
    ERRCHK_AC(acDeviceLoadProfile(device, host_profile.get(), local_mz, PROFILE_B12mean_x));
    ERRCHK_AC(acDeviceLoadProfile(device, host_profile.get(), local_mz, PROFILE_B22mean_y));

    return 0;
}

/** Synchronous write snapshots to disk */
static int
write_snapshots_to_disk(const MPI_Comm& parent_comm, const Device& device, const size_t step)
{
    PRINT_LOG_DEBUG("Enter");
    VertexBufferArray vba{};
    ERRCHK_AC(acDeviceGetVBA(device, &vba));

    AcMeshInfo local_info{};
    ERRCHK_AC(acDeviceGetLocalConfig(device, &local_info));

    const auto local_mm{acr::get_local_mm(local_info)};
    const auto local_mm_count{prod(local_mm)};
    ac::buffer<AcReal, ac::mr::host_allocator> staging_buffer{local_mm_count};

    // Global mesh (collective)
    for (int i = 0; i < NUM_VTXBUF_HANDLES; ++i) {
        char filepath[4096];
        sprintf(filepath, "debug-step-%012zu-tfm-%s.mesh", step, vtxbuf_names[i]);
        PRINT_LOG_TRACE("Writing %s", filepath);
        ac::mr::copy(acr::make_ptr(vba, static_cast<Field>(i), BufferGroup::input),
                     staging_buffer.get());
        ac::mpi::write_collective_simple(parent_comm,
                                         ac::mpi::get_dtype<AcReal>(),
                                         acr::get_global_nn(local_info),
                                         acr::get_local_nn_offset(),
                                         staging_buffer.data(),
                                         std::string(filepath));
    }
    PRINT_LOG_TRACE("Exit");
    return 0;
}

static int
write_slices_to_disk(const MPI_Comm& parent_comm, const Device& device, const size_t step)
{
    AcMeshInfo info{};
    ERRCHK_AC(acDeviceGetLocalConfig(device, &info));

    VertexBufferArray vba{};
    ERRCHK_AC(acDeviceGetVBA(device, &vba));

    const auto global_nn{acr::get_global_nn(info)};
    const auto global_nn_offset{acr::get_global_nn_offset(info)};

    const auto local_mm{acr::get_local_mm(info)};
    const auto local_nn{acr::get_local_nn(info)};
    const auto local_rr{acr::get_local_rr()};

    const uint64_t global_pos{global_nn[2] / 2}; // Midpoint in the z axis
    const uint64_t global_pos_min{global_nn_offset[2]};
    const uint64_t global_pos_max{global_pos_min + local_nn[2]};

    const auto key{ac::mpi::get_rank(parent_comm)};
    int color{MPI_UNDEFINED};
    if (global_pos_min <= global_pos && global_pos < global_pos_max)
        color = 0;

    MPI_Comm neighbors{MPI_COMM_NULL};
    ERRCHK_MPI_API(MPI_Comm_split(parent_comm, color, key, &neighbors));
    if (neighbors == MPI_COMM_NULL)
        return 0; // Not contributing, return early

    const auto coords{ac::mpi::get_coords(parent_comm)};
    const ac::index slice_coords{coords[0], coords[1], 0};

    const uint64_t local_pos{global_pos - global_nn_offset[2]};
    const ac::shape local_slice_nn{local_nn[0], local_nn[1], 1};
    const ac::index local_slice_nn_offset{local_rr[0], local_rr[1], local_pos};
    const ac::shape global_slice_nn{global_nn[0], global_nn[1], 1};
    const ac::index global_slice_nn_offset{slice_coords * local_slice_nn};

    static ac::device_ndbuffer<AcReal> pack_buffer{local_slice_nn};
    static ac::host_ndbuffer<AcReal> staging_buffer{local_slice_nn};
    for (size_t i{0}; i < NUM_VTXBUF_HANDLES; ++i) {
        const auto input{acr::make_ptr(vba, static_cast<Field>(i), BufferGroup::input)};
        pack(local_mm, local_slice_nn, local_slice_nn_offset, {input}, pack_buffer.get());
        ac::mr::copy(pack_buffer.get(), staging_buffer.get());

        char filepath[4096];
        sprintf(filepath, "%s-%012zu.slice", vtxbuf_names[i], step);
        PRINT_LOG_TRACE("Writing %s", filepath);
        ac::mpi::write_collective(neighbors,
                                  ac::mpi::get_dtype<AcReal>(),
                                  global_slice_nn,
                                  global_slice_nn_offset,
                                  local_slice_nn,
                                  local_slice_nn,
                                  ac::make_index(local_slice_nn.size(), 0),
                                  staging_buffer.data(),
                                  std::string(filepath));
    }

    ERRCHK_MPI_API(MPI_Comm_free(&neighbors));
    return 0;
}

/**
 * Synchronous distributed write snapshots to disk.
 * Each process writes out their own, full mesh, to separate files
 */
static int
write_distributed_snapshots_to_disk(const MPI_Comm& parent_comm, const Device& device,
                                    const size_t step)
{
    PRINT_LOG_DEBUG("Enter");
    VertexBufferArray vba{};
    ERRCHK_AC(acDeviceGetVBA(device, &vba));

    AcMeshInfo local_info{};
    ERRCHK_AC(acDeviceGetLocalConfig(device, &local_info));

    const auto local_mm{acr::get_local_mm(local_info)};
    const auto local_mm_count{prod(local_mm)};
    ac::buffer<AcReal, ac::mr::host_allocator> staging_buffer{local_mm_count};

    // Local mesh incl. ghost zones
    for (int i{0}; i < NUM_VTXBUF_HANDLES; ++i) {
        char filepath[4096];
        sprintf(filepath,
                "proc-%d-debug-step-%012zu-tfm-%s.mesh-distributed",
                ac::mpi::get_rank(parent_comm),
                step,
                vtxbuf_names[i]);
        PRINT_LOG_TRACE("Writing %s", filepath);
        ac::mr::copy(acr::make_ptr(vba, static_cast<Field>(i), BufferGroup::input),
                     staging_buffer.get());
        ac::mpi::write_distributed(parent_comm,
                                   ac::mpi::get_dtype<AcReal>(),
                                   acr::get_local_mm(local_info),
                                   staging_buffer.data(),
                                   std::string(filepath));
    }
    PRINT_LOG_TRACE("Exit");
    return 0;
}

static int
write_profiles_to_disk(const MPI_Comm& parent_comm, const Device& device, const size_t step)
{
    PRINT_LOG_DEBUG("Enter");
    VertexBufferArray vba{};
    ERRCHK_AC(acDeviceGetVBA(device, &vba));

    AcMeshInfo local_info{};
    ERRCHK_AC(acDeviceGetLocalConfig(device, &local_info));

    const auto local_mm{acr::get_local_mm(local_info)};
    const auto local_mm_count{prod(local_mm)};
    ac::buffer<AcReal, ac::mr::host_allocator> staging_buffer{local_mm_count};

    for (int i = 0; i < NUM_PROFILES; ++i) {
        char filepath[4096];
        sprintf(filepath, "debug-step-%012zu-tfm-%s.profile", step, profile_names[i]);
        PRINT_LOG_TRACE("Writing %s", filepath);
        const ac::shape profile_global_nz{as<uint64_t>(acr::get(local_info, AC_global_nz))};
        const ac::shape profile_local_mz{as<uint64_t>(acr::get(local_info, AC_mz))};
        const ac::shape profile_local_nz{as<uint64_t>(acr::get(local_info, AC_nz))};
        const ac::shape profile_local_nz_offset{as<uint64_t>(acr::get(local_info, AC_nz_min))};
        const ac::index coords{ac::mpi::get_coords(parent_comm)[2]};
        const ac::shape profile_global_nz_offset{coords * profile_local_nz};

        const int rank{ac::mpi::get_rank(parent_comm)};
        const ac::index coords_3d{ac::mpi::get_coords(parent_comm)};
        const ac::shape decomp_3d{ac::mpi::get_decomposition(parent_comm)};
        const int color = (coords_3d[0] + coords_3d[1] * decomp_3d[0]) == 0 ? 0 : MPI_UNDEFINED;

        MPI_Comm profile_comm{MPI_COMM_NULL};
        ERRCHK_MPI_API(MPI_Comm_split(parent_comm, color, rank, &profile_comm));

        if (profile_comm != MPI_COMM_NULL) {
            ac::mr::copy(acr::make_ptr(vba, static_cast<Profile>(i), BufferGroup::input),
                         staging_buffer.get());
            ac::mpi::write_collective(profile_comm,
                                      ac::mpi::get_dtype<AcReal>(),
                                      profile_global_nz,
                                      profile_global_nz_offset,
                                      profile_local_mz,
                                      profile_local_nz,
                                      profile_local_nz_offset,
                                      staging_buffer.data(),
                                      std::string(filepath));
            ERRCHK_MPI_API(MPI_Comm_free(&profile_comm));
        }
    }

    PRINT_LOG_TRACE("Exit");
    return 0;
}

/** Write both the mesh snapshot and profiles synchronously to disk */
static int
write_diagnostic_step(const MPI_Comm& parent_comm, const Device& device, const size_t step)
{
    PRINT_LOG_DEBUG("Enter");
    write_snapshots_to_disk(parent_comm, device, step);
    // write_distributed_snapshots_to_disk(parent_comm, device, step);
    write_profiles_to_disk(parent_comm, device, step);
    PRINT_LOG_TRACE("Exit");
    return 0;
}

static MPI_Op
get_mpi_op(const ReductionType& rtype)
{
    switch (rtype) {
    case RTYPE_MAX:
        return MPI_MAX;
    case RTYPE_MIN:
        return MPI_MIN;
    case RTYPE_SUM: /* Fallthrough */
    case RTYPE_RMS:
        return MPI_SUM;
    default:
        return MPI_OP_NULL;
    }
}

static AcReal
reduce_vec(const MPI_Comm& parent_comm, const Device& device, const ReductionType& rtype,
           const Field& a, const Field& b, const Field& c)
{
    MPI_Comm comm{MPI_COMM_NULL};
    ERRCHK_MPI_API(MPI_Comm_dup(parent_comm, &comm));
    ERRCHK_MPI(comm != MPI_COMM_NULL);

    AcReal local_result{-1};
    ERRCHK_AC(acDeviceReduceVecNotAveraged(device, STREAM_DEFAULT, rtype, a, b, c, &local_result));

    AcReal global_result{-1};
    const int root{0};
    const MPI_Op op{get_mpi_op(rtype)};
    ERRCHK_MPI_API(MPI_Reduce(&local_result, &global_result, 1, AC_REAL_MPI_TYPE, op, root, comm));

    ERRCHK_MPI_API(MPI_Comm_free(&comm));
    return global_result;
}

static AcReal
reduce_scal(const MPI_Comm& parent_comm, const Device& device, const ReductionType& rtype,
            const Field& field)
{
    MPI_Comm comm{MPI_COMM_NULL};
    ERRCHK_MPI_API(MPI_Comm_dup(parent_comm, &comm));
    ERRCHK_MPI(comm != MPI_COMM_NULL);

    AcReal local_result{-1};
    ERRCHK_AC(acDeviceReduceScalNotAveraged(device, STREAM_DEFAULT, rtype, field, &local_result));

    AcReal global_result{-1};
    const int root{0};
    const MPI_Op op{get_mpi_op(rtype)};
    ERRCHK_MPI_API(MPI_Reduce(&local_result, &global_result, 1, AC_REAL_MPI_TYPE, op, root, comm));

    ERRCHK_MPI_API(MPI_Comm_free(&comm));
    return global_result;
}

#include "acm/detail/print_debug.h"

// Format std::printf("label, step, t_step, dt, min, rms, max\n");
static int
write_vec_timeseries(const MPI_Comm& parent_comm, const Device& device, const size_t step,
                     const AcReal simulation_time, const AcReal dt, const Field& a, const Field& b,
                     const Field& c, const std::string& label)
{
    AcMeshInfo info{};
    ERRCHK_AC(acDeviceGetLocalConfig(device, &info));
    const auto global_nn{acr::get_global_nn(info)};
    const auto count{prod(global_nn)};

    const AcReal vmax{reduce_vec(parent_comm, device, RTYPE_MAX, a, b, c)};
    const AcReal vmin{reduce_vec(parent_comm, device, RTYPE_MIN, a, b, c)};
    const AcReal vsqsum{reduce_vec(parent_comm, device, RTYPE_RMS, a, b, c)};
    const AcReal vrms{std::sqrt(vsqsum / count)};
    const AcReal vavg{reduce_vec(parent_comm, device, RTYPE_SUM, a, b, c) / count};

    const auto rank{ac::mpi::get_rank(parent_comm)};
    if (rank == 0) {
        std::printf("%-6s %5zu, %.3g, %.3g, %.3g, %.3g, %.3g, %.3g\n",
                    label.c_str(),
                    step,
                    static_cast<double>(simulation_time),
                    static_cast<double>(dt),
                    static_cast<double>(vmin),
                    static_cast<double>(vrms),
                    static_cast<double>(vmax),
                    static_cast<double>(vavg));

        FILE* fp{fopen("timeseries.csv", "a")};
        ERRCHK_MPI(fp != NULL);
        std::fprintf(fp,
                     "%s,%zu,%e,%e,%e,%e,%e,%e\n",
                     label.c_str(),
                     step,
                     static_cast<double>(simulation_time),
                     static_cast<double>(dt),
                     static_cast<double>(vmin),
                     static_cast<double>(vrms),
                     static_cast<double>(vmax),
                     static_cast<double>(vavg));
        ERRCHK_MPI(fclose(fp) == 0);
    }

    return 0;
}

// Format std::printf("label, step, t_step, dt, min, rms, max\n");
static int
write_scal_timeseries(const MPI_Comm& parent_comm, const Device& device, const size_t step,
                      const AcReal simulation_time, const AcReal dt, const Field& field)
{
    AcMeshInfo info{};
    ERRCHK_AC(acDeviceGetLocalConfig(device, &info));
    const auto global_nn{acr::get_global_nn(info)};
    const auto count{prod(global_nn)};

    const AcReal vmax{reduce_scal(parent_comm, device, RTYPE_MAX, field)};
    const AcReal vmin{reduce_scal(parent_comm, device, RTYPE_MIN, field)};
    const AcReal vsqsum{reduce_scal(parent_comm, device, RTYPE_RMS, field)};
    const AcReal vrms{std::sqrt(vsqsum / count)};
    const AcReal vavg{reduce_scal(parent_comm, device, RTYPE_SUM, field) / count};

    const auto rank{ac::mpi::get_rank(parent_comm)};
    if (rank == 0) {
        std::printf("%-6s %5zu, %.3g, %.3g, %.3g, %.3g, %.3g, %.3g\n",
                    field_names[field],
                    step,
                    static_cast<double>(simulation_time),
                    static_cast<double>(dt),
                    static_cast<double>(vmin),
                    static_cast<double>(vrms),
                    static_cast<double>(vmax),
                    static_cast<double>(vavg));

        FILE* fp{fopen("timeseries.csv", "a")};
        ERRCHK_MPI(fp != NULL);
        std::fprintf(fp,
                     "%s,%zu,%e,%e,%e,%e,%e,%e\n",
                     field_names[field],
                     step,
                     static_cast<double>(simulation_time),
                     static_cast<double>(dt),
                     static_cast<double>(vmin),
                     static_cast<double>(vrms),
                     static_cast<double>(vmax),
                     static_cast<double>(vavg));
        ERRCHK_MPI(fclose(fp) == 0);
    }

    return 0;
}

static int
write_timeseries(const MPI_Comm& parent_comm, const Device& device, const size_t step,
                 const AcReal simulation_time, const AcReal dt)
{
    PRINT_LOG_DEBUG("Enter");

    std::printf("label,step,t_step,dt,min,rms,max,avg\n");

    std::vector<std::vector<Field>> vecfields{
        {VTXBUF_UUX, VTXBUF_UUY, VTXBUF_UUZ},
        {TF_a11_x, TF_a11_y, TF_a11_z},
        {TF_a12_x, TF_a12_y, TF_a12_z},
        {TF_a21_x, TF_a21_y, TF_a21_z},
        {TF_a22_x, TF_a22_y, TF_a22_z},
        {TF_uxb11_x, TF_uxb11_y, TF_uxb11_z},
        {TF_uxb12_x, TF_uxb12_y, TF_uxb12_z},
        {TF_uxb21_x, TF_uxb21_y, TF_uxb21_z},
        {TF_uxb22_x, TF_uxb22_y, TF_uxb22_z},
#if LOO
        {VTXBUF_OOX, VTXBUF_OOY, VTXBUF_OOZ},
#endif
    };
    std::vector<std::string> vecfield_names{
        "uu",
        "TF_a11",
        "TF_a12",
        "TF_a21",
        "TF_a22",
        "TF_uxb11",
        "TF_uxb12",
        "TF_uxb21",
        "TF_uxb22",
#if LOO
        "curl(uu)",
#endif
    };
    ERRCHK(vecfields.size() == vecfield_names.size());
    for (size_t i{0}; i < vecfields.size(); ++i)
        write_vec_timeseries(parent_comm,
                             device,
                             step,
                             simulation_time,
                             dt,
                             vecfields[i][0],
                             vecfields[i][1],
                             vecfields[i][2],
                             vecfield_names[i]);

    for (size_t i{0}; i < NUM_FIELDS; ++i)
        write_scal_timeseries(parent_comm,
                              device,
                              step,
                              simulation_time,
                              dt,
                              static_cast<Field>(i));

    return 0;
}

/** Calculate the timestep length and distribute it to all devices in the grid */
static AcReal
calc_and_distribute_timestep(const MPI_Comm& parent_comm, const Device& device)
{
    PRINT_LOG_DEBUG("Enter");
    // VertexBufferArray vba{};
    // ERRCHK_AC(acDeviceGetVBA(device, &vba));

    AcMeshInfo info{};
    ERRCHK_AC(acDeviceGetLocalConfig(device, &info));

    AcReal uumax{0};
    AcReal vAmax{0};
    AcReal shock_max{0};

    static bool warning_shown{false};
    if (!warning_shown) {
        WARNING_DESC("vAmax and shock_max not used in timestepping, set to 0");
        warning_shown = true;
    }

    MPI_Comm comm{MPI_COMM_NULL};
    ERRCHK_MPI_API(MPI_Comm_dup(parent_comm, &comm));
    ERRCHK_MPI(comm != MPI_COMM_NULL);

    ERRCHK_AC(acDeviceReduceVec(device,
                                STREAM_DEFAULT,
                                RTYPE_MAX,
                                VTXBUF_UUX,
                                VTXBUF_UUY,
                                VTXBUF_UUZ,
                                &uumax));
    ERRCHK_MPI_API(MPI_Allreduce(MPI_IN_PLACE, &uumax, 1, AC_REAL_MPI_TYPE, MPI_MAX, comm));
    ERRCHK_MPI_API(MPI_Comm_free(&comm));

    return calc_timestep(uumax, vAmax, shock_max, info);
}

/**
 * Experimental: Partition the domain into segments based on the segment group.
 * Could be made more general by recursive partitioning instead of the current
 * hard-coded segmentation levels (ComputeInner, etc.)
 */
enum class SegmentGroup {
    halo,
    compute_outer,
    compute_inner,
    compute_full,
    full,
};

static std::vector<ac::segment>
partition(const ac::shape& local_mm, const ac::shape& local_nn, const ac::shape& local_rr,
          const SegmentGroup& group)
{
    PRINT_LOG_TRACE("Enter");
    switch (group) {
    case SegmentGroup::halo: {
        const ac::shape mm{local_mm};
        const ac::shape nn{local_nn};
        const ac::shape rr{local_rr};

        auto segments{partition(mm, nn, rr)};
        auto it{
            std::remove_if(segments.begin(), segments.end(), [nn, rr](const ac::segment& segment) {
                return within_box(segment.offset, nn, rr);
            })};
        segments.erase(it, segments.end());
        return segments;
    }
    case SegmentGroup::compute_outer: {
        const ac::shape mm{local_nn};
        const ac::shape nn{local_nn - 2 * local_rr};
        const ac::shape rr{local_rr};

        auto segments{partition(mm, nn, rr)};
        auto it{
            std::remove_if(segments.begin(), segments.end(), [nn, rr](const ac::segment& segment) {
                return within_box(segment.offset, nn, rr);
            })};
        segments.erase(it, segments.end());

        // Offset the segments to start in the computational domain
        for (auto& segment : segments)
            segment.offset = segment.offset + local_rr;

        return segments;
    }
    case SegmentGroup::compute_inner: {
        return std::vector<ac::segment>{ac::segment{local_nn - 2 * local_rr, 2 * local_rr}};
    }
    case SegmentGroup::compute_full: {
        return std::vector<ac::segment>{ac::segment{local_nn, local_rr}};
    }
    case SegmentGroup::full: {
        return std::vector<ac::segment>{ac::segment{local_mm}};
    }
    default:
        ERRCHK(false);
        return std::vector<ac::segment>{};
    }
}

static std::vector<ac::segment>
partition(const AcMeshInfo& info, const SegmentGroup& group)
{
    return partition(acr::get_local_mm(info), acr::get_local_nn(info), acr::get_local_rr(), group);
}

static std::vector<ac::segment>
partition(const Device& device, const SegmentGroup& group)
{
    AcMeshInfo info{};
    ERRCHK_AC(acDeviceGetLocalConfig(device, &info));
    return partition(info, group);
}

static void
test_experimental_partition()
{
    const ac::shape nn{128, 128};
    const ac::index rr{3, 3};
    const ac::shape mm{nn + 2 * rr};
    auto segments{partition(mm, nn, rr, SegmentGroup::halo)};
    // auto segments{get_segments(mm, nn, rr, SegmentGroup::compute_outer)};
    for (const auto& segment : segments)
        std::cout << segment << std::endl;
}

/**
 * Launches a vector of compute kernels, each applied on a group of Segments specified by the group
 * parameter.
 */
static void
compute(const Device& device, const std::vector<Kernel>& compute_kernels,
        const std::vector<ac::segment>& segments)
{
    ERRCHK_AC(acDeviceSynchronizeStream(device, STREAM_ALL));
    for (const auto& segment : segments) {
        for (const auto& kernel : compute_kernels) {
            ERRCHK_AC(acDeviceLaunchKernel(device,
                                           STREAM_DEFAULT,
                                           kernel,
                                           convert_to_int3(segment.offset),
                                           convert_to_int3(segment.offset + segment.dims)));
        }
    }
    ERRCHK_AC(acDeviceSynchronizeStream(device, STREAM_ALL));
}

static void
compute(const Device& device, const std::vector<Kernel>& compute_kernels, const SegmentGroup& group)
{
    const auto segments{partition(device, group)};
    compute(device, compute_kernels, segments);
}

static void
reset(const Device& device, const std::vector<Field>& fields, const BufferGroup& group)
{
    VertexBufferArray vba{};
    ERRCHK_AC(acDeviceGetVBA(device, &vba));
    auto ptrs{ac::get_ptrs(device, fields, group)};

    for (auto& ptr : ptrs)
        ERRCHK_CUDA_API(cudaMemset(ptr.data(), 0, sizeof(ptr[0]) * ptr.size()));
}

static void
randomize(const MPI_Comm& comm, const Device& device, const std::vector<Field>& fields,
          const BufferGroup& group, const unsigned long seed = 6789)
{
    VertexBufferArray vba{};
    ERRCHK_AC(acDeviceGetVBA(device, &vba));
    auto ptrs{ac::get_ptrs(device, fields, group)};

    std::default_random_engine gen{as<unsigned long>(ac::mpi::get_rank(comm)) * 12345 + seed};
    std::uniform_real_distribution<AcReal> dist{0, 1};
    for (auto& ptr : ptrs) {
        ac::host_buffer<AcReal> tmp{ptr.size()};
        std::generate(tmp.begin(), tmp.end(), [&dist, &gen]() { return dist(gen); });
        ac::mr::copy(tmp.get(), ptr);
    }
}

static auto
make_halo_exchange_task(const AcMeshInfo& info, const size_t max_nbuffers)
{
    return HaloExchangeTask{acr::get_local_mm(info),
                            acr::get_local_nn(info),
                            acr::get_local_rr(),
                            max_nbuffers};
}

static auto
make_io_task(const AcMeshInfo& info, const size_t max_nbuffers)
{
    return IOTask{acr::get_global_nn(info),
                  acr::get_global_nn_offset(info),
                  acr::get_local_mm(info),
                  acr::get_local_nn(info),
                  acr::get_local_nn_offset(),
                  max_nbuffers};
}

class Grid {
  private:
    MPI_Comm cart_comm{MPI_COMM_NULL};
    AcMeshInfo local_info{};
    Device device{nullptr};
    AcReal current_time{0};

    HaloExchangeTask hydro_he;
    HaloExchangeTask tfm_he;
    IOTask hydro_io;
    IOTask uxb_io;

  public:
    explicit Grid(const AcMeshInfo& raw_info)
    {
        // Setup communicator and local mesh info
        auto global_nn{acr::get_global_nn(raw_info)};
        cart_comm  = ac::mpi::cart_comm_create(MPI_COMM_WORLD, global_nn);
        local_info = get_local_mesh_info(cart_comm, raw_info);
        ERRCHK(acPrintMeshInfoTFM(local_info) == 0);

        // Select and setup device
        int original_rank{MPI_PROC_NULL};
        ERRCHK_MPI_API(MPI_Comm_rank(MPI_COMM_WORLD, &original_rank));

        int nprocs{0};
        ERRCHK_MPI_API(MPI_Comm_size(MPI_COMM_WORLD, &nprocs));

        int device_count{0};
        ERRCHK_CUDA_API(cudaGetDeviceCount(&device_count));

        const int device_id{original_rank % device_count};
        ERRCHK_CUDA_API(cudaSetDevice(device_id));
        ERRCHK_CUDA_API(cudaDeviceSynchronize());

        ERRCHK_AC(acDeviceCreate(device_id, local_info, &device));

        // Setup halo exchange buffers
        hydro_he = make_halo_exchange_task(local_info, hydro_fields.size());
        tfm_he   = make_halo_exchange_task(local_info, tfm_fields.size());

        // Setup write tasks
        hydro_io = make_io_task(local_info, hydro_fields.size());
        uxb_io   = make_io_task(local_info, uxb_fields.size());

        // Dryrun
        reset_init_cond();
        tfm_pipeline(3);
        reset_init_cond();
    }

    ~Grid() noexcept
    {
        ERRCHK_MPI(acDeviceDestroy(device) == AC_SUCCESS);
        ac::mpi::cart_comm_destroy(&cart_comm);
    }

    void reset_init_cond()
    {
        PRINT_LOG_DEBUG("Enter");
        // Stencil coefficients
        AcReal stencils[NUM_STENCILS][STENCIL_DEPTH][STENCIL_HEIGHT][STENCIL_WIDTH]{};
        ERRCHK(get_stencil_coeffs(local_info, stencils) == 0);
        ERRCHK_AC(acDeviceLoadStencils(device, STREAM_DEFAULT, stencils));
        ERRCHK_AC(acDevicePrintInfo(device));

        // Forcing parameters
        ERRCHK(acHostUpdateForcingParams(&local_info) == 0);
        ERRCHK_AC(acDeviceLoadMeshInfo(device, local_info));

        // Fields
        ERRCHK_AC(acDeviceResetMesh(device, STREAM_DEFAULT));
        ERRCHK_AC(acDeviceSwapBuffers(device));
        ERRCHK_AC(acDeviceResetMesh(device, STREAM_DEFAULT));

        // Profiles
        ERRCHK(init_tfm_profiles(device) == 0);
// Note: all fields and profiles are initialized to 0 except
// the test profiles (PROFILE_B11 to PROFILE_B22)

// Debug:
#if defined(TFM_DEBUG_AVG_KERNEL)
        // randomize(cart_comm, device, all_fields, BufferGroup::input);
        AcMeshInfo info{};
        ERRCHK_AC(acDeviceGetLocalConfig(device, &info));
        const auto global_nn{acr::get_global_nn(info)};
        const auto rr{acr::get_local_rr()};
        const auto local_mm{ac::mpi::get_local_mm(cart_comm, global_nn, rr)};
        const auto local_nn{ac::mpi::get_local_nn(cart_comm, global_nn)};

        ac::host_ndbuffer<double> tmp{global_nn};
        ac::host_ndbuffer<double> ltmp{local_mm};
        std::iota(tmp.begin(), tmp.end(), 1);

        VertexBufferArray vba{};
        ERRCHK_AC(acDeviceGetVBA(device, &vba));
        for (size_t i{0}; i < NUM_FIELDS; ++i) {
            std::iota(tmp.begin(), tmp.end(), 1 + i * prod(global_nn));
            ac::mpi::scatter_advanced(cart_comm,
                                      ac::mpi::get_dtype<double>(),
                                      global_nn,
                                      ac::make_index(global_nn.size(), 0),
                                      tmp.data(),
                                      local_mm,
                                      local_nn,
                                      rr,
                                      vba.in[static_cast<Field>(i)]);
        }
#endif
    }

    void reduce_xy_averages(const Stream stream)
    {
        PRINT_LOG_DEBUG("Enter");
        ERRCHK_MPI(cart_comm != MPI_COMM_NULL);

        // Strategy:
        // 1) Reduce the local result to device->vba.profiles.in
        ERRCHK_AC(acDeviceReduceXYAverages(device, stream));
        ERRCHK_AC(acDeviceSynchronizeStream(device, STREAM_ALL));

        VertexBufferArray vba{};
        ERRCHK_AC(acDeviceGetVBA(device, &vba));

        const size_t axis{2};
        // NOTE: possible bug in the MPI implementation on LUMI
        // Reducing only a subset of profiles by using `nonlocal_tfm_profiles.size()`
        // as the profile count causes garbage values to appear at the end (constant B profiles)
        // This issue does not appear on Triton or Puhti.
        const size_t count{NUM_PROFILES * vba.profiles.count};
        AcReal* data{vba.profiles.in[0]};
        const auto collaborated_procs{ac::mpi::reduce_axis(cart_comm,
                                                           ac::mpi::get_dtype<AcReal>(),
                                                           MPI_SUM,
                                                           axis,
                                                           count,
                                                           data)};
        acMultiplyInplace(1 / static_cast<AcReal>(collaborated_procs), count, data);

        PRINT_LOG_TRACE("Exit");
    }

#if defined(AC_ENABLE_ASYNC_AVERAGES)
    MPI_Request launch_reduce_xy_averages(const Stream stream)
    {
        // Strategy:
        // 1) Reduce the local result to device->vba.profiles.in
        ERRCHK_AC(acDeviceReduceXYAverages(device, stream));

        // 2) Create communicator that encompasses neighbors in the xy direction
        const ac::index coords{ac::mpi::get_coords(cart_comm)};
        // Key used to order the ranks in the new communicator: let MPI_Comm_split
        // decide (should the same ordering as in the parent communicator by default)
        const int color{as<int>(coords[2])};
        const int key{ac::mpi::get_rank(cart_comm)};
        MPI_Comm xy_neighbors{MPI_COMM_NULL};
        ERRCHK_MPI_API(MPI_Comm_split(cart_comm, color, key, &xy_neighbors));

        // 3) Allreduce
        VertexBufferArray vba{};
        ERRCHK_AC(acDeviceGetVBA(device, &vba));

        // Note: assumes that all profiles are contiguous and in correct order
        // Error-prone: should be improved when time
        MPI_Request req{MPI_REQUEST_NULL};
        ERRCHK_MPI_API(
            MPI_Iallreduce(MPI_IN_PLACE,
                           vba.profiles.in[0],
                           nonlocal_tfm_profiles.size() *
                               vba.profiles.count, // Note possible MPI BUG here (see above)
                           AC_REAL_MPI_TYPE,
                           MPI_SUM,
                           xy_neighbors,
                           &req));

        // 5) Free resources
        ERRCHK_MPI_API(MPI_Comm_free(&xy_neighbors));

        return req;
    }
#endif

    void tfm_pipeline(const size_t nsteps)
    {
        PRINT_LOG_INFO("Launching TFM pipeline");

        // Clear the time series
        FILE* fp{fopen("timeseries.csv", "w")};
        ERRCHK_MPI(fp != NULL);
        std::fprintf(fp, "label,step,t_step,dt,min,rms,max,avg\n");
        ERRCHK_MPI(fclose(fp) == 0);

#if !defined(AC_DISABLE_IO)
        // Write time series
        write_timeseries(cart_comm, device, 0, 0, 0);

        // Write profiles
        write_profiles_to_disk(cart_comm, device, 0);

        // Write slices
        write_slices_to_disk(cart_comm, device, 0);

// Write snapshots
#if defined(AC_WRITE_SYNCHRONOUS_SNAPSHOTS)
        write_snapshots_to_disk(cart_comm, device, 0);
#else
        hydro_io.launch(cart_comm,
                        ac::get_ptrs(device, hydro_fields, BufferGroup::input),
                        get_field_paths(hydro_fields, 0));
        uxb_io.launch(cart_comm,
                      ac::get_ptrs(device, uxb_fields, BufferGroup::input),
                      get_field_paths(uxb_fields, 0));
#endif
#endif

        // Ensure halos are up-to-date before starting integration
        hydro_he.launch(cart_comm, ac::get_ptrs(device, hydro_fields, BufferGroup::input));
        tfm_he.launch(cart_comm, ac::get_ptrs(device, tfm_fields, BufferGroup::input));

#if defined(AC_ENABLE_ASYNC_AVERAGES)
        MPI_Request xy_average_req{launch_reduce_xy_averages(STREAM_DEFAULT)}; // Averaging
#else
        reduce_xy_averages(STREAM_DEFAULT);
#endif

        for (uint64_t step{1}; step < nsteps; ++step) {
            PRINT_LOG_INFO("New integration step");

            // Check whether to reset the test fields
            if ((as<int>(step) % acr::get(local_info, AC_simulation_reset_test_field_interval)) ==
                0) {
                PRINT_LOG_INFO("Resetting test fields");

                // Test fields: Discard previous halo exchange, reset, and update halos
                tfm_he.wait(ac::get_ptrs(device, tfm_fields, BufferGroup::input));
                reset(device, tfm_fields, BufferGroup::input);
                reset(device, tfm_fields, BufferGroup::output);
                tfm_he.launch(cart_comm, ac::get_ptrs(device, tfm_fields, BufferGroup::output));
            }

            // Current time
            acr::set(AC_current_time, current_time, local_info);

            // Timestep dependencies: local hydro (reduction skips ghost zones)
            const AcReal dt = calc_and_distribute_timestep(cart_comm, device);
            acr::set(AC_dt, dt, local_info);

            // Forcing
            ERRCHK(acHostUpdateForcingParams(&local_info) == 0);

            // Load the updated mesh info
            ERRCHK_AC(acLoadMeshInfo(local_info, STREAM_DEFAULT));

            for (int substep{0}; substep < 3; ++substep) {
                PRINT_LOG_DEBUG("Integration substep");

                // Hydro dependencies: hydro
                hydro_he.wait(ac::get_ptrs(device, hydro_fields, BufferGroup::input));

                // Outer segments
                // Note: outer integration kernels fused here and AC_exclude_inner set:
                // Operates only on the outer domain even though SegmentGroup:Full passed
                // TODO note: end index is not properly used, exits early
                ERRCHK_AC(acDeviceLoadIntUniform(device, STREAM_DEFAULT, AC_exclude_inner, 1));
                compute(device, hydro_kernels[as<size_t>(substep)], SegmentGroup::compute_full);
                // compute(device, hydro_kernels[as<size_t>(substep)], SegmentGroup::compute_full);

                // ERRCHK_AC(acDeviceLoadIntUniform(device, STREAM_DEFAULT, AC_exclude_inner, 0));
                // compute(device, hydro_kernels[as<size_t>(substep)], SegmentGroup::compute_outer);
                hydro_he.launch(cart_comm, ac::get_ptrs(device, hydro_fields, BufferGroup::output));

// TFM dependencies: hydro, tfm, profiles
#if defined(AC_ENABLE_ASYNC_AVERAGES)
                ERRCHK_MPI(xy_average_req != MPI_REQUEST_NULL);
                ac::mpi::request_wait_and_destroy(&xy_average_req); // Averaging
#endif
                // TFM dependencies: tfm
                tfm_he.wait(ac::get_ptrs(device, tfm_fields, BufferGroup::input));

                // Note: outer integration kernels fused here and AC_exclude_inner set:
                // Operates only on the outer domain even though SegmentGroup:Full passed
                compute(device, tfm_kernels[as<size_t>(substep)], SegmentGroup::compute_full);
                // compute(device, tfm_kernels[as<size_t>(substep)], SegmentGroup::compute_outer);
                tfm_he.launch(cart_comm, ac::get_ptrs(device, tfm_fields, BufferGroup::output));

                // Inner segments
                ERRCHK_AC(acDeviceLoadIntUniform(device, STREAM_DEFAULT, AC_exclude_inner, 0));
                compute(device, hydro_kernels[as<size_t>(substep)], SegmentGroup::compute_inner);
                compute(device, tfm_kernels[as<size_t>(substep)], SegmentGroup::compute_inner);
                ERRCHK_AC(acDeviceSwapBuffers(device));

// Profile dependencies: local tfm (uxb)
#if defined(AC_ENABLE_ASYNC_AVERAGES)
                ERRCHK_MPI(xy_average_req == MPI_REQUEST_NULL);
                xy_average_req = launch_reduce_xy_averages(STREAM_DEFAULT); // Averaging
#else
                reduce_xy_averages(STREAM_DEFAULT);
#endif
            }
            current_time += dt;

#if !defined(AC_DISABLE_IO)
// Write snapshot
#if defined(AC_WRITE_SYNCHRONOUS_SNAPSHOTS)
            if ((step %
                 as<uint64_t>(acr::get(local_info, AC_simulation_snapshot_output_interval))) == 0)
                write_snapshots_to_disk(cart_comm, device, step);
#else
            if ((step %
                 as<uint64_t>(acr::get(local_info, AC_simulation_snapshot_output_interval))) == 0) {
                hydro_io.wait();
                hydro_io.launch(cart_comm,
                                ac::get_ptrs(device, hydro_fields, BufferGroup::input),
                                get_field_paths(hydro_fields, step));
                uxb_io.wait();
                uxb_io.launch(cart_comm,
                              ac::get_ptrs(device, uxb_fields, BufferGroup::input),
                              get_field_paths(uxb_fields, step));
            }

#endif
#if defined(AC_WRITE_SYNCHRONOUS_PROFILES)
            if ((step %
                 as<uint64_t>(acr::get(local_info, AC_simulation_profile_output_interval))) == 0)
                write_profiles_to_disk(cart_comm, device, step);
#endif

#if defined(AC_WRITE_SYNCHRONOUS_TIMESERIES)
            if ((step %
                 as<uint64_t>(acr::get(local_info, AC_simulation_profile_output_interval))) == 0)
                write_timeseries(cart_comm, device, step, current_time, dt);
#endif
#if defined(AC_WRITE_SYNCHRONOUS_SLICES)
            if ((step %
                 as<uint64_t>(acr::get(local_info, AC_simulation_profile_output_interval))) == 0)
                write_slices_to_disk(cart_comm, device, step);
#endif
#endif
        }
        hydro_he.wait(ac::get_ptrs(device, hydro_fields, BufferGroup::input));
        tfm_he.wait(ac::get_ptrs(device, tfm_fields, BufferGroup::input));

#if !defined(AC_DISABLE_IO)
#if !defined(AC_WRITE_SYNCHRONOUS_SNAPSHOTS)
        hydro_io.wait();
        uxb_io.wait();
#endif
#endif

#if defined(AC_ENABLE_ASYNC_AVERAGES)
        ERRCHK(xy_average_req != MPI_REQUEST_NULL);
        ac::mpi::request_wait_and_destroy(&xy_average_req);
#endif
    }

    Grid(const Grid&)            = delete; // Copy constructor
    Grid& operator=(const Grid&) = delete; // Copy assignment operator
    Grid(Grid&&)                 = delete; // Move constructor
    Grid& operator=(Grid&&)      = delete; // Move assignment operator
};

int
main(int argc, char* argv[])
{
    ac::mpi::init_funneled();
    cudaProfilerStop();
    try {

        // Parse arguments
        Arguments args{};
        ERRCHK(acParseArguments(argc, argv, &args) == 0);
        ERRCHK(acPrintArguments(args) == 0);

        // Load configuration
        AcMeshInfo raw_info{};
        if (args.config_path) {
            ERRCHK(acParseINI(args.config_path, &raw_info) == 0);
        }
        else {
            const std::string default_config{AC_DEFAULT_TFM_CONFIG};
            PRINT_LOG_WARNING("No config path supplied, using %s", default_config.c_str());
            ERRCHK(acParseINI(default_config.c_str(), &raw_info) == 0);
        }

        // Override global_nn if instructed by the user
        if (args.global_nn_override[0] > 0) {
            ERRCHK_MPI(std::all_of(args.global_nn_override,
                                   args.global_nn_override + 3,
                                   [](const auto& val) { return val > 0; }));
            acr::set(AC_global_nx, args.global_nn_override[0], raw_info);
            acr::set(AC_global_ny, args.global_nn_override[1], raw_info);
            acr::set(AC_global_nz, args.global_nn_override[2], raw_info);
        }

        // Disable MPI_Abort on error and do manual error handling instead
        ERRCHK_MPI_API(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN));

        Grid grid{raw_info};

        cudaProfilerStart();
        grid.tfm_pipeline(as<uint64_t>(acr::get(raw_info, AC_simulation_nsteps)));
        cudaProfilerStop();
    }
    catch (std::exception& e) {
        std::cerr << e.what() << std::endl;
        ac::mpi::abort();
    }
    ac::mpi::finalize();
    std::cout << "Complete" << std::endl;
    return EXIT_SUCCESS;
}
