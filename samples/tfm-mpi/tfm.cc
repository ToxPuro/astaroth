#include <cmath>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <numeric>

#include "astaroth.h"
#include "astaroth_forcing.h"
#include "device_detail.h"

#include "acm/detail/errchk_mpi.h"
#include "acm/detail/mpi_utils.h"
#include "acm/detail/type_conversion.h"

#include "acm/detail/memory_resource.h"

#include "acm/detail/halo_exchange_packed.h"
#include "acm/detail/io.h"

#include "acm/detail/ndbuffer.h"

#include "stencil_loader.h"
#include "tfm_utils.h"

#include "acr_utils.h"
#include "tfm_utils_cc.h"

#include <mpi.h>

#define AC_ENABLE_ASYNC_AVERAGES (1)

#define BENCHMARK(cmd)                                                                             \
    do {                                                                                           \
        const auto start__{std::chrono::system_clock::now()};                                      \
        (cmd);                                                                                     \
        const auto ms_elapsed__ = std::chrono::duration_cast<std::chrono::milliseconds>(           \
            std::chrono::system_clock::now() - start__);                                           \
        std::cout << "[" << ms_elapsed__.count() << " ms] " << #cmd << std::endl;                  \
    } while (0)

/** Apply a static cast to all elements of the input vector from type U to T */
template <typename T, typename U>
ac::vector<T>
static_cast_vec(const ac::vector<U>& in)
{
    ac::vector<T> out(in.size());
    for (size_t i{0}; i < in.size(); ++i)
        out[i] = static_cast<T>(in[i]);
    return out;
}

/**
 * Decompose the domain set in the input info and returns a complete local mesh info with all
 * parameters set (incl. multi-device offsets, and others)
 */
static AcMeshInfo
get_local_mesh_info(const MPI_Comm& cart_comm, const AcMeshInfo& info)
{
    // Calculate local dimensions
    Shape global_nn{acr::get_global_nn(info)};
    Dims global_ss{acr::get_global_ss(info)};

    const Shape decomp{ac::mpi::get_decomposition(cart_comm)};
    const Shape local_nn{global_nn / decomp};
    const Dims local_ss{global_ss / static_cast_vec<AcReal>(decomp)};

    const Index coords{ac::mpi::get_coords(cart_comm)};
    const Index global_nn_offset{coords * local_nn};

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
    acr::set(AC_dummy_real3, (AcReal3){0, 0, 0}, local_info);

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
    AcMeshInfo info{};
    ERRCHK_AC(acDeviceGetLocalConfig(device, &info));

    const AcReal global_sz{acr::get(info, AC_global_sz)};
    const size_t global_nz{as<size_t>(acr::get(info, AC_global_grid_n).z)};
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
    acHostInitProfileToCosineWave(global_sz,
                                  global_nz,
                                  offset,
                                  amplitude,
                                  wavenumber,
                                  local_mz,
                                  host_profile.get());
    ERRCHK_AC(acDeviceLoadProfile(device, host_profile.get(), local_mz, PROFILE_B11mean_x));
    ERRCHK_AC(acDeviceLoadProfile(device, host_profile.get(), local_mz, PROFILE_B21mean_y));

    // B1s (here B12) and B2s (here B22)
    acHostInitProfileToSineWave(global_sz,
                                global_nz,
                                offset,
                                amplitude,
                                wavenumber,
                                local_mz,
                                host_profile.get());
    ERRCHK_AC(acDeviceLoadProfile(device, host_profile.get(), local_mz, PROFILE_B12mean_x));
    ERRCHK_AC(acDeviceLoadProfile(device, host_profile.get(), local_mz, PROFILE_B22mean_y));

    return 0;
}

/** Write both the mesh snapshot and profiles synchronously to disk */
static int
write_diagnostic_step(const MPI_Comm& parent_comm, const Device& device, const size_t step)
{
    VertexBufferArray vba{};
    ERRCHK_AC(acDeviceGetVBA(device, &vba));

    AcMeshInfo local_info{};
    ERRCHK_AC(acDeviceGetLocalConfig(device, &local_info));

    for (int i = 0; i < NUM_VTXBUF_HANDLES; ++i) {
        char filepath[4096];
        sprintf(filepath, "debug-step-%012zu-tfm-%s.mesh", step, vtxbuf_names[i]);
        printf("Writing %s\n", filepath);
        ac::mpi::write_collective_simple(parent_comm,
                                         ac::mpi::get_dtype<AcReal>(),
                                         acr::get_global_nn(local_info),
                                         acr::get_local_nn_offset(),
                                         vba.in[i],
                                         std::string(filepath));
    }
    for (int i = 0; i < NUM_PROFILES; ++i) {
        char filepath[4096];
        sprintf(filepath, "debug-step-%012zu-tfm-%s.profile", step, profile_names[i]);
        printf("Writing %s\n", filepath);
        const Shape profile_global_nz{as<uint64_t>(acr::get(local_info, AC_global_nz))};
        const Shape profile_local_mz{as<uint64_t>(acr::get(local_info, AC_mz))};
        const Shape profile_local_nz{as<uint64_t>(acr::get(local_info, AC_nz))};
        const Shape profile_local_nz_offset{as<uint64_t>(acr::get(local_info, AC_nz_min))};
        const Index coords{ac::mpi::get_coords(parent_comm)[2]};
        const Shape profile_global_nz_offset{coords * profile_local_nz};

        const int rank{ac::mpi::get_rank(parent_comm)};
        const Index coords_3d{ac::mpi::get_coords(parent_comm)};
        const Shape decomp_3d{ac::mpi::get_decomposition(parent_comm)};
        const int color = (coords_3d[0] + coords_3d[1] * decomp_3d[0]) == 0 ? 0 : MPI_UNDEFINED;

        MPI_Comm profile_comm{MPI_COMM_NULL};
        ERRCHK_MPI_API(MPI_Comm_split(parent_comm, color, rank, &profile_comm));

        if (profile_comm != MPI_COMM_NULL) {
            ac::mpi::write_collective(profile_comm,
                                      ac::mpi::get_dtype<AcReal>(),
                                      profile_global_nz,
                                      profile_global_nz_offset,
                                      profile_local_mz,
                                      profile_local_nz,
                                      profile_local_nz_offset,
                                      vba.profiles.in[i],
                                      std::string(filepath));
            ERRCHK_MPI_API(MPI_Comm_free(&profile_comm));
        }
    }
    return 0;
}

/** Calculate the timestep length and distribute it to all devices in the grid */
static AcReal
calc_and_distribute_timestep(const MPI_Comm& parent_comm, const Device& device)
{
    VertexBufferArray vba{};
    ERRCHK_AC(acDeviceGetVBA(device, &vba));

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

/** Return outer or inner segments based on the return_outer_segments parameter */
static std::vector<ac::segment>
get_compute_segments(const Device& device, const SegmentGroup group)
{
    AcMeshInfo info{};
    ERRCHK_AC(acDeviceGetLocalConfig(device, &info));

    // Inner domain dimensions
    const Shape mm{acr::get_local_nn(info)};
    const Shape nn{acr::get_local_nn(info) - 2 * acr::get_local_rr()};
    const Shape rr{acr::get_local_rr()};

    // Partition the mesh
    auto segments{partition(mm, nn, rr)};

    switch (group) {
    case SegmentGroup::Outer: {
        // Prune the segments overlapping with the computational domain
        auto it{
            std::remove_if(segments.begin(), segments.end(), [nn, rr](const ac::segment& segment) {
                return within_box(segment.offset, nn, rr);
            })};
        segments.erase(it, segments.end());
        break;
    }
    case SegmentGroup::Inner: {
        // Prune the segments not within computational domain
        auto it{
            std::remove_if(segments.begin(), segments.end(), [nn, rr](const ac::segment& segment) {
                return !within_box(segment.offset, nn, rr);
            })};
        segments.erase(it, segments.end());
        break;
    }
    case SegmentGroup::Full: {
        return std::vector<ac::segment>{ac::segment{nn, rr}};
        break;
    }
    default:
        ERRCHK(false);
    }

    // Offset the segments to start in the computational domain
    for (auto& segment : segments)
        segment.offset = segment.offset + acr::get_local_rr();

    return segments;
}

/**
 * Launches a vector of compute kernels, each applied on a group of Segments specified by the group
 * parameter.
 * TODO: deprecated
 */
static void
compute(const Device& device, const std::vector<Kernel>& compute_kernels, const SegmentGroup group)
{
    auto segments{get_compute_segments(device, group)};

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

/** Concatenates the field name and ".mesh" of a vector of handles */
static auto
get_field_paths(const std::vector<VertexBufferHandle>& handles)
{
    std::vector<std::string> paths;

    for (const auto& handle : handles)
        paths.push_back(std::string(field_names[handle]) + ".mesh");

    return paths;
}

/** Extracts a vector of pointers to the data associated with a field group of some type */
static auto
get_fields(const Device& device, const FieldGroup& group, const BufferGroup& type)
{
    AcMeshInfo info{};
    ERRCHK_AC(acDeviceGetLocalConfig(device, &info));
    const auto local_mm{acr::get_local_mm(info)};
    const auto count{prod(local_mm)};

    VertexBufferArray vba{};
    ERRCHK_AC(acDeviceGetVBA(device, &vba));

    std::vector<VertexBufferHandle> fields{get_field_handles(group)};

    AcReal** field_ptrs{nullptr};
    switch (type) {
    case BufferGroup::Input:
        field_ptrs = vba.in;
        break;
    case BufferGroup::Output:
        field_ptrs = vba.out;
        break;
    default:
        ERRCHK(false);
    }

    std::vector<ac::mr::device_ptr<AcReal>> output;

    for (const auto& field : fields)
        output.push_back(ac::mr::device_ptr<AcReal>{count, field_ptrs[field]});

    return output;
}

template <typename T, typename MemoryResource>
static auto
create_tfm_halo_exchange_task(const Device& device, const FieldGroup group)
{
    AcMeshInfo info{};
    ERRCHK_AC(acDeviceGetLocalConfig(device, &info));

    return ac::comm::AsyncHaloExchangeTask<T, MemoryResource>{acr::get_local_mm(info),
                                                              acr::get_local_nn(info),
                                                              acr::get_local_rr(),
                                                              get_fields(device,
                                                                         group,
                                                                         BufferGroup::Input)
                                                                  .size()};
}

template <typename T, typename MemoryResource>
static auto
create_tfm_io_batch_task(const Device& device, const FieldGroup group)
{
    AcMeshInfo info{};
    ERRCHK_AC(acDeviceGetLocalConfig(device, &info));

    return ac::io::BatchedAsyncWriteTask<T, MemoryResource>{acr::get_global_nn(info),
                                                            acr::get_global_nn_offset(info),
                                                            acr::get_local_mm(info),
                                                            acr::get_local_nn(info),
                                                            acr::get_local_nn_offset(),
                                                            get_fields(device,
                                                                       group,
                                                                       BufferGroup::Input)
                                                                .size()};
}

namespace ac {

template <typename T, typename HaloExchangeMemoryResource = ac::mr::device_memory_resource,
          typename IOMemoryResource = ac::mr::pinned_host_memory_resource>
class Grid {
  private:
    MPI_Comm cart_comm{MPI_COMM_NULL};
    AcMeshInfo local_info{};
    Device device{nullptr};
    AcReal current_time{0};

    std::vector<VertexBufferHandle> write_fields{VTXBUF_LNRHO, VTXBUF_UUX, VTXBUF_UUY, VTXBUF_UUZ};

    ac::comm::AsyncHaloExchangeTask<AcReal, HaloExchangeMemoryResource> hydro_he;
    ac::comm::AsyncHaloExchangeTask<AcReal, HaloExchangeMemoryResource> tfm_he;
    ac::io::BatchedAsyncWriteTask<AcReal, IOMemoryResource> hydro_io;
    ac::io::BatchedAsyncWriteTask<AcReal, IOMemoryResource> bfield_io;

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
        hydro_he = create_tfm_halo_exchange_task<T, HaloExchangeMemoryResource>(device,
                                                                                FieldGroup::Hydro);
        tfm_he   = create_tfm_halo_exchange_task<T, HaloExchangeMemoryResource>(device,
                                                                              FieldGroup::TFM);

        // Setup write tasks
        hydro_io  = create_tfm_io_batch_task<T, IOMemoryResource>(device, FieldGroup::Hydro);
        bfield_io = create_tfm_io_batch_task<T, IOMemoryResource>(device, FieldGroup::Bfield);

        // Dryrun
        reset_init_cond();
        tfm_pipeline(5);

        reset_init_cond();
        test();

        reset_init_cond();
    }

    ~Grid()
    {
        ERRCHK_AC(acDeviceDestroy(device));
        ac::mpi::cart_comm_destroy(cart_comm);
    }

    void test()
    {
        /////////////////
        // // Domain
        // const auto mm{acr::get_local_mm(local_info)};
        // const auto nn{acr::get_local_nn(local_info)};
        // const auto rr{acr::get_local_rr()};
        // const auto global_nn_offset{acr::get_global_nn_offset(local_info)};

        // // Buffer
        // ac::ndbuffer<AcReal, ac::mr::host_memory_resource> hbuf{mm, NAN};
        // for (size_t k{rr[2]}; k < rr[2] + nn[2]; ++k) {
        //     const Shape slice{nn[0], nn[1], 1};
        //     const Index offset{rr[0], rr[1], k};

        //     ac::fill(static_cast<AcReal>(k + acr::get_global_nn_offset(local_info)[2]),
        //              slice,
        //              offset,
        //              hbuf);
        // }
        // hbuf.display();
        // exit(0);
        /////////////////

        // ac::buffer<AcReal, ac::mr::host_memory_resource> buf{prod(mm)};
        // std::iota(buf.begin(), buf.end(), prod(global_nn_offset) - prod(rr));
        // buf.display();
        // std::cout << "Global nn offset " << global_nn_offset << std::endl;

        // std::vector<Field> fields{VTXBUF_UUX, VTXBUF_UUY};

        // // Partition
        // auto segments{partition(mm, nn, rr)};
        // auto it{std::remove_if(segments.begin(), segments.end(), [nn, rr](const auto& segment) {
        //     return within_box(segment.offset, nn, rr);
        // })};
        // segments.erase(it, segments.end());

        // // Setup buffers
        // ac::ndbuffer<AcReal, ac::mr::host_memory_resource> hux{mm, NAN}, huy{mm};
        // for (size_t k{rr[2]}; k < rr[2] + nn[2]; ++k) {
        //     const Shape slice{nn[0], nn[1], 1};
        //     const Index offset{rr[0], rr[1], k};

        //     ac::fill(static_cast<AcReal>(k + acr::get_global_nn_offset(local_info)[2]),
        //              slice,
        //              offset,
        //              hux);
        //     ac::fill(static_cast<AcReal>(k + acr::get_global_nn_offset(local_info)[2]),
        //              slice,
        //              offset,
        //              huy);
        // }

        // ac::ndbuffer<AcReal, ac::mr::device_memory_resource> dux{mm}, duy{mm};
        // ac::mr::copy(hux.get(), dux.get());
        // ac::mr::copy(huy.get(), duy.get());

        // ac::mr::copy(dux.get(), hux.get());
        // ac::mr::copy(duy.get(), huy.get());

        // hux.display();
    }

    void reset_init_cond()
    {
        // Stencil coefficients
        AcReal stencils[NUM_STENCILS][STENCIL_DEPTH][STENCIL_HEIGHT][STENCIL_WIDTH]{};
        ERRCHK(get_stencil_coeffs(local_info, stencils) == 0);
        ERRCHK_AC(acDeviceLoadStencils(device, STREAM_DEFAULT, stencils));
        ERRCHK_AC(acDevicePrintInfo(device));

        // Forcing parameter
        ERRCHK(acHostUpdateForcingParams(&local_info) == 0);
        ERRCHK_AC(acDeviceLoadMeshInfo(device, local_info));

        // Profiles
        ERRCHK(init_tfm_profiles(device) == 0);

        // Fields
        ERRCHK_AC(acDeviceResetMesh(device, STREAM_DEFAULT));
        ERRCHK_AC(acDeviceSwapBuffers(device));
        ERRCHK_AC(acDeviceResetMesh(device, STREAM_DEFAULT));

        // Note: all fields and profiles are initialized to 0 except
        // the test profiles (PROFILE_B11 to PROFILE_B22)
    }

    void reduce_xy_averages(const Stream stream)
    {
        ERRCHK_MPI(cart_comm != MPI_COMM_NULL);

        // Strategy:
        // 1) Reduce the local result to device->vba.profiles.in
        ERRCHK_AC(acDeviceReduceXYAverages(device, stream));

        // 2) Create communicator that encompasses neighbors in the xy direction
        const Index coords{ac::mpi::get_coords(cart_comm)};
        // Key used to order the ranks in the new communicator: let MPI_Comm_split
        // decide (should the same ordering as in the parent communicator by default)
        const int color{as<int>(coords[2])};
        const int key{ac::mpi::get_rank(cart_comm)};
        MPI_Comm xy_neighbors{MPI_COMM_NULL};
        ERRCHK_MPI_API(MPI_Comm_split(cart_comm, color, key, &xy_neighbors));
        ERRCHK_MPI(xy_neighbors != MPI_COMM_NULL);

        // 3) Allreduce
        VertexBufferArray vba{};
        ERRCHK_AC(acDeviceGetVBA(device, &vba));

        // Note: assumes that all profiles are contiguous and in correct order
        // Error-prone: should be improved when time
        ERRCHK_MPI_API(MPI_Allreduce(MPI_IN_PLACE,
                                     vba.profiles.in[0],
                                     get_profile_handles(ProfileGroup::TFM_Nonlocal).size() *
                                         vba.profiles.count,
                                     AC_REAL_MPI_TYPE,
                                     MPI_SUM,
                                     xy_neighbors));

        // 4) Free resources
        ERRCHK_MPI_API(MPI_Comm_free(&xy_neighbors));
    }

#if AC_ENABLE_ASYNC_AVERAGES
    MPI_Request launch_reduce_xy_averages(const Stream stream)
    {
        // Strategy:
        // 1) Reduce the local result to device->vba.profiles.in
        ERRCHK_AC(acDeviceReduceXYAverages(device, stream));

        // 2) Create communicator that encompasses neighbors in the xy direction
        const Index coords{ac::mpi::get_coords(cart_comm)};
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
        ERRCHK_MPI_API(MPI_Iallreduce(MPI_IN_PLACE,
                                      vba.profiles.in[0],
                                      get_profile_handles(ProfileGroup::TFM_Nonlocal).size() *
                                          vba.profiles.count,
                                      AC_REAL_MPI_TYPE,
                                      MPI_SUM,
                                      xy_neighbors,
                                      &req));

        // 5) Free resources
        ERRCHK_MPI_API(MPI_Comm_free(&xy_neighbors));

        return req;
    }
#endif

    void tfm_pipeline(const size_t niters)
    {
        // Ensure halos are up-to-date before starting integration
        hydro_he.launch(cart_comm, get_fields(device, FieldGroup::Hydro, BufferGroup::Input));
        tfm_he.launch(cart_comm, get_fields(device, FieldGroup::TFM, BufferGroup::Input));

#if AC_ENABLE_ASYNC_AVERAGES
        MPI_Request xy_average_req{launch_reduce_xy_averages(STREAM_DEFAULT)}; // Averaging
#else
        reduce_xy_averages(STREAM_DEFAULT);
#endif

        // Write the initial step
        hydro_io.launch(cart_comm,
                        get_fields(device, FieldGroup::Hydro, BufferGroup::Input),
                        get_field_paths(get_field_handles(FieldGroup::Hydro)));
        bfield_io.launch(cart_comm,
                         get_fields(device, FieldGroup::Bfield, BufferGroup::Input),
                         get_field_paths(get_field_handles(FieldGroup::Bfield)));

        for (size_t iter{0}; iter < niters; ++iter) {

            // Current time
            acr::set(AC_current_time, current_time, local_info);

            // Timestep dependencies: local hydro (reduction skips ghost zones)
            const AcReal dt = calc_and_distribute_timestep(cart_comm, device);
            acr::set(AC_dt, dt, local_info);

            // Forcing
            ERRCHK(acHostUpdateForcingParams(&local_info) == 0);

            // Load the updated mesh info
            ERRCHK_AC(acLoadMeshInfo(local_info, STREAM_DEFAULT));

            for (int step{0}; step < 3; ++step) {

                // Outer segments
                ERRCHK_AC(acDeviceLoadIntUniform(device, STREAM_DEFAULT, AC_exclude_inner, 1));

                // Hydro dependencies: hydro
                hydro_he.wait(get_fields(device, FieldGroup::Hydro, BufferGroup::Input));

                // Note: outer integration kernels fused here and AC_exclude_inner set:
                // Operates only on the outer domain even though SegmentGroup:Full passed
                compute(device, get_kernels(FieldGroup::Hydro, step), SegmentGroup::Full);
                hydro_he.launch(cart_comm,
                                get_fields(device, FieldGroup::Hydro, BufferGroup::Output));

// TFM dependencies: hydro, tfm, profiles
#if AC_ENABLE_ASYNC_AVERAGES
                ERRCHK_MPI(xy_average_req != MPI_REQUEST_NULL);
                ac::mpi::request_wait_and_destroy(xy_average_req); // Averaging
#endif
                // TFM dependencies: tfm
                tfm_he.wait(get_fields(device, FieldGroup::TFM, BufferGroup::Input));

                // Note: outer integration kernels fused here and AC_exclude_inner set:
                // Operates only on the outer domain even though SegmentGroup:Full passed
                compute(device, get_kernels(FieldGroup::TFM, step), SegmentGroup::Full);
                tfm_he.launch(cart_comm, get_fields(device, FieldGroup::TFM, BufferGroup::Output));

                // Inner segments
                ERRCHK_AC(acDeviceLoadIntUniform(device, STREAM_DEFAULT, AC_exclude_inner, 0));
                compute(device, get_kernels(FieldGroup::Hydro, step), SegmentGroup::Inner);
                compute(device, get_kernels(FieldGroup::TFM, step), SegmentGroup::Inner);
                ERRCHK_AC(acDeviceSwapBuffers(device));

// Profile dependencies: local tfm (uxb)
#if AC_ENABLE_ASYNC_AVERAGES
                ERRCHK_MPI(xy_average_req == MPI_REQUEST_NULL);
                xy_average_req = launch_reduce_xy_averages(STREAM_DEFAULT); // Averaging
#else
                reduce_xy_averages(STREAM_DEFAULT);
#endif
            }
            current_time += dt;

// Write snapshot
#if 0
            hydro_io.wait();
            hydro_io.launch(cart_comm,
                            get_fields(device, FieldGroup::Hydro, BufferGroup::Input),
                            get_field_paths(get_field_handles(FieldGroup::Hydro)));
            bfield_io.wait();
            bfield_io.launch(cart_comm,
                             get_fields(device, FieldGroup::Bfield, BufferGroup::Input),
                             get_field_paths(get_field_handles(FieldGroup::Bfield)));
#endif

            // Write mesh and profiles
            // TODO: this is synchronous. Consider async.
            write_diagnostic_step(cart_comm, device, iter);
        }
        hydro_he.wait(get_fields(device, FieldGroup::Hydro, BufferGroup::Input));
        tfm_he.wait(get_fields(device, FieldGroup::TFM, BufferGroup::Input));

        hydro_io.wait();
        bfield_io.wait();

#if AC_ENABLE_ASYNC_AVERAGES
        ERRCHK(xy_average_req != MPI_REQUEST_NULL);
        ac::mpi::request_wait_and_destroy(xy_average_req);
#endif
    }

    Grid(const Grid&)            = delete; // Copy constructor
    Grid& operator=(const Grid&) = delete; // Copy assignment operator
    Grid(Grid&&)                 = delete; // Move constructor
    Grid& operator=(Grid&&)      = delete; // Move assignment operator
};

} // namespace ac

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
            PRINT_LOG("No config path supplied, using %s", default_config.c_str());
            ERRCHK(acParseINI(default_config.c_str(), &raw_info) == 0);
        }

        // Disable MPI_Abort on error and do manual error handling instead
        ERRCHK_MPI_API(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN));

        auto grid{ac::Grid<AcReal>(raw_info)};
        grid.reset_init_cond();

        cudaProfilerStart();
        grid.tfm_pipeline(10);
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
