#include <cstdlib>
#include <exception>
#include <iostream>
#include <numeric>

#include "astaroth.h"
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

#include <mpi.h>

#define ERRCHK_AC(errcode)                                                                         \
    do {                                                                                           \
        const AcResult _tmp_ac_api_errcode_ = (errcode);                                           \
        if (_tmp_ac_api_errcode_ != AC_SUCCESS) {                                                  \
            errchk_print_error(__func__, __FILE__, __LINE__, #errcode, "Astaroth error");          \
            errchk_print_stacktrace();                                                             \
            MPI_Abort(MPI_COMM_WORLD, -1);                                                         \
        }                                                                                          \
    } while (0)

#define BENCHMARK(cmd)                                                                             \
    do {                                                                                           \
        const auto start__{std::chrono::system_clock::now()};                                      \
        (cmd);                                                                                     \
        const auto ms_elapsed__ = std::chrono::duration_cast<std::chrono::milliseconds>(           \
            std::chrono::system_clock::now() - start__);                                           \
        std::cout << "[" << ms_elapsed__.count() << " ms] " << #cmd << std::endl;                  \
    } while (0)

using Dims = ac::vector<AcReal>;

template <typename T, typename U>
ac::vector<T>
static_cast_vec(const ac::vector<U>& in)
{
    ac::vector<T> out(in.size());
    for (size_t i{0}; i < in.size(); ++i)
        out[i] = static_cast<T>(in[i]);
    return out;
}

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
get_local_nn_offset()
{
    return Index{(STENCIL_WIDTH - 1) / 2, (STENCIL_HEIGHT - 1) / 2, (STENCIL_DEPTH - 1) / 2};
}

Index
get_global_nn_offset(const AcMeshInfo& info)
{
    ERRCHK(acVerifyMeshInfo(info) == 0);
    return Index{as<uint64_t>(acr::get(info, AC_multigpu_offset).x),
                 as<uint64_t>(acr::get(info, AC_multigpu_offset).y),
                 as<uint64_t>(acr::get(info, AC_multigpu_offset).z)};
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

    local_info.int3_params[AC_multigpu_offset] = {
        as<int>(global_nn_offset[0]),
        as<int>(global_nn_offset[1]),
        as<int>(global_nn_offset[2]),
    };

    acr::set(AC_sx, static_cast<AcReal>(local_ss[0]), local_info);
    acr::set(AC_sy, static_cast<AcReal>(local_ss[1]), local_info);
    acr::set(AC_sz, static_cast<AcReal>(local_ss[2]), local_info);

    // Backwards compatibility
    acr::set(AC_global_grid_n,
             int3{as<int>(global_nn[0]), as<int>(global_nn[1]), as<int>(global_nn[2])},
             local_info);

    ERRCHK(acHostUpdateLocalBuiltinParams(&local_info) == 0);
    ERRCHK(acHostUpdateMHDSpecificParams(&local_info) == 0);
    ERRCHK(acHostUpdateTFMSpecificGlobalParams(&local_info) == 0);

    // Others to ensure nothing is left uninitialized
    acr::set(AC_init_type, 0, local_info);
    // acr::set(AC_step_number, 0, local_info);
    acr::set(AC_dt, 0, local_info);
    acr::set(AC_dummy_real3, (AcReal3){0, 0, 0}, local_info);

    ERRCHK(acVerifyMeshInfo(local_info) == 0);
    return local_info;
}

} // namespace acr

static int
init_tfm_profiles(const Device& device)
{
    AcMeshInfo info{};
    ERRCHK_AC(acDeviceGetLocalConfig(device, &info));

    const AcReal global_sz = acr::get(info, AC_global_sz);
    const size_t global_nz = as<size_t>(acr::get(info, AC_global_grid_n).z);
    const long offset      = -acr::get(info, AC_nz_min) + acr::get(info, AC_multigpu_offset).z;
    const size_t local_mz  = as<size_t>(acr::get(info, AC_mz));

    const AcReal amplitude  = acr::get(info, AC_profile_amplitude);
    const AcReal wavenumber = acr::get(info, AC_profile_wavenumber);

    auto host_profile{std::make_unique<AcReal[]>(local_mz)};

    // All to zero
    acHostInitProfileToValue(0, local_mz, host_profile.get());
    for (size_t profile{0}; profile < NUM_PROFILES; ++profile)
        ERRCHK_AC(acDeviceLoadProfile(device,
                                      host_profile.get(),
                                      local_mz,
                                      static_cast<Profile>(profile)));

    // acDeviceLoadProfile(device, host_profile.get(), local_mz, PROFILE_B11mean_x);
    // acDeviceLoadProfile(device, host_profile.get(), local_mz, PROFILE_B11mean_y);
    // acDeviceLoadProfile(device, host_profile.get(), local_mz, PROFILE_B11mean_z);
    // acDeviceLoadProfile(device, host_profile.get(), local_mz, PROFILE_B12mean_x);
    // acDeviceLoadProfile(device, host_profile.get(), local_mz, PROFILE_B12mean_y);
    // acDeviceLoadProfile(device, host_profile.get(), local_mz, PROFILE_B12mean_z);
    // acDeviceLoadProfile(device, host_profile.get(), local_mz, PROFILE_B21mean_x);
    // acDeviceLoadProfile(device, host_profile.get(), local_mz, PROFILE_B21mean_y);
    // acDeviceLoadProfile(device, host_profile.get(), local_mz, PROFILE_B21mean_z);
    // acDeviceLoadProfile(device, host_profile.get(), local_mz, PROFILE_B22mean_x);
    // acDeviceLoadProfile(device, host_profile.get(), local_mz, PROFILE_B22mean_y);
    // acDeviceLoadProfile(device, host_profile.get(), local_mz, PROFILE_B22mean_z);

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

namespace acc {
Shape
get_global_nn(const AcMeshInfo& info)
{
    return Shape{as<uint64_t>(acr::get(info, AC_global_nx)),
                 as<uint64_t>(acr::get(info, AC_global_ny)),
                 as<uint64_t>(acr::get(info, AC_global_nz))};
}
Index
get_local_nn_offset()
{
    return Index{(STENCIL_WIDTH - 1) / 2, (STENCIL_HEIGHT - 1) / 2, (STENCIL_DEPTH - 1) / 2};
}
} // namespace acc

static int
acDeviceWriteProfileToDisk(const Device device, const Profile profile, const char* filepath)
{
    AcMeshInfo info{};
    acDeviceGetLocalConfig(device, &info);
    const size_t mz = as<size_t>(acr::get(info, AC_global_grid_n).z +
                                 2 * ((STENCIL_DEPTH - 1) / 2));

    AcBuffer host_profile = acBufferCreate(mz, false);
    acDeviceStoreProfile(device, profile, host_profile.data, host_profile.count);
    acHostWriteProfileToFile(filepath, host_profile.data, host_profile.count);
    acBufferDestroy(&host_profile);
    return EXIT_SUCCESS;
}

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
                                         acc::get_global_nn(local_info),
                                         acc::get_local_nn_offset(),
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

namespace ac {

class Grid {
  private:
    MPI_Comm cart_comm{MPI_COMM_NULL};
    AcMeshInfo local_info{};
    Device device{nullptr};
    AcReal current_time{0};

  public:
    explicit Grid(const AcMeshInfo& raw_info)
    {
        // Setup communicator and local mesh info
        auto global_nn{acr::get_global_nn(raw_info)};
        cart_comm  = ac::mpi::cart_comm_create(MPI_COMM_WORLD, global_nn);
        local_info = acr::get_local_mesh_info(cart_comm, raw_info);
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

        // Dryrun
        reset_init_cond();
        tfm_pipeline(5);
        reset_init_cond();
    }

    ~Grid()
    {
        ERRCHK_AC(acDeviceDestroy(device));
        ac::mpi::cart_comm_destroy(cart_comm);
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
        init_tfm_profiles(device);

        // Fields
        ERRCHK_AC(acDeviceResetMesh(device, STREAM_DEFAULT));
        ERRCHK_AC(acDeviceSwapBuffers(device));
        ERRCHK_AC(acDeviceResetMesh(device, STREAM_DEFAULT));

        // Note: all fields and profiles are initialized to 0 except
        // the test profiles (PROFILE_B11 to PROFILE_B22)
    }

    void tfm_pipeline(const size_t niters)
    {
        for (size_t iter{0}; iter < niters; ++iter) {

            // Current time
            ERRCHK_AC(
                acDeviceLoadScalarUniform(device, STREAM_DEFAULT, AC_current_time, current_time));

            // Timestep dependencies: hydro
            // TODO hydro dependency
            const AcReal dt = calc_and_distribute_timestep(cart_comm, device);
            ERRCHK_AC(acDeviceLoadScalarUniform(device, STREAM_DEFAULT, AC_dt, dt));

            // for (int step = 0; step < 3; ++step) {
            //     ERRCHK_AC(acDeviceLoadIntUniform(device, STREAM_DEFAULT, AC_step_number, step));

            //     // Hydro dependencies: hydro
            //     hydro_he.wait(...);
            //     hydro_compute_outer(...); // Needs to be synchronous
            //     hydro_he.launch(...);

            //     // TFM dependencies: hydro, tfm, profiles
            //     tfm_he.wait(...);
            //     profiles_he.wait(...);
            //     tfm_compute_outer(...); // Needs to be synchronous
            //     tfm_he.launch(...);

            //     hydro_compute_inner(...);
            //     tfm_compute_inner(...);
            //     ERRCHK_AC(acDeviceSwapBuffers(device));

            //     // Profile dependencies: local tfm (uxb)
            //     profiles_compute(...);
            //     profiles_he.launch(...);
            // }

            current_time += dt;
        }
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

        auto grid{ac::Grid(raw_info)};
#if 0
        Shape global_nn{as<uint64_t>(acr::get(raw_info, AC_global_nx)),
                        as<uint64_t>(acr::get(raw_info, AC_global_ny)),
                        as<uint64_t>(acr::get(raw_info, AC_global_nz))};
        Dims global_ss{static_cast<AcReal>(acr::get(raw_info, AC_global_sx)),
                       static_cast<AcReal>(acr::get(raw_info, AC_global_sy)),
                       static_cast<AcReal>(acr::get(raw_info, AC_global_sz))};
        Index local_nn_offset{(STENCIL_WIDTH - 1) / 2,
                              (STENCIL_HEIGHT - 1) / 2,
                              (STENCIL_DEPTH - 1) / 2};

        // Create the Cartesian communicator
        MPI_Comm cart_comm{ac::mpi::cart_comm_create(MPI_COMM_WORLD, global_nn)};

        // Fill the local mesh configuration (overwrite the earlier info
        const AcMeshInfo local_info = acr::get_local_mesh_info(cart_comm, raw_info);
        ERRCHK(acPrintMeshInfoTFM(local_info) == 0);

        // Select device
        int original_rank{MPI_PROC_NULL};
        ERRCHK_MPI_API(MPI_Comm_rank(MPI_COMM_WORLD, &original_rank));

        int nprocs{0};
        ERRCHK_MPI_API(MPI_Comm_size(MPI_COMM_WORLD, &nprocs));

        int device_count{-1};
        ERRCHK_CUDA_API(cudaGetDeviceCount(&device_count));

        const int device_id{original_rank % device_count};
        ERRCHK_CUDA_API(cudaSetDevice(device_id));
        ERRCHK_CUDA_API(cudaDeviceSynchronize());

        // Create device
        Device device{nullptr};
        ERRCHK_AC(acDeviceCreate(device_id, local_info, &device));

        // Setup device memory
        AcReal stencils[NUM_STENCILS][STENCIL_DEPTH][STENCIL_HEIGHT][STENCIL_WIDTH]{};
        ERRCHK(get_stencil_coeffs(local_info, stencils) == 0);
        ERRCHK_AC(acDeviceLoadStencils(device, STREAM_DEFAULT, stencils));
        ERRCHK_AC(acDevicePrintInfo(device));

        // Read & Write
        VertexBufferArray vba{};
        ERRCHK_AC(acDeviceGetVBA(device, &vba));
        ac::mpi::write_collective_simple(cart_comm,
                                         ac::mpi::get_dtype<AcReal>(),
                                         global_nn,
                                         local_nn_offset,
                                         vba.in[VTXBUF_LNRHO],
                                         "test.dat");
        ac::mpi::read_collective_simple(cart_comm,
                                        ac::mpi::get_dtype<AcReal>(),
                                        global_nn,
                                        local_nn_offset,
                                        "test.dat",
                                        vba.out[VTXBUF_LNRHO]);

        // Dryrun
        const AcMeshDims dims{acGetMeshDims(local_info)};

        std::vector<Kernel> hydro_kernels{singlepass_solve_step0};
        std::vector<Kernel> tfm_kernels{singlepass_solve_step0_tfm_b11,
                                        singlepass_solve_step0_tfm_b12,
                                        singlepass_solve_step0_tfm_b21,
                                        singlepass_solve_step0_tfm_b22};
        std::vector<Kernel> hydro_kernels{singlepass_solve_step1};
        std::vector<Kernel> tfm_kernels{singlepass_solve_step1_tfm_b11,
                                        singlepass_solve_step1_tfm_b12,
                                        singlepass_solve_step1_tfm_b21,
                                        singlepass_solve_step1_tfm_b22};
        std::vector<Kernel> hydro_kernels{singlepass_solve_step2};
        std::vector<Kernel> tfm_kernels{singlepass_solve_step2_tfm_b11,
                                        singlepass_solve_step2_tfm_b12,
                                        singlepass_solve_step2_tfm_b21,
                                        singlepass_solve_step2_tfm_b22};
        std::vector<Kernel> all_kernels;
        all_kernels.insert(all_kernels.end(), hydro_kernels.begin(), hydro_kernels.end());
        all_kernels.insert(all_kernels.end(), tfm_kernels.begin(), tfm_kernels.end());
        all_kernels.insert(all_kernels.end(), hydro_kernels.begin(), hydro_kernels.end());
        all_kernels.insert(all_kernels.end(), tfm_kernels.begin(), tfm_kernels.end());
        all_kernels.insert(all_kernels.end(), hydro_kernels.begin(), hydro_kernels.end());
        all_kernels.insert(all_kernels.end(), tfm_kernels.begin(), tfm_kernels.end());

        for (const auto& kernel : all_kernels)
            ERRCHK_AC(acDeviceLaunchKernel(device, STREAM_DEFAULT, kernel, dims.n0, dims.n1));

        for (const auto& kernel : all_kernels) {
            BENCHMARK(acDeviceLaunchKernel(device, STREAM_DEFAULT, kernel, dims.n0, dims.n1));
        }

        ERRCHK_AC(acDeviceResetMesh(device, STREAM_DEFAULT));
        ERRCHK_AC(acDeviceSwapBuffers(device));
        ERRCHK(init_tfm_profiles(device) == 0);

        // Write data out
        ERRCHK(write_diagnostic_step(cart_comm, device, 0) == 0);

        const auto global_nn_offset{ac::mpi::get_global_nn_offset(cart_comm, global_nn)};
        const auto local_mm{ac::mpi::get_local_mm(cart_comm, global_nn, local_nn_offset)};
        const auto local_nn{ac::mpi::get_local_nn(cart_comm, global_nn)};
        auto iotask = ac::io::AsyncWriteTask<AcReal>(global_nn,
                                                     global_nn_offset,
                                                     local_mm,
                                                     local_nn,
                                                     local_nn_offset);
        const size_t count{prod(local_mm)};
        iotask.launch_write_collective(cart_comm,
                                       ac::mr::device_ptr<AcReal>{count, vba.in[VTXBUF_LNRHO]},
                                       "test.out");
        iotask.wait_write_collective();

// Halo exchange
        std::vector<ac::mr::device_ptr<AcReal>> hydro_fields{
            ac::mr::device_ptr<AcReal>{count, vba.in[VTXBUF_LNRHO]},
            ac::mr::device_ptr<AcReal>{count, vba.in[VTXBUF_UUX]},
            ac::mr::device_ptr<AcReal>{count, vba.in[VTXBUF_UUY]},
            ac::mr::device_ptr<AcReal>{count, vba.in[VTXBUF_UUZ]},
        };
        std::vector<ac::mr::device_ptr<AcReal>> tfm_fields{
            ac::mr::device_ptr<AcReal>{count, vba.in[TF_a11_x]},
            ac::mr::device_ptr<AcReal>{count, vba.in[TF_a11_y]},
            ac::mr::device_ptr<AcReal>{count, vba.in[TF_a11_z]},
            ac::mr::device_ptr<AcReal>{count, vba.in[TF_a12_x]},
            ac::mr::device_ptr<AcReal>{count, vba.in[TF_a12_y]},
            ac::mr::device_ptr<AcReal>{count, vba.in[TF_a12_z]},
            ac::mr::device_ptr<AcReal>{count, vba.in[TF_a21_x]},
            ac::mr::device_ptr<AcReal>{count, vba.in[TF_a21_y]},
            ac::mr::device_ptr<AcReal>{count, vba.in[TF_a21_z]},
            ac::mr::device_ptr<AcReal>{count, vba.in[TF_a22_x]},
            ac::mr::device_ptr<AcReal>{count, vba.in[TF_a22_y]},
            ac::mr::device_ptr<AcReal>{count, vba.in[TF_a22_z]},
        };
        std::vector<ac::mr::device_ptr<AcReal>> derived_fields{
            ac::mr::device_ptr<AcReal>{count, vba.in[TF_uxb11_x]},
            ac::mr::device_ptr<AcReal>{count, vba.in[TF_uxb11_y]},
            ac::mr::device_ptr<AcReal>{count, vba.in[TF_uxb11_z]},
            ac::mr::device_ptr<AcReal>{count, vba.in[TF_uxb12_x]},
            ac::mr::device_ptr<AcReal>{count, vba.in[TF_uxb12_y]},
            ac::mr::device_ptr<AcReal>{count, vba.in[TF_uxb12_z]},
            ac::mr::device_ptr<AcReal>{count, vba.in[TF_uxb21_x]},
            ac::mr::device_ptr<AcReal>{count, vba.in[TF_uxb21_y]},
            ac::mr::device_ptr<AcReal>{count, vba.in[TF_uxb21_z]},
            ac::mr::device_ptr<AcReal>{count, vba.in[TF_uxb22_x]},
            ac::mr::device_ptr<AcReal>{count, vba.in[TF_uxb22_y]},
            ac::mr::device_ptr<AcReal>{count, vba.in[TF_uxb22_z]},
        };

        ac::ndbuffer<AcReal, ac::mr::host_memory_resource> debug_mesh{local_mm};
        if (nprocs == 1) {
            std::iota(debug_mesh.begin(),
                      debug_mesh.end(),
                      static_cast<AcReal>(ac::mpi::get_rank(cart_comm)) *
                          static_cast<AcReal>(count));
        }
        else {
            std::fill(debug_mesh.begin(),
                      debug_mesh.end(),
                      static_cast<AcReal>(ac::mpi::get_rank(cart_comm)));
        }
        // MPI_SYNCHRONOUS_BLOCK_START(cart_comm)
        // debug_mesh.display();
        // MPI_SYNCHRONOUS_BLOCK_END(cart_comm)
        for (auto& ptr : hydro_fields)
            ac::mr::copy(ac::mr::host_ptr<AcReal>{debug_mesh.size(), debug_mesh.data()}, ptr);

        auto hydro_he{
            ac::comm::AsyncHaloExchangeTask<AcReal, ac::mr::device_memory_resource>(local_mm,
                                                                                    local_nn,
                                                                                    local_nn_offset,
                                                                                    hydro_fields
                                                                                        .size())};
        hydro_he.launch(cart_comm, hydro_fields);
        hydro_he.wait(hydro_fields);

        auto tfm_he{
            ac::comm::AsyncHaloExchangeTask<AcReal, ac::mr::device_memory_resource>(local_mm,
                                                                                    local_nn,
                                                                                    local_nn_offset,
                                                                                    tfm_fields
                                                                                        .size())};
        tfm_he.launch(cart_comm, tfm_fields);
        tfm_he.wait(tfm_fields);

        for (auto& ptr : hydro_fields)
            ac::mr::copy(ptr, ac::mr::host_ptr<AcReal>{debug_mesh.size(), debug_mesh.data()});
        // MPI_SYNCHRONOUS_BLOCK_START(cart_comm)
        // debug_mesh.display();
        // MPI_SYNCHRONOUS_BLOCK_END(cart_comm)

        // Cleanup
        ERRCHK_AC(acDeviceDestroy(device));
        ac::mpi::cart_comm_destroy(cart_comm);
#endif
    }
    catch (std::exception& e) {
        std::cerr << e.what() << std::endl;
        ac::mpi::abort();
    }
    ac::mpi::finalize();
    std::cout << "Complete" << std::endl;
    return EXIT_SUCCESS;
}
