#include <cstdlib>
#include <exception>
#include <iostream>

#include "astaroth.h"
#include "device_detail.h"

#include "acm/detail/errchk_mpi.h"
#include "acm/detail/mpi_utils.h"
#include "acm/detail/type_conversion.h"

#include "acm/detail/io.h"
#include "acm/detail/memory_resource.h"

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

static AcMeshInfo
get_local_mesh_info(const MPI_Comm& cart_comm, const AcMeshInfo& info)
{
    // Calculate local dimensions
    Shape global_nn{as<uint64_t>(info.int_params[AC_global_nx]),
                    as<uint64_t>(info.int_params[AC_global_ny]),
                    as<uint64_t>(info.int_params[AC_global_nz])};
    Dims global_ss{static_cast<AcReal>(info.real_params[AC_global_sx]),
                   static_cast<AcReal>(info.real_params[AC_global_sy]),
                   static_cast<AcReal>(info.real_params[AC_global_sz])};

    const Shape decomp{ac::mpi::get_decomposition(cart_comm)};
    const Shape local_nn{global_nn / decomp};
    const Dims local_ss{global_ss / static_cast_vec<AcReal>(decomp)};

    const Index coords{ac::mpi::get_coords(cart_comm)};
    const Index global_nn_offset{coords * local_nn};

    // Fill AcMeshInfo
    AcMeshInfo local_info{info};

    local_info.int_params[AC_nx] = as<int>(local_nn[0]);
    local_info.int_params[AC_ny] = as<int>(local_nn[1]);
    local_info.int_params[AC_nz] = as<int>(local_nn[2]);

    local_info.int3_params[AC_multigpu_offset] = {
        as<int>(global_nn_offset[0]),
        as<int>(global_nn_offset[1]),
        as<int>(global_nn_offset[2]),
    };

    local_info.real_params[AC_sx] = static_cast<AcReal>(local_ss[0]);
    local_info.real_params[AC_sy] = static_cast<AcReal>(local_ss[1]);
    local_info.real_params[AC_sz] = static_cast<AcReal>(local_ss[2]);

    // Backwards compatibility
    local_info.int3_params[AC_global_grid_n] = {
        as<int>(global_nn[0]),
        as<int>(global_nn[1]),
        as<int>(global_nn[2]),
    };

    ERRCHK(acHostUpdateLocalBuiltinParams(&local_info) == 0);
    ERRCHK(acHostUpdateMHDSpecificParams(&local_info) == 0);
    ERRCHK(acHostUpdateTFMSpecificGlobalParams(&local_info) == 0);

    // Others to ensure nothing is left uninitialized
    local_info.int_params[AC_init_type]     = 0;
    local_info.int_params[AC_step_number]   = 0;
    local_info.real_params[AC_dt]           = 0;
    local_info.real3_params[AC_dummy_real3] = (AcReal3){0, 0, 0};

    ERRCHK(acVerifyMeshInfo(local_info) == 0);
    return local_info;
}

static int
init_tfm_profiles(const Device& device)
{
    AcMeshInfo info{};
    ERRCHK_AC(acDeviceGetLocalConfig(device, &info));

    const AcReal global_sz = info.real_params[AC_global_sz];
    const size_t global_nz = as<size_t>(info.int3_params[AC_global_grid_n].z);
    const long offset      = -info.int_params[AC_nz_min] + info.int3_params[AC_multigpu_offset].z;
    const size_t local_mz  = as<size_t>(info.int_params[AC_mz]);

    const AcReal amplitude  = info.real_params[AC_profile_amplitude];
    const AcReal wavenumber = info.real_params[AC_profile_wavenumber];

    auto host_profile{std::make_unique<AcReal[]>(local_mz)};

    // All to zero
    acHostInitProfileToValue(0, local_mz, host_profile.get());
    acDeviceLoadProfile(device, host_profile.get(), local_mz, PROFILE_B11mean_x);
    acDeviceLoadProfile(device, host_profile.get(), local_mz, PROFILE_B11mean_y);
    acDeviceLoadProfile(device, host_profile.get(), local_mz, PROFILE_B11mean_z);
    acDeviceLoadProfile(device, host_profile.get(), local_mz, PROFILE_B12mean_x);
    acDeviceLoadProfile(device, host_profile.get(), local_mz, PROFILE_B12mean_y);
    acDeviceLoadProfile(device, host_profile.get(), local_mz, PROFILE_B12mean_z);
    acDeviceLoadProfile(device, host_profile.get(), local_mz, PROFILE_B21mean_x);
    acDeviceLoadProfile(device, host_profile.get(), local_mz, PROFILE_B21mean_y);
    acDeviceLoadProfile(device, host_profile.get(), local_mz, PROFILE_B21mean_z);
    acDeviceLoadProfile(device, host_profile.get(), local_mz, PROFILE_B22mean_x);
    acDeviceLoadProfile(device, host_profile.get(), local_mz, PROFILE_B22mean_y);
    acDeviceLoadProfile(device, host_profile.get(), local_mz, PROFILE_B22mean_z);

    // B1c (here B11) and B2c (here B21) to cosine
    acHostInitProfileToCosineWave(global_sz, global_nz, offset, amplitude, wavenumber, local_mz,
                                  host_profile.get());
    acDeviceLoadProfile(device, host_profile.get(), local_mz, PROFILE_B11mean_x);
    acDeviceLoadProfile(device, host_profile.get(), local_mz, PROFILE_B21mean_y);

    // B1s (here B12) and B2s (here B22)
    acHostInitProfileToSineWave(global_sz, global_nz, offset, amplitude, wavenumber, local_mz,
                                host_profile.get());
    acDeviceLoadProfile(device, host_profile.get(), local_mz, PROFILE_B12mean_x);
    acDeviceLoadProfile(device, host_profile.get(), local_mz, PROFILE_B22mean_y);

    return 0;
}

namespace acc {
Shape
get_global_nn(const AcMeshInfo& info)
{
    return Shape{as<uint64_t>(info.int_params[AC_global_nx]),
                 as<uint64_t>(info.int_params[AC_global_ny]),
                 as<uint64_t>(info.int_params[AC_global_nz])};
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
    const size_t mz = as<size_t>(info.int3_params[AC_global_grid_n].z +
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
        ac::mpi::write_collective_simple(parent_comm, ac::mpi::get_dtype<AcReal>(),
                                         acc::get_global_nn(local_info), acc::get_local_nn_offset(),
                                         vba.in[i], std::string(filepath));
    }
    for (int i = 0; i < NUM_PROFILES; ++i) {
        char filepath[4096];
        sprintf(filepath, "debug-step-%012zu-tfm-%s.profile", step, profile_names[i]);
        printf("Writing %s\n", filepath);
        const Shape profile_global_nz{as<uint64_t>(local_info.int_params[AC_global_nz])};
        const Shape profile_local_mz{as<uint64_t>(local_info.int_params[AC_mz])};
        const Shape profile_local_nz{as<uint64_t>(local_info.int_params[AC_nz])};
        const Shape profile_local_nz_offset{as<uint64_t>(local_info.int_params[AC_nz_min])};
        const Index coords{ac::mpi::get_coords(parent_comm)[2]};
        const Shape profile_global_nz_offset{coords * profile_local_nz};

        const int rank{ac::mpi::get_rank(parent_comm)};
        const Index coords_3d{ac::mpi::get_coords(parent_comm)};
        const Shape decomp_3d{ac::mpi::get_decomposition(parent_comm)};
        const int color = (coords_3d[0] + coords_3d[1] * decomp_3d[0]) == 0 ? 0 : MPI_UNDEFINED;

        MPI_Comm profile_comm{MPI_COMM_NULL};
        ERRCHK_MPI_API(MPI_Comm_split(parent_comm, color, rank, &profile_comm));

        if (profile_comm != MPI_COMM_NULL) {
            ac::mpi::write_collective(profile_comm, ac::mpi::get_dtype<AcReal>(), profile_global_nz,
                                      profile_global_nz_offset, profile_local_mz, profile_local_nz,
                                      profile_local_nz_offset, vba.profiles.in[i],
                                      std::string(filepath));
            ERRCHK_MPI_API(MPI_Comm_free(&profile_comm));
        }
    }
    return 0;
}

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

        Shape global_nn{as<uint64_t>(raw_info.int_params[AC_global_nx]),
                        as<uint64_t>(raw_info.int_params[AC_global_ny]),
                        as<uint64_t>(raw_info.int_params[AC_global_nz])};
        Dims global_ss{static_cast<AcReal>(raw_info.real_params[AC_global_sx]),
                       static_cast<AcReal>(raw_info.real_params[AC_global_sy]),
                       static_cast<AcReal>(raw_info.real_params[AC_global_sz])};
        Index local_nn_offset{(STENCIL_WIDTH - 1) / 2, (STENCIL_HEIGHT - 1) / 2,
                              (STENCIL_DEPTH - 1) / 2};

        // Create the Cartesian communicator
        MPI_Comm cart_comm = ac::mpi::cart_comm_create(MPI_COMM_WORLD, global_nn);

        // Fill the local mesh configuration (overwrite the earlier info
        const AcMeshInfo local_info = get_local_mesh_info(cart_comm, raw_info);
        ERRCHK(acPrintMeshInfoTFM(local_info) == 0);

        // Select device
        int original_rank, nprocs;
        ERRCHK_MPI_API(MPI_Comm_rank(MPI_COMM_WORLD, &original_rank));
        ERRCHK_MPI_API(MPI_Comm_size(MPI_COMM_WORLD, &nprocs));

        int device_count;
        ERRCHK_CUDA_API(cudaGetDeviceCount(&device_count));

        const int device_id = original_rank % device_count;
        ERRCHK_CUDA_API(cudaSetDevice(device_id));
        ERRCHK_CUDA_API(cudaDeviceSynchronize());

        // Create device
        Device device;
        ERRCHK_AC(acDeviceCreate(device_id, local_info, &device));

        // Setup device memory
        AcReal stencils[NUM_STENCILS][STENCIL_DEPTH][STENCIL_HEIGHT][STENCIL_WIDTH]{};
        ERRCHK(get_stencil_coeffs(local_info, stencils) == 0);
        ERRCHK_AC(acDeviceLoadStencils(device, STREAM_DEFAULT, stencils));
        ERRCHK_AC(acDevicePrintInfo(device));

        // Read & Write
        VertexBufferArray vba;
        ERRCHK_AC(acDeviceGetVBA(device, &vba));
        ac::mpi::write_collective_simple(cart_comm, ac::mpi::get_dtype<AcReal>(), global_nn,
                                         local_nn_offset, vba.in[VTXBUF_LNRHO], "test.dat");
        ac::mpi::read_collective_simple(cart_comm, ac::mpi::get_dtype<AcReal>(), global_nn,
                                        local_nn_offset, "test.dat", vba.out[VTXBUF_LNRHO]);

        // Dryrun
        const AcMeshDims dims = acGetMeshDims(local_info);
        ERRCHK_AC(acDeviceIntegrateSubstep(device, STREAM_DEFAULT, 0, dims.n0, dims.n1, 1e-5));
        ERRCHK_AC(acDeviceResetMesh(device, STREAM_DEFAULT));
        ERRCHK_AC(acDeviceSwapBuffers(device));
        ERRCHK(init_tfm_profiles(device) == 0);

        // Write data out
        ERRCHK(write_diagnostic_step(cart_comm, device, 0) == 0);

        const auto global_nn_offset{ac::mpi::get_global_nn_offset(cart_comm, global_nn)};
        const auto local_mm{ac::mpi::get_local_mm(cart_comm, global_nn, local_nn_offset)};
        const auto local_nn{ac::mpi::get_local_nn(cart_comm, global_nn)};
        auto iotask = ac::io::AsyncWriteTask<AcReal>(global_nn, global_nn_offset, local_mm,
                                                     local_nn, local_nn_offset);
        iotask.launch_write_collective(cart_comm,
                                       ac::mr::device_ptr<AcReal>{prod(local_mm),
                                                                  vba.in[VTXBUF_LNRHO]},
                                       "test.out");
        iotask.wait_write_collective();
        // iotask.launch_write_collective(cart_comm,
        //                                ac::mr::device_ptr(prod(local_mm), vba.in[VTXBUF_LNRHO]),
        //                                "test_mesh.dat");
        // auto test_ptr{ac::mr::device_ptr<AcReal>(prod(local_mm), vba.in[VTXBUF_LNRHO])};

        // Cleanup
        ERRCHK_AC(acDeviceDestroy(device));
        ac::mpi::cart_comm_destroy(cart_comm);
    }
    catch (std::exception& e) {
        std::cerr << e.what() << std::endl;
        ac::mpi::abort();
    }
    ac::mpi::finalize();
    std::cout << "Complete" << std::endl;
    return EXIT_SUCCESS;
}
