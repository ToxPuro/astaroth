#include <cstdlib>
#include <exception>
#include <iostream>

#include "astaroth.h"

#include "acm/detail/errchk_mpi.h"
#include "acm/detail/mpi_utils.h"
#include "acm/detail/type_conversion.h"

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

template <typename T, typename U> static static_cast(const ac::vector<U>& in)
{
    ac::vector<T> out(in.size());
    for (size_t i{0}; i < in.size(); ++i)
        out[i] = static_cast<T>(in[i]);
    return out;
}

static AcMeshInfo
get_local_mesh_info(const MPI_Comm cart_comm, const AcMeshInfo info)
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
    const Dims local_ss{global_ss / static_cast<AcReal>(decomp)};

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

int
acInitTFMProfiles(const Device device)
{
    const AcMeshInfo info  = acDeviceGetLocalConfig(device);
    const AcReal global_lz = info.real_params[AC_sz];
    const size_t global_nz = as<size_t>(info.int3_params[AC_global_grid_n].z);
    const long offset      = -info.int_params[AC_nz_min]; // TODO take multigpu into account
    const size_t local_mz  = as<size_t>(info.int_params[AC_mz]);

    const AcReal amplitude  = info.real_params[AC_profile_amplitude];
    const AcReal wavenumber = info.real_params[AC_profile_wavenumber];

    AcReal host_profile[local_mz];

    // All to zero
    acHostInitProfileToValue(0, local_mz, host_profile);
    acDeviceLoadProfile(device, host_profile, local_mz, PROFILE_B11mean_x);
    acDeviceLoadProfile(device, host_profile, local_mz, PROFILE_B11mean_y);
    acDeviceLoadProfile(device, host_profile, local_mz, PROFILE_B11mean_z);
    acDeviceLoadProfile(device, host_profile, local_mz, PROFILE_B12mean_x);
    acDeviceLoadProfile(device, host_profile, local_mz, PROFILE_B12mean_y);
    acDeviceLoadProfile(device, host_profile, local_mz, PROFILE_B12mean_z);
    acDeviceLoadProfile(device, host_profile, local_mz, PROFILE_B21mean_x);
    acDeviceLoadProfile(device, host_profile, local_mz, PROFILE_B21mean_y);
    acDeviceLoadProfile(device, host_profile, local_mz, PROFILE_B21mean_z);
    acDeviceLoadProfile(device, host_profile, local_mz, PROFILE_B22mean_x);
    acDeviceLoadProfile(device, host_profile, local_mz, PROFILE_B22mean_y);
    acDeviceLoadProfile(device, host_profile, local_mz, PROFILE_B22mean_z);

    // B1c (here B11) and B2c (here B21) to cosine
    acHostInitProfileToCosineWave(global_lz, global_nz, offset, amplitude, wavenumber, local_mz,
                                  host_profile);
    acDeviceLoadProfile(device, host_profile, local_mz, PROFILE_B11mean_x);
    acDeviceLoadProfile(device, host_profile, local_mz, PROFILE_B21mean_y);

    // B1s (here B12) and B2s (here B22)
    acHostInitProfileToSineWave(global_lz, global_nz, offset, amplitude, wavenumber, local_mz,
                                host_profile);
    acDeviceLoadProfile(device, host_profile, local_mz, PROFILE_B12mean_x);
    acDeviceLoadProfile(device, host_profile, local_mz, PROFILE_B22mean_y);

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
        AcReal stencils[NUM_STENCILS][STENCIL_DEPTH][STENCIL_HEIGHT][STENCIL_WIDTH];
        ERRCHK(get_stencil_coeffs(local_info, stencils) == 0);
        ERRCHK_AC(acDeviceLoadStencils(device, STREAM_DEFAULT, stencils));
        ERRCHK_AC(acDevicePrintInfo(device));

        // Dryrun
        const AcMeshDims dims = acGetMeshDims(local_info);
        acDeviceIntegrateSubstep(device, STREAM_DEFAULT, 0, dims.n0, dims.n1, 1e-5);
        acDeviceResetMesh(device, STREAM_DEFAULT);

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
