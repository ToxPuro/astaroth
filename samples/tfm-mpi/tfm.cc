#include <iostream>
#include <cstdlib>
#include <functional>

#include "acc-runtime/api/acc_runtime.h"

#include "acc-comm/acm.h"
#include "acc-comm/errchk.h"
#include "acc-comm/type_conversion.h"

#include "acc-comm/vector.h"

#include "tfm_utils.h"

#include "acc-comm/print_debug.h"

#include <mpi.h>
#include "acc-comm/mpi_utils.h"

// #include "astaroth.h"
#include "stencil_loader.h"

#define ERRCHK_ACM(errcode)                                                                        \
    do {                                                                                           \
        const ACM_Errorcode _tmp_acm_api_errcode_ = (errcode);                                     \
        if (_tmp_acm_api_errcode_ != ACM_ERRORCODE_SUCCESS) {                                      \
            errchk_print_error(__func__, __FILE__, __LINE__, #errcode,                             \
                               ACM_Get_errorcode_description(errcode));                            \
            errchk_print_stacktrace();                                                             \
            MPI_Abort(MPI_COMM_WORLD, -1);                                                         \
        }                                                                                          \
    } while (0)

#define ERRCHK_AC(errcode)                                                                        \
    do {                                                                                           \
        const AcResult _tmp_ac_api_errcode_ = (errcode);                                     \
        if (_tmp_ac_api_errcode_ != AC_SUCCESS) {                                      \
            errchk_print_error(__func__, __FILE__, __LINE__, #errcode, "Astaroth error");                            \
            errchk_print_stacktrace();                                                             \
            MPI_Abort(MPI_COMM_WORLD, -1);                                                         \
        }                                                                                          \
    } while (0)

constexpr size_t NDIMS{3};

static int get_local_box_size(const size_t ndims, const uint64_t* global_nn, const double* global_box_size, const uint64_t* local_nn, double* local_box_size)
{
    for (size_t i = 0; i < ndims; ++i) {
        
        // Division by zero
        if (!global_nn[i])
            return -1;

        local_box_size[i] = (static_cast<double>(local_nn[i]) / static_cast<double>(global_nn[i])) * global_box_size[i];
    }
    
    return 0;
}

// static VertexBufferArray* vba_alloc(const Shape& mm)
// {
//     auto* vba = new VertexBufferArray;
//     *vba = acVBACreate(mm[0], mm[1], mm[2]);
//     return vba;
// }

// static void vba_dealloc(VertexBufferArray* vba)
// {
//     acVBADestroy(vba);
//     delete vba;
// }

using vba_ptr_t = std::unique_ptr<VertexBufferArray, std::function<void(VertexBufferArray*)>>;

static auto
make_vba(const Shape& mm)
{
    return vba_ptr_t{
        [&mm](){
            auto* vba = new VertexBufferArray;
            *vba = acVBACreate(mm[0], mm[1], mm[2]);
            return vba;
        }(), 
        [](VertexBufferArray* vba){
            acVBADestroy(vba); 
            delete vba;
        }
    };
}

// using device_ptr_t = std::unique_ptr<Device, std::function<void(Device*)>>;

// static auto
// make_device(const int id, const AcMeshInfo& local_info)
// {
//     return device_ptr_t{
//         [id, &local_info](){
//             auto* device = new Device;
//             ERRCHK_AC(acDeviceCreate(id, local_info, device));
//             return device;
//         }(), 
//         [](Device* device){
//             acDeviceDestroy(*device); 
//             delete device;
//         }
//     };
// }

static int3 make_int3(const Shape& shape)
{
    ERRCHK(shape.size() == 3);
    return {as<int>(shape[0]), as<int>(shape[1]), as<int>(shape[2])};
}

static AcMeshInfo get_local_mesh_info(const MPI_Comm cart_comm, const AcMeshInfo info)
{
    const uint64_t global_nn[NDIMS]{
        as<uint64_t>(info.int_params[AC_global_nx]), 
        as<uint64_t>(info.int_params[AC_global_ny]), 
        as<uint64_t>(info.int_params[AC_global_nz])
    };
    const AcReal global_box_size[NDIMS]{
        static_cast<double>(info.real_params[AC_global_sx]), 
        static_cast<double>(info.real_params[AC_global_sy]), 
        static_cast<double>(info.real_params[AC_global_sz])
    };
    uint64_t local_nn[NDIMS]{};
    ERRCHK_ACM(ACM_Get_local_nn(cart_comm, NDIMS, global_nn, local_nn));
    
    uint64_t global_nn_offset[NDIMS]{};
    ERRCHK_ACM(ACM_Get_global_nn_offset(cart_comm, NDIMS, global_nn, global_nn_offset));
    
    double local_box_size[NDIMS]{};
    ERRCHK(get_local_box_size(NDIMS, global_nn, global_box_size, local_nn, local_box_size) == 0);

    AcMeshInfo local_info{info};
    local_info.int_params[AC_nx] = as<int>(local_nn[0]);
    local_info.int_params[AC_ny] = as<int>(local_nn[1]);
    local_info.int_params[AC_nz] = as<int>(local_nn[2]);

    local_info.int3_params[AC_multigpu_offset] = {
        as<int>(global_nn_offset[0]),
        as<int>(global_nn_offset[1]),
        as<int>(global_nn_offset[2]),
    };

    local_info.real_params[AC_sx] = static_cast<AcReal>(local_box_size[0]);
    local_info.real_params[AC_sy] = static_cast<AcReal>(local_box_size[1]);
    local_info.real_params[AC_sz] = static_cast<AcReal>(local_box_size[2]);
    
    // Backwards compatibility
    local_info.int3_params[AC_global_grid_n] = {
        as<int>(global_nn[0]),
        as<int>(global_nn[1]),
        as<int>(global_nn[2]),
    };

    ERRCHK(acHostUpdateLocalBuiltinParams(&local_info) == 0);
    ERRCHK(acHostUpdateMHDSpecificParams(&local_info) == 0);
    ERRCHK(acHostUpdateTFMSpecificGlobalParams(&local_info) == 0);

    WARNCHK(acVerifyMeshInfo(local_info) == 0);
    return local_info;
}

int main(int argc, char* argv[])
{
    // Init MPI
    ERRCHK_ACM(ACM_MPI_Init_funneled());

    try {

        // Parse arguments
        Arguments args{};
        ERRCHK(acParseArguments(argc, argv, &args) == 0);
        ERRCHK(acPrintArguments(args) == 0);

        // Load configuration
        AcMeshInfo raw_info{};
        if (args.config_path) {
            ERRCHK(acParseINI(args.config_path, &raw_info) == 0);
        } else {
            const std::string config_path{AC_DEFAULT_TFM_CONFIG};
            PRINT_LOG("No config path supplied, using %s", config_path.c_str());
            ERRCHK(acParseINI(config_path.c_str(), &raw_info) == 0);
        }
        // ERRCHK(acPrintMeshInfo(raw_info) == 0);

        const uint64_t global_nn_c[NDIMS]{
            as<uint64_t>(raw_info.int_params[AC_global_nx]), 
            as<uint64_t>(raw_info.int_params[AC_global_ny]), 
            as<uint64_t>(raw_info.int_params[AC_global_nz])
        };
        PRINT_DEBUG_ARRAY(NDIMS, global_nn_c);

        // Create the Cartesian communicator
        MPI_Comm cart_comm;
        ERRCHK_ACM(ACM_MPI_Cart_comm_create(MPI_COMM_WORLD, NDIMS, global_nn_c, &cart_comm));

        // Fill the local mesh configuration (overwrite the earlier info
        const AcMeshInfo local_info = get_local_mesh_info(cart_comm, raw_info);
        ERRCHK(acPrintMeshInfo(local_info) == 0);

        // Select device
        int original_rank, nprocs;
        ERRCHK_MPI_API(MPI_Comm_rank(MPI_COMM_WORLD, &original_rank));
        ERRCHK_MPI_API(MPI_Comm_size(MPI_COMM_WORLD, &nprocs));

        int device_count;
        ERRCHK_CUDA_API(cudaGetDeviceCount(&device_count));
        ERRCHK_CUDA_API(cudaSetDevice(original_rank % device_count));
        ERRCHK_CUDA_API(cudaDeviceSynchronize());
    
        // Setup mesh dimensions
        const Shape global_nn {
            as<uint64_t>(local_info.int_params[AC_global_nx]), 
            as<uint64_t>(local_info.int_params[AC_global_ny]), 
            as<uint64_t>(local_info.int_params[AC_global_nz])
        };
        const Shape local_nn {
            as<uint64_t>(local_info.int_params[AC_nx]), 
            as<uint64_t>(local_info.int_params[AC_ny]), 
            as<uint64_t>(local_info.int_params[AC_nz])
        };
        const Index rr {
            as<uint64_t>((STENCIL_WIDTH - 1)/2), 
            as<uint64_t>((STENCIL_HEIGHT - 1)/2), 
            as<uint64_t>((STENCIL_DEPTH - 1)/2)
        };
        const Shape local_mm {
            as<uint64_t>(local_info.int_params[AC_mx]), 
            as<uint64_t>(local_info.int_params[AC_my]), 
            as<uint64_t>(local_info.int_params[AC_mz])
        };


        // Setup device memory
        const cudaStream_t stream = 0;
        AcReal stencils[NUM_STENCILS][STENCIL_DEPTH][STENCIL_HEIGHT][STENCIL_WIDTH];
        get_stencil_coeffs(local_info, stencils);
        
        auto vba_ptr = make_vba(local_mm);
        ERRCHK_AC(acLoadMeshInfo(local_info, stream));
        ERRCHK_AC(acLoadRealUniform(stream, AC_dt, 1e-3));
        ERRCHK_AC(acLoadIntUniform(stream, AC_step_number, 0));
        ERRCHK_AC(acLoadStencils(stream, stencils));
        ERRCHK_CUDA_API(cudaDeviceSynchronize());

        ERRCHK_AC(acVBAReset(stream, vba_ptr.get()));
        ERRCHK_CUDA_API(cudaDeviceSynchronize());

        // ERRCHK_AC(acLaunchKernel(singlepass_solve, stream, make_int3(rr), make_int3(rr + Shape(rr.size(), 1)), *(vba_ptr.get())));
        ERRCHK_AC(acLaunchKernel(singlepass_solve, stream, make_int3(rr), make_int3(rr + local_nn), *vba_ptr));
        ERRCHK_CUDA_KERNEL();
        ERRCHK_CUDA_API(cudaStreamSynchronize(stream));
        ERRCHK_CUDA_API(cudaDeviceSynchronize());


        ERRCHK_ACM(ACM_MPI_Cart_comm_destroy(&cart_comm));

    } catch (const std::exception& e){
        std::cerr << e.what() << std::endl;
        ERRCHK_ACM(ACM_MPI_Abort());
    }

    // Finalize MPI
    ERRCHK_ACM(ACM_MPI_Finalize());
    std::cout << "Complete" << std::endl;
    return EXIT_SUCCESS;
}
