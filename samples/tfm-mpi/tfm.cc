#include <iostream>
#include <cstdlib>

#include "acc-runtime/api/acc_runtime.h"

#include "acc-comm/acm.h"
#include "acc-comm/errchk.h"
#include "acc-comm/type_conversion.h"

#include "acc-comm/vector.h"

#include "tfm_utils.h"

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

constexpr size_t NDIMS{3};

// static get_local_box_size(const size_t ndims, )

static AcMeshInfo get_local_mesh_info(const MPI_Comm cart_comm, const AcMeshInfo info)
{
    return AcMeshInfo{};
    // const uint64_t global_nn[NDIMS]{
    //     as<uint64_t>(info.int_params[AC_global_nx]), 
    //     as<uint64_t>(info.int_params[AC_global_ny]), 
    //     as<uint64_t>(info.int_params[AC_global_nz])
    // };
    // const AcReal global_box_size[NDIMS]{
    //     info.real_params[AC_global_box_size_x], 
    //     info.real_params[AC_global_box_size_y], 
    //     info.real_params[AC_global_box_size_z]
    // };
    // const uint64_t rr[NDIMS]{
    //     as<uint64_t>((STENCIL_WIDTH - 1)/2), 
    //     as<uint64_t>((STENCIL_HEIGHT - 1)/2), 
    //     as<uint64_t>((STENCIL_DEPTH - 1)/2)
    // };
    // uint64_t local_nn[NDIMS]{};
    // uint64_t global_nn_offset[NDIMS]{};
    // double box_size[NDIMS]{};

    // ERRCHK_ACM(ACM_Get_local_nn(cart_comm, NDIMS, global_nn, local_nn));
    // ERRCHK_ACM(ACM_Get_global_nn_offset(cart_comm, NDIMS, global_nn, global_nn_offset));
    // ERRCHK(get_local_box_size())

    // AcMeshInfo local_info{info};
    // local_info.int_params[AC_nx] = local_nn[0];
    // local_info.int_params[AC_ny] = local_nn[1];
    // local_info.int_params[AC_nz] = local_nn[2];

    // local_info.int3_params[AC_multigpu_offset] = (int3){
    //     global_nn_offset[0],
    //     global_nn_offset[1],
    //     global_nn_offset[2],
    // };

    // local_info.int3_params[AC_global_grid_n] = (int3){
    //     global_nn[0],
    //     global_nn[1],
    //     global_nn[2],
    // };
    // ERRCHK(acHostUpdateBuiltinParams(&local_info) == 0);
    // ERRCHK(acHostUpdateMHDSpecificParams(&local_info) == 0);
    // ERRCHK(acHostUpdateTFMSpecificGlobalParams(&local_info) == 0);

    // return local_info;
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

        AcMeshInfo info{};
        if (args.config_path) {
            ERRCHK(acParseINI(args.config_path, &info) == 0);
        } else {
            const std::string config_path{AC_DEFAULT_TFM_CONFIG};
            PRINT_LOG("No config path supplied, using %s", config_path.c_str());
            ERRCHK(acParseINI(config_path.c_str(), &info) == 0);
        }
        ERRCHK(acPrintMeshInfo(info) == 0);

        const uint64_t global_nn[NDIMS]{
            as<uint64_t>(info.int_params[AC_global_nx]), 
            as<uint64_t>(info.int_params[AC_global_ny]), 
            as<uint64_t>(info.int_params[AC_global_nz])
        };
        PRINT_DEBUG(global_nn);

        MPI_Comm cart_comm;
        ERRCHK_ACM(ACM_MPI_Cart_comm_create(MPI_COMM_WORLD, NDIMS, global_nn, &cart_comm));

        const AcMeshInfo local_info = get_local_mesh_info(cart_comm, info);
        ERRCHK(acPrintMeshInfo(local_info) == 0);

    // constexpr size_t NDIMS{3};
    // const uint64_t global_nn[NDIMS];
    // const uint64_t local_nn[NDIMS];
    // const uint64_t global_nn_offset[NDIMS];
    // const uint64_t rr[NDIMS];
    

    } catch (const std::exception& e){
        std::cerr << e.what() << std::endl;
        ERRCHK_ACM(ACM_MPI_Abort());
    }

    // Finalize MPI
    ERRCHK_ACM(ACM_MPI_Finalize());
    std::cout << "Complete" << std::endl;
    return EXIT_SUCCESS;
}
