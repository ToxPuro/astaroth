#pragma once

#include "acc_runtime.h"

typedef struct {
    char* config_path;
    int global_nn_override[3];
} Arguments;

#ifdef __cplusplus
extern "C" {
#endif

int acParseArguments(const int argc, char* argv[], Arguments* args);

int acParseINI(const char* filepath, AcMeshInfo* info);

int acPrintArguments(const Arguments args);

int acHostUpdateLocalBuiltinParams(AcMeshInfo* config);

int acHostUpdateForcingParams(AcMeshInfo* info);

int acHostUpdateMHDSpecificParams(AcMeshInfo* info);

int acHostUpdateTFMSpecificGlobalParams(AcMeshInfo* info);

int acPrintMeshInfoTFM(const AcMeshInfo config);

AcReal calc_timestep(const AcReal uumax, const AcReal vAmax, const AcReal shock_max,
                     const AcMeshInfo info);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
#include <vector>

static const std::vector<Field> hydro_fields{VTXBUF_LNRHO, VTXBUF_UUX, VTXBUF_UUY, VTXBUF_UUZ};
static const std::vector<Field> tfm_fields{TF_a11_x,
                                           TF_a11_y,
                                           TF_a11_z,
                                           TF_a12_x,
                                           TF_a12_y,
                                           TF_a12_z,
                                           TF_a21_x,
                                           TF_a21_y,
                                           TF_a21_z,
                                           TF_a22_x,
                                           TF_a22_y,
                                           TF_a22_z};

static const std::vector<Field> uxb_fields{TF_uxb11_x,
                                           TF_uxb11_y,
                                           TF_uxb11_z,
                                           TF_uxb12_x,
                                           TF_uxb12_y,
                                           TF_uxb12_z,
                                           TF_uxb21_x,
                                           TF_uxb21_y,
                                           TF_uxb21_z,
                                           TF_uxb22_x,
                                           TF_uxb22_y,
                                           TF_uxb22_z};

static const std::vector<Profile> nonlocal_tfm_profiles{PROFILE_Umean_x,
                                                        PROFILE_Umean_y,
                                                        PROFILE_Umean_z,
                                                        PROFILE_ucrossb11mean_x,
                                                        PROFILE_ucrossb11mean_y,
                                                        PROFILE_ucrossb11mean_z,
                                                        PROFILE_ucrossb12mean_x,
                                                        PROFILE_ucrossb12mean_y,
                                                        PROFILE_ucrossb12mean_z,
                                                        PROFILE_ucrossb21mean_x,
                                                        PROFILE_ucrossb21mean_y,
                                                        PROFILE_ucrossb21mean_z,
                                                        PROFILE_ucrossb22mean_x,
                                                        PROFILE_ucrossb22mean_y,
                                                        PROFILE_ucrossb22mean_z};

// #define TFM_DEBUG_AVG_KERNEL
#if defined(TFM_DEBUG_AVG_KERNEL)
static const std::vector<std::vector<Kernel>> hydro_kernels{std::vector<Kernel>{average_hydro},
                                                            std::vector<Kernel>{average_hydro},
                                                            std::vector<Kernel>{average_hydro}};

static const std::vector<std::vector<Kernel>> tfm_kernels{std::vector<Kernel>{average_tfm},
                                                          std::vector<Kernel>{average_tfm},
                                                          std::vector<Kernel>{average_tfm}};
#else
static const std::vector<std::vector<Kernel>>
    hydro_kernels{std::vector<Kernel>{singlepass_solve_step0},
                  std::vector<Kernel>{singlepass_solve_step1},
                  std::vector<Kernel>{singlepass_solve_step2}};

static const std::vector<std::vector<Kernel>>
    tfm_kernels{std::vector<Kernel>{singlepass_solve_step0_tfm_b11,
                                    singlepass_solve_step0_tfm_b12,
                                    singlepass_solve_step0_tfm_b21,
                                    singlepass_solve_step0_tfm_b22},
                std::vector<Kernel>{singlepass_solve_step1_tfm_b11,
                                    singlepass_solve_step1_tfm_b12,
                                    singlepass_solve_step1_tfm_b21,
                                    singlepass_solve_step1_tfm_b22},
                std::vector<Kernel>{singlepass_solve_step2_tfm_b11,
                                    singlepass_solve_step2_tfm_b12,
                                    singlepass_solve_step2_tfm_b21,
                                    singlepass_solve_step2_tfm_b22}};
#endif

static const std::vector<Field> all_fields{VTXBUF_LNRHO, VTXBUF_UUX, VTXBUF_UUY, VTXBUF_UUZ,
                                           TF_a11_x,     TF_a11_y,   TF_a11_z,   TF_a12_x,
                                           TF_a12_y,     TF_a12_z,   TF_a21_x,   TF_a21_y,
                                           TF_a21_z,     TF_a22_x,   TF_a22_y,   TF_a22_z,
                                           TF_uxb11_x,   TF_uxb11_y, TF_uxb11_z, TF_uxb12_x,
                                           TF_uxb12_y,   TF_uxb12_z, TF_uxb21_x, TF_uxb21_y,
                                           TF_uxb21_z,   TF_uxb22_x, TF_uxb22_y, TF_uxb22_z};
#endif
