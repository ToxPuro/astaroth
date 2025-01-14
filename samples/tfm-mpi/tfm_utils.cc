#include "tfm_utils_cc.h"

#include "acm/detail/errchk.h"

/**
 * Returns a vector of field handles in a field group.
 * For example: get_field_handles(FieldGroup::Hydro)
 * returns the fields required for computing the hydro step.
 */
std::vector<VertexBufferHandle>
get_field_handles(const FieldGroup& group)
{
    switch (group) {
    case FieldGroup::Hydro:
        return std::vector<VertexBufferHandle>{VTXBUF_LNRHO, VTXBUF_UUX, VTXBUF_UUY, VTXBUF_UUZ};
    case FieldGroup::TFM:
        return std::vector<VertexBufferHandle>{TF_a11_x,
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
    case FieldGroup::Bfield:
        return std::vector<VertexBufferHandle>{TF_uxb11_x,
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
    default:
        ERRCHK(false);
        return std::vector<VertexBufferHandle>{};
    }
}

std::vector<Profile>
get_profile_handles(const ProfileGroup& group)
{
    switch (group) {
    case ProfileGroup::TFM_Nonlocal:
        return std::vector<Profile>{PROFILE_Umean_x,
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
    default:
        ERRCHK(false);
        return std::vector<Profile>{};
    }
}

std::vector<Kernel>
get_kernels(const FieldGroup group, const int step)
{
    switch (group) {
    case FieldGroup::Hydro: {
        switch (step) {
        case 0:
            return std::vector<Kernel>{singlepass_solve_step0};
        case 1:
            return std::vector<Kernel>{singlepass_solve_step1};
        case 2:
            return std::vector<Kernel>{singlepass_solve_step2};
        default:
            ERRCHK(false);
            return std::vector<Kernel>{};
        }
    }
    case FieldGroup::TFM: {
        switch (step) {
        case 0:
            return std::vector<Kernel>{singlepass_solve_step0_tfm_b11,
                                       singlepass_solve_step0_tfm_b12,
                                       singlepass_solve_step0_tfm_b21,
                                       singlepass_solve_step0_tfm_b22};
        case 1:
            return std::vector<Kernel>{singlepass_solve_step1_tfm_b11,
                                       singlepass_solve_step1_tfm_b12,
                                       singlepass_solve_step1_tfm_b21,
                                       singlepass_solve_step1_tfm_b22};
        case 2:
            return std::vector<Kernel>{singlepass_solve_step2_tfm_b11,
                                       singlepass_solve_step2_tfm_b12,
                                       singlepass_solve_step2_tfm_b21,
                                       singlepass_solve_step2_tfm_b22};
        default:
            ERRCHK(false);
            return std::vector<Kernel>{};
        }
    }
    default:
        ERRCHK(false);
        return std::vector<Kernel>{};
    }
    ERRCHK(false);
    return std::vector<Kernel>{};
}
