#pragma once
#include "acc_runtime.h"

#include <vector>

enum class FieldGroup { Hydro, TFM, Bfield };
enum class ProfileGroup { TFM_Nonlocal };
enum class BufferGroup { Input, Output };
enum class SegmentGroup { Inner, Outer, Full };

/**
 * Returns a vector of field handles in a field group.
 * For example: get_field_handles(FieldGroup::Hydro)
 * returns the fields required for computing the hydro step.
 */
std::vector<VertexBufferHandle> get_field_handles(const FieldGroup& group);

/**
 * Returns a vector of profile handles in a profile group.
 * For example: get_profile_handles(ProfileGroup::TFM_Nonlocal).
 */
std::vector<Profile> get_profile_handles(const ProfileGroup& group);

/** Returns a vector of kernels used to update the fields in a field group */
std::vector<Kernel> get_kernels(const FieldGroup group, const size_t step);
