#pragma once

#include "acc_runtime.h"
#include "acm/detail/datatypes.h"

namespace acr {

auto get(const AcMeshInfo& info, const AcIntParam& param);
auto get(const AcMeshInfo& info, const AcInt3Param& param);
auto get(const AcMeshInfo& info, const AcRealParam& param);
auto get(const AcMeshInfo& info, const AcReal3Param& param);

void set(const AcIntParam& param, const int value, AcMeshInfo& info);
void set(const AcInt3Param& param, const int3& value, AcMeshInfo& info);
void set(const AcRealParam& param, const AcReal value, AcMeshInfo& info);
void set(const AcReal3Param& param, const AcReal3& value, AcMeshInfo& info);

Shape get_global_nn(const AcMeshInfo& info);

Dims get_global_ss(const AcMeshInfo& info);

Index get_local_nn_offset();
Index get_local_rr();
Index get_global_nn_offset(const AcMeshInfo& info);
Shape get_local_nn(const AcMeshInfo& info);
Shape get_local_mm(const AcMeshInfo& info);

} // namespace acr
