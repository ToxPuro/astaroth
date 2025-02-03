#pragma once

#include "acc_runtime.h"
#include "acm/detail/datatypes.h"

#define ERRCHK_AC(errcode)                                                                         \
    do {                                                                                           \
        const AcResult _tmp_ac_api_errcode_ = (errcode);                                           \
        if (_tmp_ac_api_errcode_ != AC_SUCCESS) {                                                  \
            errchk_print_error(__func__, __FILE__, __LINE__, #errcode, "Astaroth error");          \
            errchk_print_stacktrace();                                                             \
            MPI_Abort(MPI_COMM_WORLD, -1);                                                         \
        }                                                                                          \
    } while (0)

namespace acr {

int get(const AcMeshInfo& info, const AcIntParam& param);
int3 get(const AcMeshInfo& info, const AcInt3Param& param);
AcReal get(const AcMeshInfo& info, const AcRealParam& param);
AcReal3 get(const AcMeshInfo& info, const AcReal3Param& param);

// std::vector<ac::mr::device_pointer<AcReal>>
// get(const VertexBufferArray& vba, const std::vector<Field>& fields, const BufferGroup& group);

void set(const AcIntParam& param, const int value, AcMeshInfo& info);
void set(const AcInt3Param& param, const int3& value, AcMeshInfo& info);
void set(const AcRealParam& param, const AcReal value, AcMeshInfo& info);
void set(const AcReal3Param& param, const AcReal3& value, AcMeshInfo& info);

Shape get_global_nn(const AcMeshInfo& info);
Shape get_local_nn(const AcMeshInfo& info);
Shape get_local_mm(const AcMeshInfo& info);

Dims get_global_ss(const AcMeshInfo& info);

Index get_global_nn_offset(const AcMeshInfo& info);
Index get_local_nn_offset();
Index get_local_rr();

} // namespace acr

inline int3
convert_to_int3(const ac::vector<uint64_t>& in)
{
    ERRCHK(in.size() == 3);
    return int3{as<int>(in[0]), as<int>(in[1]), as<int>(in[2])};
}
