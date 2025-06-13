#pragma once

#include "acc_runtime.h"
#include "acm/detail/datatypes.h"
#include "acm/detail/type_conversion.h"

#define ERRCHK_AC(errcode)                                                                         \
    do {                                                                                           \
        const AcResult _tmp_ac_api_errcode_ = (errcode);                                           \
        if (_tmp_ac_api_errcode_ != AC_SUCCESS) {                                                  \
            errchk_print_error(__func__, __FILE__, __LINE__, #errcode, "Astaroth error");          \
            errchk_print_stacktrace();                                                             \
            throw std::runtime_error("Assertion " #errcode " failed");                             \
        }                                                                                          \
    } while (0)

#define WARNCHK_AC(errcode)                                                                         \
    do {                                                                                           \
        const AcResult _tmp_ac_api_errcode_ = (errcode);                                           \
        if (_tmp_ac_api_errcode_ != AC_SUCCESS) {                                                  \
            errchk_print_warning(__func__, __FILE__, __LINE__, #errcode, "Astaroth error");          \
        }                                                                                          \
    } while (0)

enum class BufferGroup { input, output };

namespace acr {

int get(const AcMeshInfo& info, const AcIntParam& param);
int3 get(const AcMeshInfo& info, const AcInt3Param& param);
AcReal get(const AcMeshInfo& info, const AcRealParam& param);
AcReal3 get(const AcMeshInfo& info, const AcReal3Param& param);

// std::vector<ac::device_view<AcReal>>
// get(const VertexBufferArray& vba, const std::vector<Field>& fields, const BufferGroup& group);

void set(const AcIntParam& param, const int value, AcMeshInfo& info);
void set(const AcInt3Param& param, const int3& value, AcMeshInfo& info);
void set(const AcRealParam& param, const AcReal value, AcMeshInfo& info);
void set(const AcReal3Param& param, const AcReal3& value, AcMeshInfo& info);

ac::shape get_global_nn(const AcMeshInfo& info);
ac::shape get_local_nn(const AcMeshInfo& info);
ac::shape get_local_mm(const AcMeshInfo& info);

void set_global_nn(const ac::shape& global_nn, AcMeshInfo& info);
void set_local_nn(const ac::shape& local_nn, AcMeshInfo& info);
void set_local_ss(const ac::shape& local_ss, AcMeshInfo& info);

Dims get_global_ss(const AcMeshInfo& info);

ac::index get_global_nn_offset(const AcMeshInfo& info);
ac::index get_local_nn_offset();
ac::index get_local_rr();

ac::device_view<AcReal> make_ptr(const VertexBufferArray& vba, const Field& field,
                                        const BufferGroup& type);
ac::device_view<AcReal> make_ptr(const VertexBufferArray& vba, const Profile& profile,
                                        const BufferGroup& type);


std::vector<ac::device_view<AcReal>>
get_ptrs(const VertexBufferArray& vba, const std::vector<Field>& fields, const BufferGroup& type);

std::vector<ac::device_view<AcReal>> get_ptrs(const VertexBufferArray& vba,
                                                     const std::vector<Profile>& profiles,
                                                     const BufferGroup& type);

/** Returns a vector of field names corresponding to the input fields */
std::vector<std::string> get_strs(const std::vector<Field>& fields);

} // namespace acr
