#pragma once

#include <memory>
#include <functional>

#include "acr_utils.h"


namespace acr {

// class mesh_info {
//   private:
//     AcMeshInfo info{};
//   public:
//     mesh_info() = default;
    
//     template<typename T, typename U>
//     auto get(const T& param) {
//       return
//     }
// };

class device {
  private:
    std::unique_ptr<Device, std::function<void(Device*)>> m_device;

    // AcMeshInfo mesh_info() const {
    //   AcMeshInfo info{};
    //   ERRCHK_AC(acDeviceGetLocalConfig(*m_device, &info));
    //   return info;
    // }

    // VertexBufferArray vba() const {
    //   VertexBufferArray vba{};
    //   ERRCHK_AC(acDeviceGetVBA(*m_device, &vba));
    //   return vba;
    // }

  public:
    device(const int id, const AcMeshInfo& info)
        : m_device{[&]() {
                       auto ptr{new Device{nullptr}};
                       ERRCHK_AC(acDeviceCreate(id, info, ptr));
                       return ptr;
                   }(),
                   [](Device* ptr) noexcept {
                       WARNCHK(*ptr != nullptr);
                       WARNCHK_AC(acDeviceDestroy(*ptr));
                       delete ptr;
                   }}
    {
    }

    Device get() const noexcept { return *m_device; }

    // template<typename T>
    // auto get_param(const T& param) {
    //   return acr::get(mesh_info(), param);
    // }

    // template<typename T, typename U>
    // void set_param(const T& param, const U& value) {
    //   auto info{mesh_info()};
    //   acr::set(param, value, info);
    //   ERRCHK_AC(acDeviceLoadMeshInfo(*m_device, info));
    // }

    // ac::device_view<AcReal> get_dfield(const Field& field, const BufferGroup& group) {
    //   return acr::make_ptr(vba(), field, group);
    // }
};

// VertexBufferArray get_vba(const Device& device) {
//   VertexBufferArray vba{};
//   ERRCHK_AC(acDeviceGetVBA(device, &vba));
//   return vba;
// }

// MeshInfo get_mesh_info(const Device& device) {
//   MeshInfo info{};
//   ERRCHK_AC(acDeviceGetMeshInfo(device, &info));
//   return info;
// }

} // namespace acr
