#pragma once

#include <memory>
#include <functional>

#include "acr_utils.h"

namespace acr {

class device {
  private:
    std::unique_ptr<Device, std::function<void(Device*)>> m_device;

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

};

VertexBufferArray get_vba(const Device& device) {
  VertexBufferArray vba{};
  ERRCHK_AC(acDeviceGetVBA(device, &vba));
  return vba;
}

MeshInfo get_mesh_info(const Device& device) {
  MeshInfo info{};
  ERRCHK_AC(acDeviceGetMeshInfo(device, &info));
  return info;
}

} // namespace acr
