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
                   [](Device* ptr) {
                       ERRCHK(*ptr != nullptr);
                       ERRCHK_AC(acDeviceDestroy(*ptr));
                       delete ptr;
                   }}
    {
    }
};

} // namespace acr
