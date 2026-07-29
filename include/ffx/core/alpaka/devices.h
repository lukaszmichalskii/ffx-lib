#pragma once

#include <alpaka/alpaka.hpp>

#include "ffx/core/alpaka/platform.h"
#include "ffx/core/detail/concepts.h"

namespace ffx {

  template <concepts::platform TPlatform>
  std::vector<alpaka::Dev<TPlatform>> const& devices() {
    static const auto devices = alpaka::getDevs(platform<TPlatform>());
    return devices;
  }

}  // namespace ffx
