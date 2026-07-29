#pragma once

#include <alpaka/alpaka.hpp>

#include "ffx/core/alpaka/config.h"

namespace ffx {

  using namespace ffx_alpaka;

  inline DevHost const& host() {
    static const auto host = alpaka::getDevByIdx(alpaka::PlatformCpu{}, 0u);
    return host;
  }

}  // namespace ffx
