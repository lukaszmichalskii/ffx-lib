#pragma once

#include "ffx/core/alpaka/config.h"
#include "ffx/core/detail/concepts.h"

namespace ffx {

  using namespace ffx_alpaka;

  template <concepts::platform TPlatform>
  TPlatform const& platform() {
    static const auto platform = TPlatform{};
    return platform;
  }

  inline PlatformHost const& host_platform() { return platform<PlatformHost>(); }

}  // namespace ffx
