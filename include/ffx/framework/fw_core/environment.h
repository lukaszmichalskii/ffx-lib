#pragma once

#include <print>
#include <alpaka/alpaka.hpp>

#include "ffx/core/ffx_core.h"

namespace ffx::framework {

  template <concepts::platform TPlatform>
  void get_environment_info() {
    const DevHost& host = ffx::host();

    std::println("Host platform: {}", alpaka::core::demangled<PlatformHost>);
    std::println("Found 1 device:");
    std::println("  - {}\n", alpaka::getName(host));

    const auto& devices = ffx::devices<TPlatform>();
    std::println("Accelerator platform: {}", alpaka::core::demangled<TPlatform>);
    std::println("Found {} device(s):", devices.size());
    for (const auto& dev : devices) {
      std::println("  - {}", alpaka::getName(dev));
    }
    std::println();
  }

}  // namespace ffx::framework