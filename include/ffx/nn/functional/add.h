#pragma once

#include <alpaka/alpaka.hpp>
#include "ffx/core/detail/concepts.h"

namespace ffx::nn::functional {

  struct Add {
    template <concepts::accelerator TAcc, typename T>
    ALPAKA_FN_ACC static auto forward(const TAcc&, const T a, const T b) -> T {
      return a + b;
    }
  };

}  // namespace ffx::nn::functional
