#pragma once

#include <alpaka/alpaka.hpp>
#include "ffx/core/detail/concepts.h"

namespace ffx::nn::functional {

  struct ReLU {
    template <concepts::accelerator TAcc, typename T>
    ALPAKA_FN_ACC static auto forward(const TAcc& acc, T val) -> T {
      return alpaka::math::max(acc, static_cast<T>(0), val);
    }
  };

}  // namespace ffx::nn::functional
