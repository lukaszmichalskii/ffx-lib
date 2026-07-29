#pragma once

#include <alpaka/alpaka.hpp>
#include "ffx/core/detail/concepts.h"

namespace ffx::nn::functional {

  struct SiLU {
    template <concepts::accelerator TAcc, typename T>
    ALPAKA_FN_ACC static auto forward(const TAcc& acc, T val) -> T {
      return val / (static_cast<T>(1) + alpaka::math::exp(acc, -val));
    }
  };

}  // namespace ffx::nn::functional
