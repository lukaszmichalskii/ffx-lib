#pragma once

#include <alpaka/alpaka.hpp>
#include "ffx/core/detail/concepts.h"

namespace ffx::nn::functional {

  struct Tanh {
    template <concepts::accelerator TAcc, typename T>
    ALPAKA_FN_ACC static auto forward(const TAcc& acc, T val) -> T {
      return alpaka::math::tanh(acc, val);
    }
  };

}  // namespace ffx::nn::functional
