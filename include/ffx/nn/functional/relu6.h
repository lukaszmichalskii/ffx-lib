#pragma once

#include <alpaka/alpaka.hpp>
#include "ffx/core/detail/concepts.h"
#include "ffx/nn/functional/relu.h"

namespace ffx::nn::functional {

  struct ReLU6 {
    template <concepts::accelerator TAcc, typename T>
    ALPAKA_FN_ACC static auto forward(const TAcc& acc, T val) -> T {
      return alpaka::math::min(acc, static_cast<T>(6), ReLU::forward(acc, val));
    }
  };

}  // namespace ffx::nn::functional
