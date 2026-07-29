#pragma once

#include <alpaka/alpaka.hpp>
#include "ffx/core/detail/concepts.h"
#include "ffx/nn/functional/relu.h"

namespace ffx::nn::functional {

  template <std::int64_t Numerator = 1, std::int64_t Denominator = 100>
  struct LeakyReLU {
    template <concepts::accelerator TAcc, typename T>
    ALPAKA_FN_ACC static auto forward(const TAcc& acc, T val) -> T {
      constexpr T kNegativeSlope = static_cast<T>(Numerator) / static_cast<T>(Denominator);
      return ReLU::forward(acc, val) + kNegativeSlope * alpaka::math::min(acc, static_cast<T>(0), val);
    }
  };

}  // namespace ffx::nn::functional
