#pragma once

#include <alpaka/alpaka.hpp>
#include "ffx/core/detail/concepts.h"

namespace ffx::nn::functional {

  struct GELU {
    template <concepts::accelerator TAcc, typename T>
    ALPAKA_FN_ACC static auto forward(const TAcc& acc, T val) -> T {
      constexpr auto sqrt_2_pi = static_cast<T>(0.7978845608);
      constexpr auto c1 = static_cast<T>(0.044715);
      const auto inner = sqrt_2_pi * (val + c1 * val * val * val);
      return static_cast<T>(0.5) * val * (static_cast<T>(1) + alpaka::math::tanh(acc, inner));
    }
  };

}  // namespace ffx::nn::functional
