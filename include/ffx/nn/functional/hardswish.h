#pragma once

#include "ffx/core/detail/concepts.h"
#include "ffx/nn/functional/relu6.h"

namespace ffx::nn::functional {

  struct Hardswish {
    template <concepts::accelerator TAcc, typename T>
    ALPAKA_FN_ACC static auto forward(const TAcc& acc, T val) -> T {
      return val * (ReLU6::forward(acc, val + static_cast<T>(3)) / static_cast<T>(6));
    }
  };

}  // namespace ffx::nn::functional
