#pragma once

#include "ffx/core/detail/concepts.h"

namespace ffx::nn::functional {

  struct Identity {
    template <concepts::accelerator TAcc, typename T>
    ALPAKA_FN_ACC static auto forward(const TAcc&, T val) -> T {
      return val;
    }
  };

}  // namespace ffx::nn::functional
