#pragma once

#include "ffx/nn/element_wise.h"
#include "ffx/nn/functional/leaky_relu.h"

namespace ffx::nn {

  template <std::size_t Size, std::int64_t Nominator = 1, std::int64_t Denominator = 100>
  using LeakyReLU = ElementWise<Size, functional::LeakyReLU<Nominator, Denominator>>;

}  // namespace ffx::nn
