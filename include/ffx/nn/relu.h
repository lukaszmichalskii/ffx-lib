#pragma once

#include "ffx/nn/element_wise.h"
#include "ffx/nn/functional/relu.h"

namespace ffx::nn {

  template <std::size_t Size>
  using ReLU = ElementWise<Size, functional::ReLU>;

}  // namespace ffx::nn
