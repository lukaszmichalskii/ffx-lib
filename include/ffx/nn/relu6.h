#pragma once

#include "ffx/nn/element_wise.h"
#include "ffx/nn/functional/relu6.h"

namespace ffx::nn {

  template <std::size_t Size>
  using ReLU6 = ElementWise<Size, functional::ReLU6>;

}  // namespace ffx::nn
