#pragma once

#include "ffx/nn/element_wise.h"
#include "ffx/nn/functional/tanh.h"

namespace ffx::nn {

  template <std::size_t Size>
  using Tanh = ElementWise<Size, functional::Tanh>;

}  // namespace ffx::nn
