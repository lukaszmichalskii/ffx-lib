#pragma once

#include "ffx/nn/element_wise.h"
#include "ffx/nn/functional/sigmoid.h"

namespace ffx::nn {

  template <std::size_t Size>
  using Sigmoid = ElementWise<Size, functional::Sigmoid>;

}  // namespace ffx::nn
