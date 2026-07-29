#pragma once

#include "ffx/nn/element_wise.h"
#include "ffx/nn/functional/silu.h"

namespace ffx::nn {

  template <std::size_t Size>
  using SiLU = ElementWise<Size, functional::SiLU>;

}  // namespace ffx::nn
