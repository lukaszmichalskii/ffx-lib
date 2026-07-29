#pragma once

#include "ffx/nn/element_wise.h"
#include "ffx/nn/functional/mul.h"

namespace ffx::nn {

  template <std::size_t Size>
  using Mul = ElementWise<Size, functional::Mul>;

}  // namespace ffx::nn
