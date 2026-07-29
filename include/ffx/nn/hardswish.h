#pragma once
#include "ffx/nn/element_wise.h"
#include "ffx/nn/functional/hardswish.h"

namespace ffx::nn {

  template <std::size_t Size>
  using Hardswish = ElementWise<Size, functional::Hardswish>;

}  // namespace ffx::nn
