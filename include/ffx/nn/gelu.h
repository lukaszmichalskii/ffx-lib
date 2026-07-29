#pragma once

#include "ffx/nn/element_wise.h"
#include "ffx/nn/functional/gelu.h"

namespace ffx::nn {

  template <std::size_t Size>
  using GELU = ElementWise<Size, functional::GELU>;

}  // namespace ffx::nn
