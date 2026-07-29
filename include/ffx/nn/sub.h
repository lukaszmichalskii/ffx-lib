#pragma once

#include "ffx/nn/element_wise.h"
#include "ffx/nn/functional/sub.h"

namespace ffx::nn {

  template <std::size_t Size>
  using Sub = ElementWise<Size, functional::Sub>;

}  // namespace ffx::nn
