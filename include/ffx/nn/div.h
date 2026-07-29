#pragma once

#include "ffx/nn/element_wise.h"
#include "ffx/nn/functional/div.h"

namespace ffx::nn {

  template <std::size_t Size>
  using Div = ElementWise<Size, functional::Div>;

}  // namespace ffx::nn
