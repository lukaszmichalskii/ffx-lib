#pragma once

#include "ffx/nn/element_wise.h"
#include "ffx/nn/functional/add.h"

namespace ffx::nn {

  template <std::size_t Size>
  using Add = ElementWise<Size, functional::Add>;

}  // namespace ffx::nn
