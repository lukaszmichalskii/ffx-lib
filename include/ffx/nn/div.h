#pragma once

#include "ffx/nn/element_wise.h"

#include "ffx/nn/functional/gelu.h"
#include "ffx/nn/functional/hardswish.h"
#include "ffx/nn/functional/identity.h"
#include "ffx/nn/functional/leaky_relu.h"
#include "ffx/nn/functional/relu.h"
#include "ffx/nn/functional/relu6.h"
#include "ffx/nn/functional/sigmoid.h"
#include "ffx/nn/functional/silu.h"
#include "ffx/nn/functional/tanh.h"

namespace ffx::nn {

  template <std::size_t Size>
  using Div = ElementWise<Size, functional::Div>;
  
  template <std::size_t Size>
  using DivReLU = ElementWise<Size, functional::Div, functional::ReLU>;
  
  template <std::size_t Size>
  using DivReLU6 = ElementWise<Size, functional::Div, functional::ReLU6>;

  template <std::size_t Size, std::int64_t FnNominator = 1, std::int64_t FnDenominator = 100>
  using DivLeakyReLU = ElementWise<Size, functional::Div, functional::LeakyReLU<FnNominator, FnDenominator>>;

  template <std::size_t Size>
  using DivGELU = ElementWise<Size, functional::Div, functional::GELU>;
  
  template <std::size_t Size>
  using DivHardswish = ElementWise<Size, functional::Div, functional::Hardswish>;
  
  template <std::size_t Size>
  using DivSigmoid = ElementWise<Size, functional::Div, functional::Sigmoid>;
  
  template <std::size_t Size>
  using DivSiLU = ElementWise<Size, functional::Div, functional::SiLU>;
  
  template <std::size_t Size>
  using DivTanh = ElementWise<Size, functional::Div, functional::Tanh>;

}  // namespace ffx::nn
