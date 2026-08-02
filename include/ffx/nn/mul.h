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
  using Mul = ElementWise<Size, functional::Mul>;
  
  template <std::size_t Size>
  using MulReLU = ElementWise<Size, functional::Mul, functional::ReLU>;
  
  template <std::size_t Size>
  using MulReLU6 = ElementWise<Size, functional::Mul, functional::ReLU6>;

  template <std::size_t Size, std::int64_t FnNominator = 1, std::int64_t FnDenominator = 100>
  using MulLeakyReLU = ElementWise<Size, functional::Mul, functional::LeakyReLU<FnNominator, FnDenominator>>;

  template <std::size_t Size>
  using MulGELU = ElementWise<Size, functional::Mul, functional::GELU>;
  
  template <std::size_t Size>
  using MulHardswish = ElementWise<Size, functional::Mul, functional::Hardswish>;
  
  template <std::size_t Size>
  using MulSigmoid = ElementWise<Size, functional::Mul, functional::Sigmoid>;
  
  template <std::size_t Size>
  using MulSiLU = ElementWise<Size, functional::Mul, functional::SiLU>;
  
  template <std::size_t Size>
  using MulTanh = ElementWise<Size, functional::Mul, functional::Tanh>;

}  // namespace ffx::nn
