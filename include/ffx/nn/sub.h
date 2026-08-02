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
  using Sub = ElementWise<Size, functional::Sub>;
  
  template <std::size_t Size>
  using SubReLU = ElementWise<Size, functional::Sub, functional::ReLU>;
  
  template <std::size_t Size>
  using SubReLU6 = ElementWise<Size, functional::Sub, functional::ReLU6>;

  template <std::size_t Size, std::int64_t FnNominator = 1, std::int64_t FnDenominator = 100>
  using SubLeakyReLU = ElementWise<Size, functional::Sub, functional::LeakyReLU<FnNominator, FnDenominator>>;

  template <std::size_t Size>
  using SubGELU = ElementWise<Size, functional::Sub, functional::GELU>;
  
  template <std::size_t Size>
  using SubHardswish = ElementWise<Size, functional::Sub, functional::Hardswish>;
  
  template <std::size_t Size>
  using SubSigmoid = ElementWise<Size, functional::Sub, functional::Sigmoid>;
  
  template <std::size_t Size>
  using SubSiLU = ElementWise<Size, functional::Sub, functional::SiLU>;
  
  template <std::size_t Size>
  using SubTanh = ElementWise<Size, functional::Sub, functional::Tanh>;

}  // namespace ffx::nn
