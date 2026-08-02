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
  using Add = ElementWise<Size, functional::Add>;

  template <std::size_t Size>
  using AddReLU = ElementWise<Size, functional::Add, functional::ReLU>;

  template <std::size_t Size>
  using AddReLU6 = ElementWise<Size, functional::Add, functional::ReLU6>;

  template <std::size_t Size, std::int64_t FnNominator = 1, std::int64_t FnDenominator = 100>
  using AddLeakyReLU = ElementWise<Size, functional::Add, functional::LeakyReLU<FnNominator, FnDenominator>>;

  template <std::size_t Size>
  using AddGELU = ElementWise<Size, functional::Add, functional::GELU>;

  template <std::size_t Size>
  using AddHardswish = ElementWise<Size, functional::Add, functional::Hardswish>;

  template <std::size_t Size>
  using AddSigmoid = ElementWise<Size, functional::Add, functional::Sigmoid>;

  template <std::size_t Size>
  using AddSiLU = ElementWise<Size, functional::Add, functional::SiLU>;

  template <std::size_t Size>
  using AddTanh = ElementWise<Size, functional::Add, functional::Tanh>;

}  // namespace ffx::nn
