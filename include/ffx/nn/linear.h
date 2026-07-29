#pragma once

#include <alpaka/alpaka.hpp>

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

  template <std::size_t NumRows,  // batch_size * sequence_length
            std::size_t InDim,    // inner reduction dim
            std::size_t OutDim,
            std::size_t InRowStride,
            std::size_t InColStride,
            std::size_t WeightRowStride,
            std::size_t WeightColStride,
            typename TOperator = functional::Identity>
  struct LinearImpl {
    template <concepts::accelerator TAcc, typename T>
    ALPAKA_FN_ACC void operator()(const TAcc& acc, const T* input, T* output, const T* weights, const T* bias) const {
      for (const auto thread_index : alpaka::uniformElements(acc, NumRows * OutDim)) {
        const auto r = thread_index / OutDim;
        const auto c = thread_index % OutDim;

        auto accum = bias ? bias[c] : static_cast<T>(0);
        // dot-product
        for (auto k = 0u; k < InDim; ++k) {
          const auto input_idx = r * InRowStride + k * InColStride;
          const auto weight_idx = c * WeightRowStride + k * WeightColStride;
          accum += input[input_idx] * weights[weight_idx];
        }
        output[thread_index] = TOperator::forward(acc, accum);
      }
    }
  };

  // Vanilla Linear
  template <std::size_t BatchSize, std::size_t InDim, std::size_t OutDim, typename TOperator = functional::Identity>
  using ContiguousLinear = LinearImpl<BatchSize, InDim, OutDim, InDim, 1, InDim, 1, TOperator>;

  // Linear
  template <std::size_t BatchSize, std::size_t InDim, std::size_t OutDim>
  using Linear = ContiguousLinear<BatchSize, InDim, OutDim>;

  // Fused Linear + ReLU
  template <std::size_t BatchSize, std::size_t InDim, std::size_t OutDim>
  using LinearReLU = ContiguousLinear<BatchSize, InDim, OutDim, functional::ReLU>;

  // Fused Linear + LeakyReLU
  template <std::size_t BatchSize,
            std::size_t InDim,
            std::size_t OutDim,
            std::int64_t FnNominator = 1,
            std::int64_t FnDenominator = 100>
  using LinearLeakyReLU = ContiguousLinear<BatchSize, InDim, OutDim, functional::LeakyReLU<FnNominator, FnDenominator>>;

  // Fused Linear + Sigmoid
  template <std::size_t BatchSize, std::size_t InDim, std::size_t OutDim>
  using LinearSigmoid = ContiguousLinear<BatchSize, InDim, OutDim, functional::Sigmoid>;

  // Fused Linear + Tanh
  template <std::size_t BatchSize, std::size_t InDim, std::size_t OutDim>
  using LinearTanh = ContiguousLinear<BatchSize, InDim, OutDim, functional::Tanh>;

  // Fused Linear + SiLU
  template <std::size_t BatchSize, std::size_t InDim, std::size_t OutDim>
  using LinearSiLU = ContiguousLinear<BatchSize, InDim, OutDim, functional::SiLU>;

  // Fused Linear + GELU
  template <std::size_t BatchSize, std::size_t InDim, std::size_t OutDim>
  using LinearGELU = ContiguousLinear<BatchSize, InDim, OutDim, functional::GELU>;

  // Fused Linear + ReLU6
  template <std::size_t BatchSize, std::size_t InDim, std::size_t OutDim>
  using LinearReLU6 = ContiguousLinear<BatchSize, InDim, OutDim, functional::ReLU6>;

  // Fused Linear + Hardswish
  template <std::size_t BatchSize, std::size_t InDim, std::size_t OutDim>
  using LinearHardswish = ContiguousLinear<BatchSize, InDim, OutDim, functional::Hardswish>;

}  // namespace ffx::nn
