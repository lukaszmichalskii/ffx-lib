#pragma once

#include <alpaka/alpaka.hpp>
#include <cstddef>

#include "ffx/core/detail/concepts.h"

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
  template <std::size_t BatchSize,
            std::size_t InHeight,
            std::size_t InWidth,
            std::size_t InChannels,
            std::size_t OutHeight,
            std::size_t OutWidth,
            typename TOperator = functional::Identity>
  struct AdaptiveAvgPool2dImpl {
    static constexpr std::size_t NumberOfElements = BatchSize * InChannels * OutHeight * OutWidth;

    template <concepts::accelerator TAcc, typename T>
    ALPAKA_FN_ACC void operator()(const TAcc& acc, const T* input, T* output) const {
      for (const auto thread_index : alpaka::uniformElements(acc, NumberOfElements)) {
        const auto w_out_idx = thread_index % OutWidth;
        auto residual = thread_index / OutWidth;
        const auto h_out_idx = residual % OutHeight;
        residual /= OutHeight;
        const auto c_idx = residual % InChannels;
        const auto batch_idx = residual / InChannels;

        const std::size_t h_start = (h_out_idx * InHeight) / OutHeight;
        const std::size_t w_start = (w_out_idx * InWidth) / OutWidth;
        const std::size_t h_end = ((h_out_idx + 1) * InHeight + OutHeight - 1) / OutHeight;
        const std::size_t w_end = ((w_out_idx + 1) * InWidth + OutWidth - 1) / OutWidth;

        const std::size_t pool_size = (h_end - h_start) * (w_end - w_start);

        T sum_val = static_cast<T>(0);
        // sum reduction over the window bounds
        for (std::size_t h_in_idx = h_start; h_in_idx < h_end; ++h_in_idx) {
          for (std::size_t w_in_idx = w_start; w_in_idx < w_end; ++w_in_idx) {
            const std::size_t input_flat_offset = batch_idx * InChannels * InHeight * InWidth +
                                                  c_idx * InHeight * InWidth + h_in_idx * InWidth + w_in_idx;

            sum_val += input[input_flat_offset];
          }
        }

        // safeguard against a zero-sized window (though mathematically impossible with valid shapes)
        T avg_val = (pool_size > 0) ? (sum_val / static_cast<T>(pool_size)) : static_cast<T>(0);

        output[thread_index] = TOperator::forward(acc, avg_val);
      }
    }
  };

  // AdaptiveAvgPool2d
  template <std::size_t BatchSize,
            std::size_t InHeight,
            std::size_t InWidth,
            std::size_t InChannels,
            std::size_t OutHeight,
            std::size_t OutWidth>
  using AdaptiveAvgPool2d = AdaptiveAvgPool2dImpl<BatchSize, InHeight, InWidth, InChannels, OutHeight, OutWidth>;

  // Fused AdaptiveAvgPool2d + ReLU
  template <std::size_t BatchSize,
            std::size_t InHeight,
            std::size_t InWidth,
            std::size_t InChannels,
            std::size_t OutHeight,
            std::size_t OutWidth>
  using AdaptiveAvgPool2dReLU =
      AdaptiveAvgPool2dImpl<BatchSize, InHeight, InWidth, InChannels, OutHeight, OutWidth, functional::ReLU>;

  // Fused AdaptiveAvgPool2d + LeakyReLU
  template <std::size_t BatchSize,
            std::size_t InHeight,
            std::size_t InWidth,
            std::size_t InChannels,
            std::size_t OutHeight,
            std::size_t OutWidth,
            std::int64_t FnNominator = 1,
            std::int64_t FnDenominator = 100>
  using AdaptiveAvgPool2dLeakyReLU = AdaptiveAvgPool2dImpl<BatchSize,
                                                           InHeight,
                                                           InWidth,
                                                           InChannels,
                                                           OutHeight,
                                                           OutWidth,
                                                           functional::LeakyReLU<FnNominator, FnDenominator>>;

  // Fused AdaptiveAvgPool2d + Sigmoid
  template <std::size_t BatchSize,
            std::size_t InHeight,
            std::size_t InWidth,
            std::size_t InChannels,
            std::size_t OutHeight,
            std::size_t OutWidth>
  using AdaptiveAvgPool2dSigmoid =
      AdaptiveAvgPool2dImpl<BatchSize, InHeight, InWidth, InChannels, OutHeight, OutWidth, functional::Sigmoid>;

  // Fused AdaptiveAvgPool2d + Tanh
  template <std::size_t BatchSize,
            std::size_t InHeight,
            std::size_t InWidth,
            std::size_t InChannels,
            std::size_t OutHeight,
            std::size_t OutWidth>
  using AdaptiveAvgPool2dTanh =
      AdaptiveAvgPool2dImpl<BatchSize, InHeight, InWidth, InChannels, OutHeight, OutWidth, functional::Tanh>;

  // Fused AdaptiveAvgPool2d + SiLU
  template <std::size_t BatchSize,
            std::size_t InHeight,
            std::size_t InWidth,
            std::size_t InChannels,
            std::size_t OutHeight,
            std::size_t OutWidth>
  using AdaptiveAvgPool2dSiLU =
      AdaptiveAvgPool2dImpl<BatchSize, InHeight, InWidth, InChannels, OutHeight, OutWidth, functional::SiLU>;

  // Fused AdaptiveAvgPool2d + GELU
  template <std::size_t BatchSize,
            std::size_t InHeight,
            std::size_t InWidth,
            std::size_t InChannels,
            std::size_t OutHeight,
            std::size_t OutWidth>
  using AdaptiveAvgPool2dGELU =
      AdaptiveAvgPool2dImpl<BatchSize, InHeight, InWidth, InChannels, OutHeight, OutWidth, functional::GELU>;

  // Fused AdaptiveAvgPool2d + ReLU6
  template <std::size_t BatchSize,
            std::size_t InHeight,
            std::size_t InWidth,
            std::size_t InChannels,
            std::size_t OutHeight,
            std::size_t OutWidth>
  using AdaptiveAvgPool2dReLU6 =
      AdaptiveAvgPool2dImpl<BatchSize, InHeight, InWidth, InChannels, OutHeight, OutWidth, functional::ReLU6>;

  // Fused AdaptiveAvgPool2d + Hardswish
  template <std::size_t BatchSize,
            std::size_t InHeight,
            std::size_t InWidth,
            std::size_t InChannels,
            std::size_t OutHeight,
            std::size_t OutWidth>
  using AdaptiveAvgPool2dHardswish =
      AdaptiveAvgPool2dImpl<BatchSize, InHeight, InWidth, InChannels, OutHeight, OutWidth, functional::Hardswish>;

}  // namespace ffx::nn
