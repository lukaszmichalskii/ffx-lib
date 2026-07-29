#pragma once

#include <alpaka/alpaka.hpp>
#include <limits>

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
  struct AdaptiveMaxPool2dImpl {
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

        T max_val = -std::numeric_limits<T>::max();

        // local max reduction over the dynamically calculated window bounds
        for (std::size_t h_in_idx = h_start; h_in_idx < h_end; ++h_in_idx) {
          for (std::size_t w_in_idx = w_start; w_in_idx < w_end; ++w_in_idx) {
            const std::size_t input_flat_offset = batch_idx * InChannels * InHeight * InWidth +
                                                  c_idx * InHeight * InWidth + h_in_idx * InWidth + w_in_idx;

            const T val = input[input_flat_offset];
            max_val = alpaka::math::max(acc, val, max_val);
          }
        }

        output[thread_index] = TOperator::forward(acc, max_val);
      }
    }
  };

  // AdaptiveMaxPool2d
  template <std::size_t BatchSize,
            std::size_t InHeight,
            std::size_t InWidth,
            std::size_t InChannels,
            std::size_t OutHeight,
            std::size_t OutWidth>
  using AdaptiveMaxPool2d = AdaptiveMaxPool2dImpl<BatchSize, InHeight, InWidth, InChannels, OutHeight, OutWidth>;

  // Fused AdaptiveMaxPool2d + ReLU
  template <std::size_t BatchSize,
            std::size_t InHeight,
            std::size_t InWidth,
            std::size_t InChannels,
            std::size_t OutHeight,
            std::size_t OutWidth>
  using AdaptiveMaxPool2dReLU =
      AdaptiveMaxPool2dImpl<BatchSize, InHeight, InWidth, InChannels, OutHeight, OutWidth, functional::ReLU>;

  // Fused AdaptiveMaxPool2d + LeakyReLU
  template <std::size_t BatchSize,
            std::size_t InHeight,
            std::size_t InWidth,
            std::size_t InChannels,
            std::size_t OutHeight,
            std::size_t OutWidth,
            std::int64_t FnNominator = 1,
            std::int64_t FnDenominator = 100>
  using AdaptiveMaxPool2dLeakyReLU = AdaptiveMaxPool2dImpl<BatchSize,
                                                           InHeight,
                                                           InWidth,
                                                           InChannels,
                                                           OutHeight,
                                                           OutWidth,
                                                           functional::LeakyReLU<FnNominator, FnDenominator>>;

  // Fused AdaptiveMaxPool2d + Sigmoid
  template <std::size_t BatchSize,
            std::size_t InHeight,
            std::size_t InWidth,
            std::size_t InChannels,
            std::size_t OutHeight,
            std::size_t OutWidth>
  using AdaptiveMaxPool2dSigmoid =
      AdaptiveMaxPool2dImpl<BatchSize, InHeight, InWidth, InChannels, OutHeight, OutWidth, functional::Sigmoid>;

  // Fused AdaptiveMaxPool2d + Tanh
  template <std::size_t BatchSize,
            std::size_t InHeight,
            std::size_t InWidth,
            std::size_t InChannels,
            std::size_t OutHeight,
            std::size_t OutWidth>
  using AdaptiveMaxPool2dTanh =
      AdaptiveMaxPool2dImpl<BatchSize, InHeight, InWidth, InChannels, OutHeight, OutWidth, functional::Tanh>;

  // Fused AdaptiveMaxPool2d + SiLU
  template <std::size_t BatchSize,
            std::size_t InHeight,
            std::size_t InWidth,
            std::size_t InChannels,
            std::size_t OutHeight,
            std::size_t OutWidth>
  using AdaptiveMaxPool2dSiLU =
      AdaptiveMaxPool2dImpl<BatchSize, InHeight, InWidth, InChannels, OutHeight, OutWidth, functional::SiLU>;

  // Fused AdaptiveMaxPool2d + GELU
  template <std::size_t BatchSize,
            std::size_t InHeight,
            std::size_t InWidth,
            std::size_t InChannels,
            std::size_t OutHeight,
            std::size_t OutWidth>
  using AdaptiveMaxPool2dGELU =
      AdaptiveMaxPool2dImpl<BatchSize, InHeight, InWidth, InChannels, OutHeight, OutWidth, functional::GELU>;

  // Fused AdaptiveMaxPool2d + ReLU6
  template <std::size_t BatchSize,
            std::size_t InHeight,
            std::size_t InWidth,
            std::size_t InChannels,
            std::size_t OutHeight,
            std::size_t OutWidth>
  using AdaptiveMaxPool2dReLU6 =
      AdaptiveMaxPool2dImpl<BatchSize, InHeight, InWidth, InChannels, OutHeight, OutWidth, functional::ReLU6>;

  // Fused AdaptiveMaxPool2d + Hardswish
  template <std::size_t BatchSize,
            std::size_t InHeight,
            std::size_t InWidth,
            std::size_t InChannels,
            std::size_t OutHeight,
            std::size_t OutWidth>
  using AdaptiveMaxPool2dHardswish =
      AdaptiveMaxPool2dImpl<BatchSize, InHeight, InWidth, InChannels, OutHeight, OutWidth, functional::Hardswish>;

}  // namespace ffx::nn
