#pragma once

#include <alpaka/alpaka.hpp>

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
            std::size_t KernelHeight,
            std::size_t KernelWidth,
            std::size_t StrideHeight = KernelHeight,
            std::size_t StrideWidth = KernelWidth,
            std::size_t PaddingHeight = 0,
            std::size_t PaddingWidth = 0,
            typename TOperator = functional::Identity>
  struct MaxPool2dImpl {
    static constexpr std::size_t OutHeight = (InHeight + 2 * PaddingHeight - KernelHeight) / StrideHeight + 1;
    static constexpr std::size_t OutWidth = (InWidth + 2 * PaddingWidth - KernelWidth) / StrideWidth + 1;
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

        T max_val = -std::numeric_limits<T>::max();

        // local pooling window
        for (std::size_t kh_idx = 0; kh_idx < KernelHeight; ++kh_idx) {
          const auto h_in_idx =
              static_cast<std::int64_t>(h_out_idx * StrideHeight + kh_idx) - static_cast<std::int64_t>(PaddingHeight);
          if (h_in_idx < 0 || h_in_idx >= static_cast<std::int64_t>(InHeight))
            continue;

          for (std::size_t kw_idx = 0; kw_idx < KernelWidth; ++kw_idx) {
            const auto w_in_idx =
                static_cast<std::int64_t>(w_out_idx * StrideWidth + kw_idx) - static_cast<std::int64_t>(PaddingWidth);
            if (w_in_idx < 0 || w_in_idx >= static_cast<std::int64_t>(InWidth))
              continue;

            const std::size_t input_flat_offset =
                batch_idx * InChannels * InHeight * InWidth + c_idx * InHeight * InWidth +
                static_cast<std::size_t>(h_in_idx) * InWidth + static_cast<std::size_t>(w_in_idx);

            const T val = input[input_flat_offset];
            max_val = alpaka::math::max(acc, val, max_val);
          }
        }
        output[thread_index] = TOperator::forward(acc, max_val);
      }
    }
  };

  // MaxPool2d
  template <std::size_t BatchSize,
            std::size_t InHeight,
            std::size_t InWidth,
            std::size_t InChannels,
            std::size_t KernelHeight,
            std::size_t KernelWidth,
            std::size_t StrideHeight = KernelHeight,
            std::size_t StrideWidth = KernelWidth,
            std::size_t PaddingHeight = 0,
            std::size_t PaddingWidth = 0>
  using MaxPool2d = MaxPool2dImpl<BatchSize,
                                  InHeight,
                                  InWidth,
                                  InChannels,
                                  KernelHeight,
                                  KernelWidth,
                                  StrideHeight,
                                  StrideWidth,
                                  PaddingHeight,
                                  PaddingWidth>;

  // Fused MaxPool2d + ReLU
  template <std::size_t BatchSize,
            std::size_t InHeight,
            std::size_t InWidth,
            std::size_t InChannels,
            std::size_t KernelHeight,
            std::size_t KernelWidth,
            std::size_t StrideHeight = KernelHeight,
            std::size_t StrideWidth = KernelWidth,
            std::size_t PaddingHeight = 0,
            std::size_t PaddingWidth = 0>
  using MaxPool2dReLU = MaxPool2dImpl<BatchSize,
                                      InHeight,
                                      InWidth,
                                      InChannels,
                                      KernelHeight,
                                      KernelWidth,
                                      StrideHeight,
                                      StrideWidth,
                                      PaddingHeight,
                                      PaddingWidth,
                                      functional::ReLU>;

  // Fused MaxPool2d + LeakyReLU
  template <std::size_t BatchSize,
            std::size_t InHeight,
            std::size_t InWidth,
            std::size_t InChannels,
            std::size_t KernelHeight,
            std::size_t KernelWidth,
            std::size_t StrideHeight = KernelHeight,
            std::size_t StrideWidth = KernelWidth,
            std::size_t PaddingHeight = 0,
            std::size_t PaddingWidth = 0,
            std::int64_t FnNominator = 1,
            std::int64_t FnDenominator = 100>
  using MaxPool2dLeakyReLU = MaxPool2dImpl<BatchSize,
                                           InHeight,
                                           InWidth,
                                           InChannels,
                                           KernelHeight,
                                           KernelWidth,
                                           StrideHeight,
                                           StrideWidth,
                                           PaddingHeight,
                                           PaddingWidth,
                                           functional::LeakyReLU<FnNominator, FnDenominator>>;

  // Fused MaxPool2d + Sigmoid
  template <std::size_t BatchSize,
            std::size_t InHeight,
            std::size_t InWidth,
            std::size_t InChannels,
            std::size_t KernelHeight,
            std::size_t KernelWidth,
            std::size_t StrideHeight = KernelHeight,
            std::size_t StrideWidth = KernelWidth,
            std::size_t PaddingHeight = 0,
            std::size_t PaddingWidth = 0>
  using MaxPool2dSigmoid = MaxPool2dImpl<BatchSize,
                                         InHeight,
                                         InWidth,
                                         InChannels,
                                         KernelHeight,
                                         KernelWidth,
                                         StrideHeight,
                                         StrideWidth,
                                         PaddingHeight,
                                         PaddingWidth,
                                         functional::Sigmoid>;

  // Fused MaxPool2d + Tanh
  template <std::size_t BatchSize,
            std::size_t InHeight,
            std::size_t InWidth,
            std::size_t InChannels,
            std::size_t KernelHeight,
            std::size_t KernelWidth,
            std::size_t StrideHeight = KernelHeight,
            std::size_t StrideWidth = KernelWidth,
            std::size_t PaddingHeight = 0,
            std::size_t PaddingWidth = 0>
  using MaxPool2dTanh = MaxPool2dImpl<BatchSize,
                                      InHeight,
                                      InWidth,
                                      InChannels,
                                      KernelHeight,
                                      KernelWidth,
                                      StrideHeight,
                                      StrideWidth,
                                      PaddingHeight,
                                      PaddingWidth,
                                      functional::Tanh>;

  // Fused MaxPool2d + SiLU
  template <std::size_t BatchSize,
            std::size_t InHeight,
            std::size_t InWidth,
            std::size_t InChannels,
            std::size_t KernelHeight,
            std::size_t KernelWidth,
            std::size_t StrideHeight = KernelHeight,
            std::size_t StrideWidth = KernelWidth,
            std::size_t PaddingHeight = 0,
            std::size_t PaddingWidth = 0>
  using MaxPool2dSiLU = MaxPool2dImpl<BatchSize,
                                      InHeight,
                                      InWidth,
                                      InChannels,
                                      KernelHeight,
                                      KernelWidth,
                                      StrideHeight,
                                      StrideWidth,
                                      PaddingHeight,
                                      PaddingWidth,
                                      functional::SiLU>;

  // Fused MaxPool2d + GELU
  template <std::size_t BatchSize,
            std::size_t InHeight,
            std::size_t InWidth,
            std::size_t InChannels,
            std::size_t KernelHeight,
            std::size_t KernelWidth,
            std::size_t StrideHeight = KernelHeight,
            std::size_t StrideWidth = KernelWidth,
            std::size_t PaddingHeight = 0,
            std::size_t PaddingWidth = 0>
  using MaxPool2dGELU = MaxPool2dImpl<BatchSize,
                                      InHeight,
                                      InWidth,
                                      InChannels,
                                      KernelHeight,
                                      KernelWidth,
                                      StrideHeight,
                                      StrideWidth,
                                      PaddingHeight,
                                      PaddingWidth,
                                      functional::GELU>;

  // Fused MaxPool2d + ReLU6
  template <std::size_t BatchSize,
            std::size_t InHeight,
            std::size_t InWidth,
            std::size_t InChannels,
            std::size_t KernelHeight,
            std::size_t KernelWidth,
            std::size_t StrideHeight = KernelHeight,
            std::size_t StrideWidth = KernelWidth,
            std::size_t PaddingHeight = 0,
            std::size_t PaddingWidth = 0>
  using MaxPool2dReLU6 = MaxPool2dImpl<BatchSize,
                                       InHeight,
                                       InWidth,
                                       InChannels,
                                       KernelHeight,
                                       KernelWidth,
                                       StrideHeight,
                                       StrideWidth,
                                       PaddingHeight,
                                       PaddingWidth,
                                       functional::ReLU6>;

  // Fused MaxPool2d + Hardswish
  template <std::size_t BatchSize,
            std::size_t InHeight,
            std::size_t InWidth,
            std::size_t InChannels,
            std::size_t KernelHeight,
            std::size_t KernelWidth,
            std::size_t StrideHeight = KernelHeight,
            std::size_t StrideWidth = KernelWidth,
            std::size_t PaddingHeight = 0,
            std::size_t PaddingWidth = 0>
  using MaxPool2dHardswish = MaxPool2dImpl<BatchSize,
                                           InHeight,
                                           InWidth,
                                           InChannels,
                                           KernelHeight,
                                           KernelWidth,
                                           StrideHeight,
                                           StrideWidth,
                                           PaddingHeight,
                                           PaddingWidth,
                                           functional::Hardswish>;

}  // namespace ffx::nn
