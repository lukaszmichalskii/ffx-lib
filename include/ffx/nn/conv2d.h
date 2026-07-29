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
            std::size_t OutChannels,
            std::size_t KernelHeight,
            std::size_t KernelWidth,
            std::size_t StrideHeight = 1,
            std::size_t StrideWidth = 1,
            std::size_t PaddingHeight = 0,
            std::size_t PaddingWidth = 0,
            bool Bias = true,
            typename TOperator = functional::Identity>
  struct Conv2dImpl {
    static constexpr std::size_t OutHeight = (InHeight + 2 * PaddingHeight - KernelHeight) / StrideHeight + 1;
    static constexpr std::size_t OutWidth = (InWidth + 2 * PaddingWidth - KernelWidth) / StrideWidth + 1;
    static constexpr std::size_t NumberOfElements = BatchSize * OutChannels * OutHeight * OutWidth;

    template <concepts::accelerator TAcc, typename T>
    ALPAKA_FN_ACC void operator()(const TAcc& acc, const T* input, T* output, const T* weights, const T* bias) const {
      for (const auto thread_index : alpaka::uniformElements(acc, NumberOfElements)) {
        const auto w_out_idx = thread_index % OutWidth;
        auto residual = thread_index / OutWidth;
        const auto h_out_idx = residual % OutHeight;
        residual /= OutHeight;
        const auto c_out_idx = residual % OutChannels;
        const auto batch_idx = residual / OutChannels;

        auto accum = Bias ? bias[c_out_idx] : static_cast<T>(0);

        for (std::size_t c_in_idx = 0; c_in_idx < InChannels; ++c_in_idx) {
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
                  batch_idx * InChannels * InHeight * InWidth + c_in_idx * InHeight * InWidth +
                  static_cast<std::size_t>(h_in_idx) * InWidth + static_cast<std::size_t>(w_in_idx);

              const std::size_t weight_flat_offset = c_out_idx * InChannels * KernelHeight * KernelWidth +
                                                     c_in_idx * KernelHeight * KernelWidth + kh_idx * KernelWidth +
                                                     kw_idx;

              accum += input[input_flat_offset] * weights[weight_flat_offset];
            }
          }
        }
        output[thread_index] = TOperator::forward(acc, accum);
      }
    }
  };

  // Conv2d
  template <std::size_t BatchSize,
            std::size_t InHeight,
            std::size_t InWidth,
            std::size_t InChannels,
            std::size_t OutChannels,
            std::size_t KernelHeight,
            std::size_t KernelWidth,
            std::size_t StrideHeight = 1,
            std::size_t StrideWidth = 1,
            std::size_t PaddingHeight = 0,
            std::size_t PaddingWidth = 0,
            bool Bias = true>
  using Conv2d = Conv2dImpl<BatchSize,
                            InHeight,
                            InWidth,
                            InChannels,
                            OutChannels,
                            KernelHeight,
                            KernelWidth,
                            StrideHeight,
                            StrideWidth,
                            PaddingHeight,
                            PaddingWidth,
                            Bias>;

  // Fused Conv2d + ReLU
  template <std::size_t BatchSize,
            std::size_t InHeight,
            std::size_t InWidth,
            std::size_t InChannels,
            std::size_t OutChannels,
            std::size_t KernelHeight,
            std::size_t KernelWidth,
            std::size_t StrideHeight = 1,
            std::size_t StrideWidth = 1,
            std::size_t PaddingHeight = 0,
            std::size_t PaddingWidth = 0,
            bool Bias = true>
  using Conv2dReLU = Conv2dImpl<BatchSize,
                                InHeight,
                                InWidth,
                                InChannels,
                                OutChannels,
                                KernelHeight,
                                KernelWidth,
                                StrideHeight,
                                StrideWidth,
                                PaddingHeight,
                                PaddingWidth,
                                Bias,
                                functional::ReLU>;

  // Fused Conv2d + LeakyReLU
  template <std::size_t BatchSize,
            std::size_t InHeight,
            std::size_t InWidth,
            std::size_t InChannels,
            std::size_t OutChannels,
            std::size_t KernelHeight,
            std::size_t KernelWidth,
            std::size_t StrideHeight = 1,
            std::size_t StrideWidth = 1,
            std::size_t PaddingHeight = 0,
            std::size_t PaddingWidth = 0,
            bool Bias = true,
            std::int64_t FnNominator = 1,
            std::int64_t FnDenominator = 100>
  using Conv2dLeakyReLU = Conv2dImpl<BatchSize,
                                     InHeight,
                                     InWidth,
                                     InChannels,
                                     OutChannels,
                                     KernelHeight,
                                     KernelWidth,
                                     StrideHeight,
                                     StrideWidth,
                                     PaddingHeight,
                                     PaddingWidth,
                                     Bias,
                                     functional::LeakyReLU<FnNominator, FnDenominator>>;

  // Fused Conv2d + Sigmoid
  template <std::size_t BatchSize,
            std::size_t InHeight,
            std::size_t InWidth,
            std::size_t InChannels,
            std::size_t OutChannels,
            std::size_t KernelHeight,
            std::size_t KernelWidth,
            std::size_t StrideHeight = 1,
            std::size_t StrideWidth = 1,
            std::size_t PaddingHeight = 0,
            std::size_t PaddingWidth = 0,
            bool Bias = true>
  using Conv2dSigmoid = Conv2dImpl<BatchSize,
                                   InHeight,
                                   InWidth,
                                   InChannels,
                                   OutChannels,
                                   KernelHeight,
                                   KernelWidth,
                                   StrideHeight,
                                   StrideWidth,
                                   PaddingHeight,
                                   PaddingWidth,
                                   Bias,
                                   functional::Sigmoid>;

  // Fused Conv2d + Tanh
  template <std::size_t BatchSize,
            std::size_t InHeight,
            std::size_t InWidth,
            std::size_t InChannels,
            std::size_t OutChannels,
            std::size_t KernelHeight,
            std::size_t KernelWidth,
            std::size_t StrideHeight = 1,
            std::size_t StrideWidth = 1,
            std::size_t PaddingHeight = 0,
            std::size_t PaddingWidth = 0,
            bool Bias = true>
  using Conv2dTanh = Conv2dImpl<BatchSize,
                                InHeight,
                                InWidth,
                                InChannels,
                                OutChannels,
                                KernelHeight,
                                KernelWidth,
                                StrideHeight,
                                StrideWidth,
                                PaddingHeight,
                                PaddingWidth,
                                Bias,
                                functional::Tanh>;

  // Fused Conv2d + SiLU
  template <std::size_t BatchSize,
            std::size_t InHeight,
            std::size_t InWidth,
            std::size_t InChannels,
            std::size_t OutChannels,
            std::size_t KernelHeight,
            std::size_t KernelWidth,
            std::size_t StrideHeight = 1,
            std::size_t StrideWidth = 1,
            std::size_t PaddingHeight = 0,
            std::size_t PaddingWidth = 0,
            bool Bias = true>
  using Conv2dSiLU = Conv2dImpl<BatchSize,
                                InHeight,
                                InWidth,
                                InChannels,
                                OutChannels,
                                KernelHeight,
                                KernelWidth,
                                StrideHeight,
                                StrideWidth,
                                PaddingHeight,
                                PaddingWidth,
                                Bias,
                                functional::SiLU>;

  // Fused Conv2d + GELU
  template <std::size_t BatchSize,
            std::size_t InHeight,
            std::size_t InWidth,
            std::size_t InChannels,
            std::size_t OutChannels,
            std::size_t KernelHeight,
            std::size_t KernelWidth,
            std::size_t StrideHeight = 1,
            std::size_t StrideWidth = 1,
            std::size_t PaddingHeight = 0,
            std::size_t PaddingWidth = 0,
            bool Bias = true>
  using Conv2dGELU = Conv2dImpl<BatchSize,
                                InHeight,
                                InWidth,
                                InChannels,
                                OutChannels,
                                KernelHeight,
                                KernelWidth,
                                StrideHeight,
                                StrideWidth,
                                PaddingHeight,
                                PaddingWidth,
                                Bias,
                                functional::GELU>;

  // Fused Conv2d + ReLU6
  template <std::size_t BatchSize,
            std::size_t InHeight,
            std::size_t InWidth,
            std::size_t InChannels,
            std::size_t OutChannels,
            std::size_t KernelHeight,
            std::size_t KernelWidth,
            std::size_t StrideHeight = 1,
            std::size_t StrideWidth = 1,
            std::size_t PaddingHeight = 0,
            std::size_t PaddingWidth = 0,
            bool Bias = true>
  using Conv2dReLU6 = Conv2dImpl<BatchSize,
                                 InHeight,
                                 InWidth,
                                 InChannels,
                                 OutChannels,
                                 KernelHeight,
                                 KernelWidth,
                                 StrideHeight,
                                 StrideWidth,
                                 PaddingHeight,
                                 PaddingWidth,
                                 Bias,
                                 functional::ReLU6>;

  // Fused Conv2d + Hardswish
  template <std::size_t BatchSize,
            std::size_t InHeight,
            std::size_t InWidth,
            std::size_t InChannels,
            std::size_t OutChannels,
            std::size_t KernelHeight,
            std::size_t KernelWidth,
            std::size_t StrideHeight = 1,
            std::size_t StrideWidth = 1,
            std::size_t PaddingHeight = 0,
            std::size_t PaddingWidth = 0,
            bool Bias = true>
  using Conv2dHardswish = Conv2dImpl<BatchSize,
                                     InHeight,
                                     InWidth,
                                     InChannels,
                                     OutChannels,
                                     KernelHeight,
                                     KernelWidth,
                                     StrideHeight,
                                     StrideWidth,
                                     PaddingHeight,
                                     PaddingWidth,
                                     Bias,
                                     functional::Hardswish>;

}  // namespace ffx::nn
