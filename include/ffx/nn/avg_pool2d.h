#pragma once

#include <alpaka/alpaka.hpp>
#include <algorithm>
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
            std::size_t KernelHeight,
            std::size_t KernelWidth,
            std::size_t StrideHeight = 1,
            std::size_t StrideWidth = 1,
            std::size_t PaddingHeight = 0,
            std::size_t PaddingWidth = 0,
            typename TOperator = functional::Identity>
  struct AvgPool2dImpl {
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

        // signed bounds to handle padding
        const std::int64_t h_start_unclamped =
            static_cast<std::int64_t>(h_out_idx * StrideHeight) - static_cast<std::int64_t>(PaddingHeight);
        const std::int64_t w_start_unclamped =
            static_cast<std::int64_t>(w_out_idx * StrideWidth) - static_cast<std::int64_t>(PaddingWidth);

        const std::size_t h_start = static_cast<std::size_t>(h_start_unclamped < 0 ? 0 : h_start_unclamped);
        const std::size_t w_start = static_cast<std::size_t>(w_start_unclamped < 0 ? 0 : w_start_unclamped);

        const std::size_t h_end = std::min(static_cast<std::size_t>(h_start_unclamped + KernelHeight), InHeight);
        const std::size_t w_end = std::min(static_cast<std::size_t>(w_start_unclamped + KernelWidth), InWidth);

        // TODO: default PyTorch AvgPool2d count_include_pad=True, allow extension
        const std::size_t pool_size = KernelHeight * KernelWidth;

        T sum_val = static_cast<T>(0);
        // reduction over non-padded input region
        for (std::size_t h_in_idx = h_start; h_in_idx < h_end; ++h_in_idx) {
          for (std::size_t w_in_idx = w_start; w_in_idx < w_end; ++w_in_idx) {
            const std::size_t input_flat_offset = batch_idx * InChannels * InHeight * InWidth +
                                                  c_idx * InHeight * InWidth + h_in_idx * InWidth + w_in_idx;

            sum_val += input[input_flat_offset];
          }
        }

        const T avg_val = sum_val / static_cast<T>(pool_size);
        output[thread_index] = TOperator::forward(acc, avg_val);
      }
    }
  };

  // AvgPool2d
  template <std::size_t BatchSize,
            std::size_t InHeight,
            std::size_t InWidth,
            std::size_t InChannels,
            std::size_t KernelHeight,
            std::size_t KernelWidth,
            std::size_t StrideHeight = 1,
            std::size_t StrideWidth = 1,
            std::size_t PaddingHeight = 0,
            std::size_t PaddingWidth = 0>
  using AvgPool2d = AvgPool2dImpl<BatchSize,
                                  InHeight,
                                  InWidth,
                                  InChannels,
                                  KernelHeight,
                                  KernelWidth,
                                  StrideHeight,
                                  StrideWidth,
                                  PaddingHeight,
                                  PaddingWidth>;

  // Fused AvgPool2d + ReLU
  template <std::size_t BatchSize,
            std::size_t InHeight,
            std::size_t InWidth,
            std::size_t InChannels,
            std::size_t KernelHeight,
            std::size_t KernelWidth,
            std::size_t StrideHeight = 1,
            std::size_t StrideWidth = 1,
            std::size_t PaddingHeight = 0,
            std::size_t PaddingWidth = 0>
  using AvgPool2dReLU = AvgPool2dImpl<BatchSize,
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

  // Fused AvgPool2d + LeakyReLU
  template <std::size_t BatchSize,
            std::size_t InHeight,
            std::size_t InWidth,
            std::size_t InChannels,
            std::size_t KernelHeight,
            std::size_t KernelWidth,
            std::size_t StrideHeight = 1,
            std::size_t StrideWidth = 1,
            std::size_t PaddingHeight = 0,
            std::size_t PaddingWidth = 0,
            std::int64_t FnNominator = 1,
            std::int64_t FnDenominator = 100>
  using AvgPool2dLeakyReLU = AvgPool2dImpl<BatchSize,
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

  // Fused AvgPool2d + Sigmoid
  template <std::size_t BatchSize,
            std::size_t InHeight,
            std::size_t InWidth,
            std::size_t InChannels,
            std::size_t KernelHeight,
            std::size_t KernelWidth,
            std::size_t StrideHeight = 1,
            std::size_t StrideWidth = 1,
            std::size_t PaddingHeight = 0,
            std::size_t PaddingWidth = 0>
  using AvgPool2dSigmoid = AvgPool2dImpl<BatchSize,
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

  // Fused AvgPool2d + Tanh
  template <std::size_t BatchSize,
            std::size_t InHeight,
            std::size_t InWidth,
            std::size_t InChannels,
            std::size_t KernelHeight,
            std::size_t KernelWidth,
            std::size_t StrideHeight = 1,
            std::size_t StrideWidth = 1,
            std::size_t PaddingHeight = 0,
            std::size_t PaddingWidth = 0>
  using AvgPool2dTanh = AvgPool2dImpl<BatchSize,
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

  // Fused AvgPool2d + SiLU
  template <std::size_t BatchSize,
            std::size_t InHeight,
            std::size_t InWidth,
            std::size_t InChannels,
            std::size_t KernelHeight,
            std::size_t KernelWidth,
            std::size_t StrideHeight = 1,
            std::size_t StrideWidth = 1,
            std::size_t PaddingHeight = 0,
            std::size_t PaddingWidth = 0>
  using AvgPool2dSiLU = AvgPool2dImpl<BatchSize,
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

  // Fused AvgPool2d + GELU
  template <std::size_t BatchSize,
            std::size_t InHeight,
            std::size_t InWidth,
            std::size_t InChannels,
            std::size_t KernelHeight,
            std::size_t KernelWidth,
            std::size_t StrideHeight = 1,
            std::size_t StrideWidth = 1,
            std::size_t PaddingHeight = 0,
            std::size_t PaddingWidth = 0>
  using AvgPool2dGELU = AvgPool2dImpl<BatchSize,
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

  // Fused AvgPool2d + ReLU6
  template <std::size_t BatchSize,
            std::size_t InHeight,
            std::size_t InWidth,
            std::size_t InChannels,
            std::size_t KernelHeight,
            std::size_t KernelWidth,
            std::size_t StrideHeight = 1,
            std::size_t StrideWidth = 1,
            std::size_t PaddingHeight = 0,
            std::size_t PaddingWidth = 0>
  using AvgPool2dReLU6 = AvgPool2dImpl<BatchSize,
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

  // Fused AvgPool2d + Hardswish
  template <std::size_t BatchSize,
            std::size_t InHeight,
            std::size_t InWidth,
            std::size_t InChannels,
            std::size_t KernelHeight,
            std::size_t KernelWidth,
            std::size_t StrideHeight = 1,
            std::size_t StrideWidth = 1,
            std::size_t PaddingHeight = 0,
            std::size_t PaddingWidth = 0>
  using AvgPool2dHardswish = AvgPool2dImpl<BatchSize,
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