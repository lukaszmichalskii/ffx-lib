#pragma once

#include <alpaka/alpaka.hpp>
#include "ffx/core/detail/concepts.h"

namespace ffx::nn {

  template <std::size_t BatchSize,
            std::size_t Channels,
            std::size_t Height,
            std::size_t Width,
            std::int64_t EpsilonNumerator = 1,
            std::int64_t EpsilonDenominator = 100000>
  struct BatchNorm2d {
    static constexpr std::size_t BatchNorm2dSize = Height * Width;
    static constexpr std::size_t NumerOfElements = BatchSize * Channels * Height * Width;

    template <concepts::accelerator TAcc, typename T>
    ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                  const T* input,
                                  T* output,
                                  const T* gemma,
                                  const T* beta,
                                  const T* mean,
                                  const T* variance) const {
      constexpr T epsilon = static_cast<T>(EpsilonNumerator) / static_cast<T>(EpsilonDenominator);
      for (const auto thread_index : alpaka::uniformElements(acc, NumerOfElements)) {
        const auto batch_norm2d_index = thread_index % BatchNorm2dSize;
        const auto channel_index = (thread_index / BatchNorm2dSize) % Channels;
        const auto batch_idx = thread_index / (Channels * BatchNorm2dSize);

        const auto mean_v = mean[channel_index];
        const auto variance_v = variance[channel_index];
        const auto gemma_v = gemma[channel_index];
        const auto beta_v = beta[channel_index];

        const auto normalized = (input[thread_index] - mean_v) / alpaka::math::sqrt(acc, variance_v + epsilon);
        output[thread_index] = normalized * gemma_v + beta_v;
      }
    }
  };

}  // namespace ffx::nn
