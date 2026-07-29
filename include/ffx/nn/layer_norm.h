#pragma once

#include <alpaka/alpaka.hpp>
#include <ffx/core/detail/concepts.h>

namespace ffx::nn {

  template <std::size_t NumTokens, std::size_t EmbeddingDim, std::int64_t Numerator = 1, std::int64_t Denominator = 1000000>
  struct LayerNormImpl {
    template <concepts::accelerator TAcc, typename T>
    ALPAKA_FN_ACC void operator()(const TAcc& acc, const T* input, T* output, const T* gamma, const T* beta) const {
      constexpr T eps = static_cast<T>(Numerator) / static_cast<T>(Denominator);

      for (const auto t : alpaka::uniformElements(acc, NumTokens)) {
        const auto offset = t * EmbeddingDim;

        // mean
        auto sum = static_cast<T>(0);
        for (auto d = 0u; d < EmbeddingDim; ++d) {
          sum += input[offset + d];
        }
        const auto mean = sum / static_cast<T>(EmbeddingDim);

        // variance
        auto variance_sum = static_cast<T>(0);
        for (auto d = 0u; d < EmbeddingDim; ++d) {
          const auto diff = input[offset + d] - mean;
          variance_sum += diff * diff;
        }
        const auto variance = variance_sum / static_cast<T>(EmbeddingDim);
        const auto rsqrt_std = static_cast<T>(1) / std::sqrt(variance + eps);

        // scale and shift (affine transform using gamma and beta)
        for (auto d = 0u; d < EmbeddingDim; ++d) {
          const auto idx = offset + d;
          const auto normalized = (input[idx] - mean) * rsqrt_std;
          const auto g = gamma ? gamma[d] : static_cast<T>(1);
          const auto b = beta ? beta[d] : static_cast<T>(0);
          output[idx] = normalized * g + b;
        }
      }
    }
  };

}  // namespace ffx::nn
