#pragma once

#include <alpaka/alpaka.hpp>
#include <ffx/core/detail/concepts.h>

namespace ffx::nn {

  template <std::size_t NumTokens, std::size_t EmbeddingDim, std::int64_t Numerator = 1, std::int64_t Denominator = 1000000>
  struct RMSNormImpl {
    template <concepts::accelerator TAcc, typename T>
    ALPAKA_FN_ACC void operator()(const TAcc& acc, const T* input, T* output, const T* gamma) const {
      constexpr T eps = static_cast<T>(Numerator) / static_cast<T>(Denominator);

      for (const auto t : alpaka::uniformElements(acc, NumTokens)) {
        const auto offset = t * EmbeddingDim;

        // mean square (RMS)
        auto ms_sum = static_cast<T>(0);
        for (auto d = 0u; d < EmbeddingDim; ++d) {
          const auto val = input[offset + d];
          ms_sum += val * val;
        }
        const auto rms = static_cast<T>(1) / alpaka::math::sqrt(acc, ms_sum / static_cast<T>(EmbeddingDim) + eps);

        // normalize and scale with gamma
        for (auto d = 0u; d < EmbeddingDim; ++d) {
          const auto idx = offset + d;
          const auto g = gamma ? gamma[d] : static_cast<T>(1);
          output[idx] = (input[idx] * rms) * g;
        }
      }
    }
  };

}  // namespace ffx::nn
