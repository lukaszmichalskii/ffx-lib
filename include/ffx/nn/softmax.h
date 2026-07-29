#pragma once

#include <alpaka/alpaka.hpp>
#include <cmath>
#include <cstddef>
#include "ffx/core/detail/concepts.h"

namespace ffx::nn {

  template <std::size_t NumReductions,        // total number of independent vectors to normalize
            std::size_t ReductionSize,        // number of elements per reduction vector
            std::size_t StrideWithinDim = 1,  // memory stride between elements IN THE SAME vector
            std::size_t OuterStride = ReductionSize * StrideWithinDim  // distance between vector bases
            >
  struct SoftmaxImpl {
    template <concepts::accelerator TAcc, typename T>
    ALPAKA_FN_ACC void operator()(const TAcc& acc, const T* input, T* output) const {
      for (const auto r : alpaka::uniformElements(acc, NumReductions)) {
        const auto base_offset = r * OuterStride;

        // max
        auto max_val = input[base_offset];
        for (auto i = 1u; i < ReductionSize; ++i) {
          const auto val = input[base_offset + i * StrideWithinDim];
          if (val > max_val)
            max_val = val;
        }

        // sum of exp
        auto sum_exp = static_cast<T>(0);
        for (auto i = 0u; i < ReductionSize; ++i) {
          sum_exp += std::exp(input[base_offset + i * StrideWithinDim] - max_val);
        }

        // normalize
        for (auto i = 0u; i < ReductionSize; ++i) {
          const auto idx = base_offset + i * StrideWithinDim;
          output[idx] = std::exp(input[idx] - max_val) / sum_exp;
        }
      }
    }
  };

  template <std::size_t VectorLength>
  using Softmax1D = SoftmaxImpl<1,            /**< NumReductions */
                                VectorLength, /**< ReductionSize */
                                1,            /**< StrideWithinDim */
                                VectorLength /**< OuterStride */>;

  template <std::size_t BatchSize, std::size_t FeatureDim>
  using Softmax2D = SoftmaxImpl<
      /* NumReductions */ BatchSize,
      /* ReductionSize */ FeatureDim,
      /* StrideWithinDim */ 1,
      /* OuterStride */ FeatureDim>;

  template <std::size_t BatchSize, std::size_t SequenceLength, std::size_t EmbeddingDim>
  using Softmax3D = SoftmaxImpl<
      /* NumReductions */ BatchSize * SequenceLength,
      /* ReductionSize */ EmbeddingDim,
      /* StrideWithinDim */ 1,
      /* OuterStride */ EmbeddingDim>;

  template <std::size_t BatchSize, std::size_t NumHeads, std::size_t QueryLength, std::size_t KeyLength>
  using Softmax4D = SoftmaxImpl<
      /* NumReductions */ BatchSize * NumHeads * QueryLength,
      /* ReductionSize */ KeyLength,
      /* StrideWithinDim */ 1,
      /* OuterStride */ KeyLength>;

  template <std::size_t NumReductions, std::size_t ReductionSize>
  using Softmax = SoftmaxImpl<NumReductions, ReductionSize, 1, ReductionSize>;

}  // namespace ffx::nn