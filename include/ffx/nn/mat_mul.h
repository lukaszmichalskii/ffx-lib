#pragma once

#include <alpaka/alpaka.hpp>
#include "ffx/core/detail/concepts.h"

namespace ffx::nn {

  // Batch Matrix Multiplication
  template <std::size_t BatchSize,
            std::size_t M,  // number of rows matrix A
            std::size_t K,  // number of shared dimension A/B
            std::size_t N,  // number of cols matrix B
            std::size_t A_BatchStride,
            std::size_t A_RowStride,
            std::size_t A_ColStride,
            std::size_t B_BatchStride,
            std::size_t B_RowStride,
            std::size_t B_ColStride>
  struct MatMulImpl {
    template <concepts::accelerator TAcc, typename T>
    ALPAKA_FN_ACC void operator()(const TAcc& acc, const T* A, const T* B, T* C) const {
      constexpr std::size_t SliceSize = M * N;

      for (const auto thread_index : alpaka::uniformElements(acc, BatchSize * SliceSize)) {
        const auto b = thread_index / SliceSize;
        const auto slice_idx = thread_index % SliceSize;
        const auto m = slice_idx / N;
        const auto n = slice_idx % N;

        auto accum = static_cast<T>(0);

        for (auto k = 0u; k < K; ++k) {
          const auto a_idx = b * A_BatchStride + m * A_RowStride + k * A_ColStride;
          const auto b_idx = b * B_BatchStride + k * B_RowStride + n * B_ColStride;
          accum += A[a_idx] * B[b_idx];
        }

        C[thread_index] = accum;
      }
    }
  };

  template <std::size_t BatchSize, std::size_t M, std::size_t K, std::size_t N>
  using MatMul = MatMulImpl<BatchSize, M, K, N, M * K, K, 1, K * N, N, 1>;

}  // namespace ffx::nn
