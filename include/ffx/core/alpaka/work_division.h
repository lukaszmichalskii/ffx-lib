#pragma once

#include <alpaka/alpaka.hpp>

#include "ffx/core/alpaka/config.h"
#include "ffx/core/detail/concepts.h"

namespace ffx {

  using namespace ffx_alpaka;

  // Trait describing whether the accelerator expects the threads-per-block and elements-per-thread to be swapped
  template <concepts::accelerator TAcc>
  struct requires_single_thread_per_block : public std::true_type {};

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
  template <typename TDim>
  struct requires_single_thread_per_block<alpaka::AccGpuCudaRt<TDim, Idx>> : public std::false_type {};
#endif  // ALPAKA_ACC_GPU_CUDA_ENABLED

#ifdef ALPAKA_ACC_GPU_HIP_ENABLED
  template <typename TDim>
  struct requires_single_thread_per_block<alpaka::AccGpuHipRt<TDim, Idx>> : public std::false_type {};
#endif  // ALPAKA_ACC_GPU_HIP_ENABLED

  // Whether the accelerator expects the threads-per-block and elements-per-thread to be swapped
  template <concepts::accelerator TAcc>
  constexpr bool requires_single_thread_per_block_v = requires_single_thread_per_block<TAcc>::value;

  // If the first argument is not a multiple of the second argument, round it up to the next multiple.
  constexpr Idx round_up_by(const Idx value, const Idx divisor) { return (value + divisor - 1) / divisor * divisor; }

  // Return the integer division of the first argument by the second argument, rounded up to the next integer.
  constexpr Idx divide_up_by(const Idx value, const Idx divisor) { return (value + divisor - 1) / divisor; }

  // Creates the accelerator-dependent work division for 1-dimensional operations.
  template <concepts::accelerator TAcc>
  WorkDiv<Dim1D> make_workdiv(const Idx blocks_per_grid, const Idx threads_per_block_or_elements_per_thread) {
    if constexpr (not requires_single_thread_per_block_v<TAcc>) {
      // On GPU backends, each thread is looking at a single element:
      //   - threads_per_block_or_elements_per_thread is the number of threads per block;
      //   - elements_per_thread is always 1.
      constexpr auto elements_per_thread = Idx{1};
      return WorkDiv<Dim1D>(blocks_per_grid, threads_per_block_or_elements_per_thread, elements_per_thread);
    } else {
      // On CPU backends, run serially with a single thread per block:
      //   - threads_per_block is always 1;
      //   - threads_per_block_or_elements_per_thread is the number of elements per thread.
      constexpr auto threads_per_block = Idx{1};
      return WorkDiv<Dim1D>(blocks_per_grid, threads_per_block, threads_per_block_or_elements_per_thread);
    }
  }

  // Creates the accelerator-dependent workdiv for N-dimensional operations.
  template <concepts::accelerator TAcc>
  WorkDiv<alpaka::Dim<TAcc>> make_workdiv(const Vec<alpaka::Dim<TAcc>>& blocks_per_grid,
                                          const Vec<alpaka::Dim<TAcc>>& threads_per_block_or_elements_per_thread) {
    using Dim = alpaka::Dim<TAcc>;
    if constexpr (not requires_single_thread_per_block_v<TAcc>) {
      // On GPU backends, each thread is looking at a single element:
      //   - threads_per_block_or_elements_per_thread is the number of threads per block;
      //   - elements_per_thread is always 1.
      const auto elements_per_thread = Vec<Dim>::ones();
      return WorkDiv<Dim>(blocks_per_grid, threads_per_block_or_elements_per_thread, elements_per_thread);
    } else {
      // On CPU backends, run serially with a single thread per block:
      //   - threads_per_block is always 1;
      //   - threads_per_block_or_elements_per_thread is the number of elements per thread.
      const auto threads_per_block = Vec<Dim>::ones();
      return WorkDiv<Dim>(blocks_per_grid, threads_per_block, threads_per_block_or_elements_per_thread);
    }
  }

}  // namespace ffx
