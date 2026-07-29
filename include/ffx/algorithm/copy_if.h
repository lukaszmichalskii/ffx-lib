#pragma once

#include <alpaka/alpaka.hpp>

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#else
#include <algorithm>
#endif

#include "ffx/core/detail/concepts.h"

namespace ffx::algorithm {

  template <typename InputIterator, typename OutputIterator, typename Predicate>
  ALPAKA_FN_HOST constexpr OutputIterator copy_if(InputIterator first,
                                                  InputIterator last,
                                                  OutputIterator result,
                                                  Predicate pred) {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
    return thrust::copy_if(thrust::device, first, last, result, pred);
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
    return thrust::copy_if(thrust::hip::par, first, last, result, pred);
#else
    return std::copy_if(first, last, result, pred);
#endif
  }

  template <typename ExecutionPolicy, typename InputIterator, typename OutputIterator, typename Predicate>
    requires(!alpaka::isQueue<std::remove_cvref_t<ExecutionPolicy>>)
  ALPAKA_FN_HOST constexpr OutputIterator copy_if(
      ExecutionPolicy&& policy, InputIterator first, InputIterator last, OutputIterator result, Predicate pred) {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
    return thrust::copy_if(std::forward<ExecutionPolicy>(policy), first, last, result, pred);
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
    return thrust::copy_if(std::forward<ExecutionPolicy>(policy), first, last, result, pred);
#else
    return std::copy_if(std::forward<ExecutionPolicy>(policy), first, last, result, pred);
#endif
  }

  template <concepts::queue TQueue, typename InputIterator, typename OutputIterator, typename Predicate>
  ALPAKA_FN_HOST constexpr OutputIterator copy_if(
      TQueue& queue, InputIterator first, InputIterator last, OutputIterator result, Predicate pred) {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
    return thrust::copy_if(thrust::cuda::par_nosync.on(queue.getNativeHandle()), first, last, result, pred);
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
    return thrust::copy_if(thrust::hip::par.on(queue.getNativeHandle()), first, last, result, pred);
#else
    alpaka::wait(queue);
    return std::copy_if(first, last, result, pred);
#endif
  }

}  // namespace ffx::algorithm