#pragma once

#include <alpaka/alpaka.hpp>

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
#include <thrust/count.h>
#include <thrust/execution_policy.h>
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
#include <thrust/count.h>
#include <thrust/execution_policy.h>
#else
#include <algorithm>
#endif

#include "ffx/core/detail/concepts.h"

namespace ffx::algorithm {

  template <typename InputIterator, typename Predicate>
  ALPAKA_FN_HOST constexpr auto count_if(InputIterator first, InputIterator last, Predicate pred) {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
    return thrust::count_if(thrust::device, first, last, pred);
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
    return thrust::count_if(thrust::hip::par, first, last, pred);
#else
    return std::count_if(first, last, pred);
#endif
  }

  template <typename ExecutionPolicy, typename InputIterator, typename Predicate>
    requires(!alpaka::isQueue<std::remove_cvref_t<ExecutionPolicy>>)
  ALPAKA_FN_HOST constexpr auto count_if(ExecutionPolicy&& policy,
                                         InputIterator first,
                                         InputIterator last,
                                         Predicate pred) {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
    return thrust::count_if(std::forward<ExecutionPolicy>(policy), first, last, pred);
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
    return thrust::count_if(std::forward<ExecutionPolicy>(policy), first, last, pred);
#else
    return std::count_if(std::forward<ExecutionPolicy>(policy), first, last, pred);
#endif
  }

  template <ffx::concepts::queue TQueue, typename InputIterator, typename Predicate>
  ALPAKA_FN_HOST constexpr auto count_if(TQueue& queue, InputIterator first, InputIterator last, Predicate pred) {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
    return thrust::count_if(thrust::cuda::par_nosync.on(queue.getNativeHandle()), first, last, pred);
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
    return thrust::count_if(thrust::hip::par.on(queue.getNativeHandle()), first, last, pred);
#else
    alpaka::wait(queue);
    return std::count_if(first, last, pred);
#endif
  }

}  // namespace ffx::algorithm