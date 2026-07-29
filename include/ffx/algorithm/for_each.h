#pragma once

#include <alpaka/alpaka.hpp>

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#else
#include <algorithm>
#endif

#include "ffx/core/detail/concepts.h"

namespace ffx::algorithm {

  template <typename InputIterator, typename UnaryOp>
  ALPAKA_FN_HOST constexpr UnaryOp for_each(InputIterator first, InputIterator last, UnaryOp op) {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
    return thrust::for_each(thrust::device, first, last, op);
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
    return thrust::for_each(thrust::hip::par, first, last, op);
#else
    return std::for_each(first, last, op);
#endif
  }

  template <typename ExecutionPolicy, typename InputIterator, typename UnaryOp>
    requires(!alpaka::isQueue<std::remove_cvref_t<ExecutionPolicy>>)
  ALPAKA_FN_HOST constexpr UnaryOp for_each(ExecutionPolicy&& policy,
                                            InputIterator first,
                                            InputIterator last,
                                            UnaryOp op) {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
    return thrust::for_each(std::forward<ExecutionPolicy>(policy), first, last, op);
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
    return thrust::for_each(std::forward<ExecutionPolicy>(policy), first, last, op);
#else
    return std::for_each(std::forward<ExecutionPolicy>(policy), first, last, op);
#endif
  }

  template <concepts::queue TQueue, typename InputIterator, typename UnaryOp>
  ALPAKA_FN_HOST constexpr void for_each(TQueue& queue, InputIterator first, InputIterator last, UnaryOp op) {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
    thrust::for_each(thrust::cuda::par_nosync.on(queue.getNativeHandle()), first, last, op);
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
    thrust::for_each(thrust::hip::par.on(queue.getNativeHandle()), first, last, op);
#else
    alpaka::wait(queue);
    std::for_each(first, last, op);
#endif
  }

}  // namespace ffx::algorithm
