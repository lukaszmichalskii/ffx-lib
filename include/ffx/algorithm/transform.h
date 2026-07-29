#pragma once

#include <alpaka/alpaka.hpp>

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
#include <thrust/execution_policy.h>
#include <thrust/transform.h>
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
#include <thrust/execution_policy.h>
#include <thrust/transform.h>
#else
#include <algorithm>
#endif

#include "ffx/core/detail/concepts.h"

namespace ffx::algorithm {

  template <typename InputIterator, typename OutputIterator, typename UnaryOp>
  ALPAKA_FN_HOST constexpr OutputIterator transform(InputIterator first,
                                                    InputIterator last,
                                                    OutputIterator result,
                                                    UnaryOp op) {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
    return thrust::transform(thrust::device, first, last, result, op);
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
    return thrust::transform(thrust::hip::par, first, last, result, op);
#else
    return std::transform(first, last, result, op);
#endif
  }

  template <typename ExecutionPolicy, typename InputIterator, typename OutputIterator, typename UnaryOp>
    requires(!alpaka::isQueue<std::remove_cvref_t<ExecutionPolicy>>)
  ALPAKA_FN_HOST constexpr OutputIterator transform(
      ExecutionPolicy&& policy, InputIterator first, InputIterator last, OutputIterator result, UnaryOp op) {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
    return thrust::transform(std::forward<ExecutionPolicy>(policy), first, last, result, op);
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
    return thrust::transform(std::forward<ExecutionPolicy>(policy), first, last, result, op);
#else
    return std::transform(std::forward<ExecutionPolicy>(policy), first, last, result, op);
#endif
  }

  template <concepts::queue TQueue, typename InputIterator, typename OutputIterator, typename UnaryOp>
  ALPAKA_FN_HOST constexpr OutputIterator transform(
      TQueue& queue, InputIterator first, InputIterator last, OutputIterator result, UnaryOp op) {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
    return thrust::transform(thrust::cuda::par_nosync.on(queue.getNativeHandle()), first, last, result, op);
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
    return thrust::transform(thrust::hip::par.on(queue.getNativeHandle()), first, last, result, op);
#else
    alpaka::wait(queue);
    return std::transform(first, last, result, op);
#endif
  }

  template <typename InputIterator1, typename InputIterator2, typename OutputIterator, typename BinaryOp>
  ALPAKA_FN_HOST constexpr OutputIterator transform(
      InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, OutputIterator result, BinaryOp op) {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
    return thrust::transform(thrust::device, first1, last1, first2, result, op);
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
    return thrust::transform(thrust::hip::par, first1, last1, first2, result, op);
#else
    return std::transform(first1, last1, first2, result, op);
#endif
  }

  template <typename ExecutionPolicy,
            typename InputIterator1,
            typename InputIterator2,
            typename OutputIterator,
            typename BinaryOp>
    requires(!alpaka::isQueue<std::remove_cvref_t<ExecutionPolicy>>)
  ALPAKA_FN_HOST constexpr OutputIterator transform(ExecutionPolicy&& policy,
                                                    InputIterator1 first1,
                                                    InputIterator1 last1,
                                                    InputIterator2 first2,
                                                    OutputIterator result,
                                                    BinaryOp op) {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
    return thrust::transform(std::forward<ExecutionPolicy>(policy), first1, last1, first2, result, op);
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
    return thrust::transform(std::forward<ExecutionPolicy>(policy), first1, last1, first2, result, op);
#else
    return std::transform(std::forward<ExecutionPolicy>(policy), first1, last1, first2, result, op);
#endif
  }

  template <concepts::queue TQueue,
            typename InputIterator1,
            typename InputIterator2,
            typename OutputIterator,
            typename BinaryOp>
  ALPAKA_FN_HOST constexpr OutputIterator transform(TQueue& queue,
                                                    InputIterator1 first1,
                                                    InputIterator1 last1,
                                                    InputIterator2 first2,
                                                    OutputIterator result,
                                                    BinaryOp op) {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
    return thrust::transform(thrust::cuda::par_nosync.on(queue.getNativeHandle()), first1, last1, first2, result, op);
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
    return thrust::transform(thrust::hip::par.on(queue.getNativeHandle()), first1, last1, first2, result, op);
#else
    alpaka::wait(queue);
    return std::transform(first1, last1, first2, result, op);
#endif
  }

}  // namespace ffx::algorithm