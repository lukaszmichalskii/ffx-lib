#pragma once

#include <alpaka/alpaka.hpp>

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#else
#include <numeric>
#endif

#include "ffx/core/detail/concepts.h"

namespace ffx::algorithm {

  template <typename InputIterator>
  ALPAKA_FN_HOST constexpr std::iterator_traits<InputIterator>::value_type reduce(InputIterator first,
                                                                                  InputIterator last) {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
    return thrust::reduce(thrust::device, first, last);
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
    return thrust::reduce(thrust::hip::par, first, last);
#else
    return std::reduce(first, last);
#endif
  }

  template <typename ExecutionPolicy, typename ForwardIterator>
    requires(!alpaka::isQueue<std::remove_cvref_t<ExecutionPolicy>>)
  ALPAKA_FN_HOST constexpr std::iterator_traits<ForwardIterator>::value_type reduce(ExecutionPolicy&& policy,
                                                                                    ForwardIterator first,
                                                                                    ForwardIterator last) {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
    return thrust::reduce(std::forward<ExecutionPolicy>(policy), first, last);
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
    return thrust::reduce(std::forward<ExecutionPolicy>(policy), first, last);
#else
    return std::reduce(std::forward<ExecutionPolicy>(policy), first, last);
#endif
  }

  template <typename InputIterator, typename T>
  ALPAKA_FN_HOST constexpr T reduce(InputIterator first, InputIterator last, T init) {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
    return thrust::reduce(thrust::device, first, last, init);
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
    return thrust::reduce(thrust::hip::par, first, last, init);
#else
    return std::reduce(first, last, init);
#endif
  }

  template <typename ExecutionPolicy, typename ForwardIterator, typename T>
    requires(!alpaka::isQueue<std::remove_cvref_t<ExecutionPolicy>>)
  ALPAKA_FN_HOST constexpr T reduce(ExecutionPolicy&& policy, ForwardIterator first, ForwardIterator last, T init) {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
    return thrust::reduce(std::forward<ExecutionPolicy>(policy), first, last, init);
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
    return thrust::reduce(std::forward<ExecutionPolicy>(policy), first, last, init);
#else
    return std::reduce(std::forward<ExecutionPolicy>(policy), first, last, init);
#endif
  }

  template <typename InputIterator, typename T, typename BinaryOperation>
  ALPAKA_FN_HOST constexpr T reduce(InputIterator first, InputIterator last, T init, BinaryOperation op) {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
    return thrust::reduce(thrust::device, first, last, init, op);
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
    return thrust::reduce(thrust::hip::par, first, last, init, op);
#else
    return std::reduce(first, last, init, op);
#endif
  }

  template <typename ExecutionPolicy, typename ForwardIterator, typename T, typename BinaryOperation>
    requires(!alpaka::isQueue<std::remove_cvref_t<ExecutionPolicy>>)
  ALPAKA_FN_HOST constexpr T reduce(
      ExecutionPolicy&& policy, ForwardIterator first, ForwardIterator last, T init, BinaryOperation op) {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
    return thrust::reduce(std::forward<ExecutionPolicy>(policy), first, last, init, op);
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
    return thrust::reduce(std::forward<ExecutionPolicy>(policy), first, last, init, op);
#else
    return std::reduce(std::forward<ExecutionPolicy>(policy), first, last, init, op);
#endif
  }

  template <concepts::queue TQueue, typename InputIterator>
  ALPAKA_FN_HOST constexpr std::iterator_traits<InputIterator>::value_type reduce(TQueue& queue,
                                                                                  InputIterator first,
                                                                                  InputIterator last) {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
    return thrust::reduce(thrust::cuda::par_nosync.on(queue.getNativeHandle()), first, last);
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
    return thrust::reduce(thrust::hip::par.on(queue.getNativeHandle()), first, last);
#else
    alpaka::wait(queue);
    return std::reduce(first, last);
#endif
  }

  template <concepts::queue TQueue, typename InputIterator, typename T>
  ALPAKA_FN_HOST constexpr T reduce(TQueue& queue, InputIterator first, InputIterator last, T init) {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
    return thrust::reduce(thrust::cuda::par_nosync.on(queue.getNativeHandle()), first, last, init);
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
    return thrust::reduce(thrust::hip::par.on(queue.getNativeHandle()), first, last, init);
#else
    alpaka::wait(queue);
    return std::reduce(first, last, init);
#endif
  }

  template <concepts::queue TQueue, typename InputIterator, typename T, typename BinaryOperation>
  ALPAKA_FN_HOST constexpr T reduce(TQueue& queue, InputIterator first, InputIterator last, T init, BinaryOperation op) {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
    return thrust::reduce(thrust::cuda::par_nosync.on(queue.getNativeHandle()), first, last, init, op);
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
    return thrust::reduce(thrust::hip::par.on(queue.getNativeHandle()), first, last, init, op);
#else
    alpaka::wait(queue);
    return std::reduce(first, last, init, op);
#endif
  }

}  // namespace ffx::algorithm