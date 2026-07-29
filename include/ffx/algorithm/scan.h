#pragma once

#include <alpaka/alpaka.hpp>

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#else
#include <numeric>
#endif

#include "ffx/core/detail/concepts.h"

namespace ffx::algorithm {

  template <typename InputIterator, typename OutputIterator>
  ALPAKA_FN_HOST constexpr void inclusive_scan(InputIterator first, InputIterator last, OutputIterator output) {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
    thrust::inclusive_scan(thrust::device, first, last, output);
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
    thrust::inclusive_scan(thrust::hip::par, first, last, output);
#else
    std::inclusive_scan(first, last, output);
#endif
  }

  template <typename ExecutionPolicy, typename ForwardIterator>
    requires(!alpaka::isQueue<std::remove_cvref_t<ExecutionPolicy>>)
  ALPAKA_FN_HOST constexpr void inclusive_scan(ExecutionPolicy&& policy,
                                               ForwardIterator first,
                                               ForwardIterator last,
                                               ForwardIterator output) {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
    thrust::inclusive_scan(std::forward<ExecutionPolicy>(policy), first, last, output);
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
    thrust::inclusive_scan(std::forward<ExecutionPolicy>(policy), first, last, output);
#else
    std::inclusive_scan(std::forward<ExecutionPolicy>(policy), first, last, output);
#endif
  }

  template <typename InputIterator, typename OutputIterator, typename BinaryOperator>
  ALPAKA_FN_HOST constexpr void inclusive_scan(InputIterator first,
                                               InputIterator last,
                                               OutputIterator output,
                                               BinaryOperator op) {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
    thrust::inclusive_scan(thrust::device, first, last, output, op);
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
    thrust::inclusive_scan(thrust::hip::par, first, last, output, op);
#else
    std::inclusive_scan(first, last, output, op);
#endif
  }

  template <typename ExecutionPolicy, typename ForwardIterator, typename BinaryOperator>
    requires(!alpaka::isQueue<std::remove_cvref_t<ExecutionPolicy>>)
  ALPAKA_FN_HOST constexpr void inclusive_scan(ExecutionPolicy&& policy,
                                               ForwardIterator first,
                                               ForwardIterator last,
                                               ForwardIterator output,
                                               BinaryOperator op) {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
    thrust::inclusive_scan(std::forward<ExecutionPolicy>(policy), first, last, output, op);
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
    thrust::inclusive_scan(std::forward<ExecutionPolicy>(policy), first, last, output, op);
#else
    std::inclusive_scan(std::forward<ExecutionPolicy>(policy), first, last, output, op);
#endif
  }

  template <typename InputIterator, typename OutputIterator, typename BinaryOperator, typename T>
  ALPAKA_FN_HOST constexpr void inclusive_scan(
      InputIterator first, InputIterator last, OutputIterator output, BinaryOperator op, T init) {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
    thrust::inclusive_scan(thrust::device, first, last, output, init, op);
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
    thrust::inclusive_scan(thrust::hip::par, first, last, output, init, op);
#else
    std::inclusive_scan(first, last, output, op, init);
#endif
  }

  template <typename ExecutionPolicy, typename ForwardIterator, typename BinaryOperator, typename T>
    requires(!alpaka::isQueue<std::remove_cvref_t<ExecutionPolicy>>)
  ALPAKA_FN_HOST constexpr void inclusive_scan(ExecutionPolicy&& policy,
                                               ForwardIterator first,
                                               ForwardIterator last,
                                               ForwardIterator output,
                                               BinaryOperator op,
                                               T init) {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
    thrust::inclusive_scan(std::forward<ExecutionPolicy>(policy), first, last, output, init, op);
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
    thrust::inclusive_scan(std::forward<ExecutionPolicy>(policy), first, last, output, init, op);
#else
    std::inclusive_scan(std::forward<ExecutionPolicy>(policy), first, last, output, op, init);
#endif
  }

  template <typename InputIterator, typename OutputIterator, typename T>
  ALPAKA_FN_HOST constexpr void exclusive_scan(InputIterator first, InputIterator last, OutputIterator output, T init) {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
    thrust::exclusive_scan(thrust::device, first, last, output, init);
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
    thrust::exclusive_scan(thrust::hip::par, first, last, output, init);
#else
    std::exclusive_scan(first, last, output, init);
#endif
  }

  template <typename ExecutionPolicy, typename ForwardIterator, typename T>
    requires(!alpaka::isQueue<std::remove_cvref_t<ExecutionPolicy>>)
  ALPAKA_FN_HOST constexpr void exclusive_scan(
      ExecutionPolicy&& policy, ForwardIterator first, ForwardIterator last, ForwardIterator output, T init) {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
    thrust::exclusive_scan(std::forward<ExecutionPolicy>(policy), first, last, output, init);
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
    thrust::exclusive_scan(std::forward<ExecutionPolicy>(policy), first, last, output, init);
#else
    std::exclusive_scan(std::forward<ExecutionPolicy>(policy), first, last, output, init);
#endif
  }

  template <typename InputIterator, typename OutputIterator, typename T, typename BinaryOperator>
  ALPAKA_FN_HOST constexpr void exclusive_scan(
      InputIterator first, InputIterator last, OutputIterator output, T init, BinaryOperator op) {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
    thrust::exclusive_scan(thrust::device, first, last, output, init, op);
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
    thrust::exclusive_scan(thrust::hip::par, first, last, output, init, op);
#else
    std::exclusive_scan(first, last, output, init, op);
#endif
  }

  template <typename ExecutionPolicy, typename ForwardIterator, typename T, typename BinaryOperator>
    requires(!alpaka::isQueue<std::remove_cvref_t<ExecutionPolicy>>)
  ALPAKA_FN_HOST constexpr void exclusive_scan(ExecutionPolicy&& policy,
                                               ForwardIterator first,
                                               ForwardIterator last,
                                               ForwardIterator output,
                                               T init,
                                               BinaryOperator op) {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
    thrust::exclusive_scan(std::forward<ExecutionPolicy>(policy), first, last, output, init, op);
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
    thrust::exclusive_scan(std::forward<ExecutionPolicy>(policy), first, last, output, init, op);
#else
    std::exclusive_scan(std::forward<ExecutionPolicy>(policy), first, last, output, init, op);
#endif
  }

  template <concepts::queue TQueue, typename InputIterator, typename OutputIterator>
  ALPAKA_FN_HOST constexpr void inclusive_scan(TQueue& queue,
                                               InputIterator first,
                                               InputIterator last,
                                               OutputIterator output) {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
    thrust::inclusive_scan(thrust::cuda::par_nosync.on(queue.getNativeHandle()), first, last, output);
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
    thrust::inclusive_scan(thrust::hip::par.on(queue.getNativeHandle()), first, last, output);
#else
    alpaka::wait(queue);
    std::inclusive_scan(first, last, output);
#endif
  }

  template <concepts::queue TQueue, typename InputIterator, typename OutputIterator, typename BinaryOperator>
  ALPAKA_FN_HOST constexpr void inclusive_scan(
      TQueue& queue, InputIterator first, InputIterator last, OutputIterator output, BinaryOperator op) {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
    thrust::inclusive_scan(thrust::cuda::par_nosync.on(queue.getNativeHandle()), first, last, output, op);
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
    thrust::inclusive_scan(thrust::hip::par.on(queue.getNativeHandle()), first, last, output, op);
#else
    alpaka::wait(queue);
    std::inclusive_scan(first, last, output, op);
#endif
  }

  template <concepts::queue TQueue, typename InputIterator, typename OutputIterator, typename BinaryOperator, typename T>
  ALPAKA_FN_HOST constexpr void inclusive_scan(
      TQueue& queue, InputIterator first, InputIterator last, OutputIterator output, BinaryOperator op, T init) {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
    thrust::inclusive_scan(thrust::cuda::par.on(queue.getNativeHandle()), first, last, output, init, op);
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
    thrust::inclusive_scan(thrust::hip::par_nosync.on(queue.getNativeHandle()), first, last, output, init, op);
#else
    alpaka::wait(queue);
    std::inclusive_scan(first, last, output, op, init);
#endif
  }

  template <concepts::queue TQueue, typename InputIterator, typename OutputIterator, typename T>
  ALPAKA_FN_HOST constexpr void exclusive_scan(
      TQueue& queue, InputIterator first, InputIterator last, OutputIterator output, T init) {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
    thrust::exclusive_scan(thrust::cuda::par_nosync.on(queue.getNativeHandle()), first, last, output, init);
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
    thrust::exclusive_scan(thrust::hip::par.on(queue.getNativeHandle()), first, last, output, init);
#else
    alpaka::wait(queue);
    std::exclusive_scan(first, last, output, init);
#endif
  }

  template <concepts::queue TQueue, typename InputIterator, typename OutputIterator, typename T, typename BinaryOperator>
  ALPAKA_FN_HOST constexpr void exclusive_scan(
      TQueue& queue, InputIterator first, InputIterator last, OutputIterator output, T init, BinaryOperator op) {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
    thrust::exclusive_scan(thrust::cuda::par_nosync.on(queue.getNativeHandle()), first, last, output, init, op);
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
    thrust::exclusive_scan(thrust::hip::par.on(queue.getNativeHandle()), first, last, output, init, op);
#else
    alpaka::wait(queue);
    std::exclusive_scan(first, last, output, init, op);
#endif
  }

}  // namespace ffx::algorithm