#pragma once

#include <alpaka/alpaka.hpp>

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>
#else
#include <algorithm>
#endif

#include "ffx/core/detail/concepts.h"

namespace ffx::algorithm {

  template <typename ForwardIterator>
  ALPAKA_FN_HOST constexpr ForwardIterator min_element(ForwardIterator first, ForwardIterator last) {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
    return thrust::min_element(thrust::device, first, last);
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
    return thrust::min_element(thrust::hip::par, first, last);
#else
    return std::min_element(first, last);
#endif
  }

  template <typename ExecutionPolicy, typename ForwardIterator>
    requires(!alpaka::isQueue<std::remove_cvref_t<ExecutionPolicy>>)
  ALPAKA_FN_HOST constexpr ForwardIterator min_element(ExecutionPolicy&& policy,
                                                       ForwardIterator first,
                                                       ForwardIterator last) {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
    return thrust::min_element(std::forward<ExecutionPolicy>(policy), first, last);
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
    return thrust::min_element(std::forward<ExecutionPolicy>(policy), first, last);
#else
    return std::min_element(std::forward<ExecutionPolicy>(policy), first, last);
#endif
  }

  template <typename ForwardIterator, typename BinaryPredicate>
  ALPAKA_FN_HOST constexpr ForwardIterator min_element(ForwardIterator first,
                                                       ForwardIterator last,
                                                       BinaryPredicate comp) {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
    return thrust::min_element(thrust::device, first, last, comp);
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
    return thrust::min_element(thrust::hip::par, first, last, comp);
#else
    return std::min_element(first, last, comp);
#endif
  }

  template <typename ExecutionPolicy, typename ForwardIterator, typename BinaryPredicate>
    requires(!alpaka::isQueue<std::remove_cvref_t<ExecutionPolicy>>)
  ALPAKA_FN_HOST constexpr ForwardIterator min_element(ExecutionPolicy&& policy,
                                                       ForwardIterator first,
                                                       ForwardIterator last,
                                                       BinaryPredicate comp) {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
    return thrust::min_element(std::forward<ExecutionPolicy>(policy), first, last, comp);
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
    return thrust::min_element(std::forward<ExecutionPolicy>(policy), first, last, comp);
#else
    return std::min_element(std::forward<ExecutionPolicy>(policy), first, last, comp);
#endif
  }

  template <typename ForwardIterator>
  ALPAKA_FN_HOST constexpr ForwardIterator max_element(ForwardIterator first, ForwardIterator last) {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
    return thrust::max_element(thrust::device, first, last);
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
    return thrust::max_element(thrust::hip::par, first, last);
#else
    return std::max_element(first, last);
#endif
  }

  template <typename ExecutionPolicy, typename ForwardIterator>
    requires(!alpaka::isQueue<std::remove_cvref_t<ExecutionPolicy>>)
  ALPAKA_FN_HOST constexpr ForwardIterator max_element(ExecutionPolicy&& policy,
                                                       ForwardIterator first,
                                                       ForwardIterator last) {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
    return thrust::max_element(std::forward<ExecutionPolicy>(policy), first, last);
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
    return thrust::max_element(std::forward<ExecutionPolicy>(policy), first, last);
#else
    return std::max_element(std::forward<ExecutionPolicy>(policy), first, last);
#endif
  }

  template <typename ForwardIterator, typename BinaryPredicate>
  ALPAKA_FN_HOST constexpr ForwardIterator max_element(ForwardIterator first,
                                                       ForwardIterator last,
                                                       BinaryPredicate comp) {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
    return thrust::max_element(thrust::device, first, last, comp);
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
    return thrust::max_element(thrust::hip::par, first, last, comp);
#else
    return std::max_element(first, last, comp);
#endif
  }

  template <typename ExecutionPolicy, typename ForwardIterator, typename BinaryPredicate>
    requires(!alpaka::isQueue<std::remove_cvref_t<ExecutionPolicy>>)
  ALPAKA_FN_HOST constexpr ForwardIterator max_element(ExecutionPolicy&& policy,
                                                       ForwardIterator first,
                                                       ForwardIterator last,
                                                       BinaryPredicate comp) {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
    return thrust::max_element(std::forward<ExecutionPolicy>(policy), first, last, comp);
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
    return thrust::max_element(std::forward<ExecutionPolicy>(policy), first, last, comp);
#else
    return std::max_element(std::forward<ExecutionPolicy>(policy), first, last, comp);
#endif
  }

  template <typename ForwardIterator>
  ALPAKA_FN_HOST constexpr std::pair<ForwardIterator, ForwardIterator> minmax_element(ForwardIterator first,
                                                                                      ForwardIterator last) {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
    return thrust::minmax_element(thrust::device, first, last);
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
    auto res = thrust::minmax_element(thrust::hip::par, first, last);
    return {res.first, res.second};
#else
    return std::minmax_element(first, last);
#endif
  }

  template <typename ExecutionPolicy, typename ForwardIterator>
    requires(!alpaka::isQueue<std::remove_cvref_t<ExecutionPolicy>>)
  ALPAKA_FN_HOST constexpr std::pair<ForwardIterator, ForwardIterator> minmax_element(ExecutionPolicy&& policy,
                                                                                      ForwardIterator first,
                                                                                      ForwardIterator last) {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
    return thrust::minmax_element(std::forward<ExecutionPolicy>(policy), first, last);
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
    auto res = thrust::minmax_element(std::forward<ExecutionPolicy>(policy), first, last);
    return {res.first, res.second};
#else
    return std::minmax_element(std::forward<ExecutionPolicy>(policy), first, last);
#endif
  }

  template <typename ForwardIterator, typename BinaryPredicate>
  ALPAKA_FN_HOST constexpr std::pair<ForwardIterator, ForwardIterator> minmax_element(ForwardIterator first,
                                                                                      ForwardIterator last,
                                                                                      BinaryPredicate comp) {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
    return thrust::minmax_element(thrust::device, first, last, comp);
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
    const auto res = thrust::minmax_element(thrust::hip::par, first, last, comp);
    return {res.first, res.second};
#else
    return std::minmax_element(first, last, comp);
#endif
  }

  template <typename ExecutionPolicy, typename ForwardIterator, typename BinaryPredicate>
    requires(!alpaka::isQueue<std::remove_cvref_t<ExecutionPolicy>>)
  ALPAKA_FN_HOST constexpr std::pair<ForwardIterator, ForwardIterator> minmax_element(ExecutionPolicy&& policy,
                                                                                      ForwardIterator first,
                                                                                      ForwardIterator last,
                                                                                      BinaryPredicate comp) {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
    return thrust::minmax_element(std::forward<ExecutionPolicy>(policy), first, last, comp);
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
    const auto res = thrust::minmax_element(std::forward<ExecutionPolicy>(policy), first, last, comp);
    return {res.first, res.second};
#else
    return std::minmax_element(std::forward<ExecutionPolicy>(policy), first, last, comp);
#endif
  }

  template <concepts::queue TQueue, typename ForwardIterator>
  ALPAKA_FN_HOST constexpr ForwardIterator min_element(TQueue& queue, ForwardIterator first, ForwardIterator last) {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
    return thrust::min_element(thrust::cuda::par_nosync.on(queue.getNativeHandle()), first, last);
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
    return thrust::min_element(thrust::hip::par.on(queue.getNativeHandle()), first, last);
#else
    alpaka::wait(queue);
    return std::min_element(first, last);
#endif
  }

  template <concepts::queue TQueue, typename ForwardIterator, typename BinaryPredicate>
  ALPAKA_FN_HOST constexpr ForwardIterator min_element(TQueue& queue,
                                                       ForwardIterator first,
                                                       ForwardIterator last,
                                                       BinaryPredicate comp) {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
    return thrust::min_element(thrust::cuda::par_nosync.on(queue.getNativeHandle()), first, last, comp);
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
    return thrust::min_element(thrust::hip::par.on(queue.getNativeHandle()), first, last, comp);
#else
    alpaka::wait(queue);
    return std::min_element(first, last, comp);
#endif
  }

  template <concepts::queue TQueue, typename ForwardIterator>
  ALPAKA_FN_HOST constexpr ForwardIterator max_element(TQueue& queue, ForwardIterator first, ForwardIterator last) {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
    return thrust::max_element(thrust::cuda::par_nosync.on(queue.getNativeHandle()), first, last);
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
    return thrust::max_element(thrust::hip::par.on(queue.getNativeHandle()), first, last);
#elif defined(ALPAKA_ACC_SYCL_ENABLED)
    return oneapi::dpl::max_element(oneapi::dpl::execution::dpcpp_default, first, last);
#else
    alpaka::wait(queue);
    return std::max_element(first, last);
#endif
  }

  template <concepts::queue TQueue, typename ForwardIterator, typename BinaryPredicate>
  ALPAKA_FN_HOST constexpr ForwardIterator max_element(TQueue& queue,
                                                       ForwardIterator first,
                                                       ForwardIterator last,
                                                       BinaryPredicate comp) {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
    return thrust::max_element(thrust::cuda::par_nosync.on(queue.getNativeHandle()), first, last, comp);
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
    return thrust::max_element(thrust::hip::par.on(queue.getNativeHandle()), first, last, comp);
#else
    alpaka::wait(queue);
    return std::max_element(first, last, comp);
#endif
  }

  template <concepts::queue TQueue, typename ForwardIterator>
  ALPAKA_FN_HOST constexpr std::pair<ForwardIterator, ForwardIterator> minmax_element(TQueue& queue,
                                                                                      ForwardIterator first,
                                                                                      ForwardIterator last) {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
    return thrust::minmax_element(thrust::cuda::par_nosync.on(queue.getNativeHandle()), first, last);
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
    const auto res = thrust::minmax_element(thrust::hip::par.on(queue.getNativeHandle()), first, last);
    return {res.first, res.second};
#else
    alpaka::wait(queue);
    return std::minmax_element(first, last);
#endif
  }

  template <concepts::queue TQueue, typename ForwardIterator, typename BinaryPredicate>
  ALPAKA_FN_HOST constexpr std::pair<ForwardIterator, ForwardIterator> minmax_element(TQueue& queue,
                                                                                      ForwardIterator first,
                                                                                      ForwardIterator last,
                                                                                      BinaryPredicate comp) {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
    return thrust::minmax_element(thrust::cuda::par_nosync.on(queue.getNativeHandle()), first, last, comp);
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
    const auto res = thrust::minmax_element(thrust::hip::par.on(queue.getNativeHandle()), first, last, comp);
    return {res.first, res.second};
#else
    alpaka::wait(queue);
    return std::minmax_element(first, last, comp);
#endif
  }

}  // namespace ffx::algorithm