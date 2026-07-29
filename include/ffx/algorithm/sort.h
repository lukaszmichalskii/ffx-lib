#pragma once

#include <alpaka/alpaka.hpp>

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#else
#include <algorithm>
#endif

#include "ffx/core/detail/concepts.h"

namespace ffx::algorithm {

  template <typename RandomAccessIterator>
  ALPAKA_FN_HOST constexpr void sort(RandomAccessIterator first, RandomAccessIterator last) {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
    thrust::sort(thrust::device, first, last);
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
    thrust::sort(thrust::hip::par, first, last);
#else
    std::sort(first, last);
#endif
  }

  template <typename ExecutionPolicy, typename RandomAccessIterator>
    requires(!alpaka::isQueue<std::remove_cvref_t<ExecutionPolicy>>)
  ALPAKA_FN_HOST constexpr void sort(ExecutionPolicy&& policy, RandomAccessIterator first, RandomAccessIterator last) {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
    thrust::sort(std::forward<ExecutionPolicy>(policy), first, last);
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
    thrust::sort(std::forward<ExecutionPolicy>(policy), first, last);
#else
    std::sort(std::forward<ExecutionPolicy>(policy), first, last);
#endif
  }

  template <typename RandomAccessIterator, typename Compare>
  ALPAKA_FN_HOST constexpr void sort(RandomAccessIterator first, RandomAccessIterator last, Compare comp) {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
    thrust::sort(thrust::device, first, last, comp);
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
    thrust::sort(thrust::hip::par, first, last, comp);
#else
    std::sort(first, last, comp);
#endif
  }

  template <typename ExecutionPolicy, typename RandomAccessIterator, typename Compare>
    requires(!alpaka::isQueue<std::remove_cvref_t<ExecutionPolicy>>)
  ALPAKA_FN_HOST constexpr void sort(ExecutionPolicy&& policy,
                                     RandomAccessIterator first,
                                     RandomAccessIterator last,
                                     Compare comp) {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
    thrust::sort(std::forward<ExecutionPolicy>(policy), first, last, comp);
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
    thrust::sort(std::forward<ExecutionPolicy>(policy), first, last, comp);
#else
    std::sort(std::forward<ExecutionPolicy>(policy), first, last, comp);
#endif
  }

  template <concepts::queue TQueue, typename RandomAccessIterator>
  ALPAKA_FN_HOST constexpr void sort(TQueue& queue, RandomAccessIterator first, RandomAccessIterator last) {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
    thrust::sort(thrust::cuda::par_nosync.on(queue.getNativeHandle()), first, last);
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
    thrust::sort(thrust::hip::par.on(queue.getNativeHandle()), first, last);
#else
    alpaka::wait(queue);
    std::sort(first, last);
#endif
  }

  template <concepts::queue TQueue, typename RandomAccessIterator, typename Compare>
  ALPAKA_FN_HOST constexpr void sort(TQueue& queue,
                                     RandomAccessIterator first,
                                     RandomAccessIterator last,
                                     Compare comp) {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
    thrust::sort(thrust::cuda::par_nosync.on(queue.getNativeHandle()), first, last, comp);
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
    thrust::sort(thrust::hip::par.on(queue.getNativeHandle()), first, last, comp);
#else
    alpaka::wait(queue);
    std::sort(first, last, comp);
#endif
  }

}  // namespace ffx::algorithm