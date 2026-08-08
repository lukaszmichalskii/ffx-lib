#pragma once

#include <alpaka/alpaka.hpp>

namespace ffx {

  // Which memory allocator to use
  //   - Synchronous:   (device and host) cudaMalloc/hipMalloc and cudaMallocHost/hipMallocHost
  //   - Asynchronous:  (device only)     cudaMallocAsync (requires CUDA >= 11.2)
  //   - Caching:       (device and host) caching allocator
  enum class AllocatorPolicy { Synchronous = 0, Asynchronous = 1, Caching = 2 };

  template <typename TDev>
  constexpr inline auto allocator_policy = AllocatorPolicy::Synchronous;

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
  template <>
  constexpr inline auto allocator_policy<alpaka::DevCudaRt> =
#ifdef FFX_CACHING_ALLOC_ENABLED
      AllocatorPolicy::Caching;
#elif CUDA_VERSION >= 11020 && !defined DISABLE_ASYNC_ALLOCATOR
      AllocatorPolicy::Asynchronous;
#else
      AllocatorPolicy::Synchronous;
#endif  // DISABLE_ASYNC_ALLOCATOR
#endif  // ALPAKA_ACC_GPU_CUDA_ENABLED

#ifdef ALPAKA_ACC_GPU_HIP_ENABLED
  template <>
  constexpr inline auto allocator_policy<alpaka::DevHipRt> =
#ifdef FFX_CACHING_ALLOC_ENABLED
      AllocatorPolicy::Caching;
#elif !defined DISABLE_ASYNC_ALLOCATOR
      AllocatorPolicy::Asynchronous;
#else
      AllocatorPolicy::Synchronous;
#endif  // DISABLE_ASYNC_ALLOCATOR
#endif  // ALPAKA_ACC_GPU_HIP_ENABLED

}  // namespace ffx
