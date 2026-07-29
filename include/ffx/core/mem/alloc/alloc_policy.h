#pragma once

#include <alpaka/alpaka.hpp>

namespace ffx {

  // Which memory allocator to use
  //   - Synchronous:   (device and host) cudaMalloc/hipMalloc and cudaMallocHost/hipMallocHost
  //   - Asynchronous:  (device only)     cudaMallocAsync (requires CUDA >= 11.2)
  enum class AllocatorPolicy { Synchronous = 0, Asynchronous = 1 };

  template <typename TDev>
  constexpr inline auto allocator_policy = AllocatorPolicy::Synchronous;

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
  template <>
  constexpr inline auto allocator_policy<alpaka::DevCpu> = AllocatorPolicy::Synchronous;
#endif  // ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
  template <>
  constexpr inline auto allocator_policy<alpaka::DevCudaRt> =
#if CUDA_VERSION >= 11020 && !defined DISABLE_ASYNC_ALLOCATOR
      AllocatorPolicy::Asynchronous;
#else
      AllocatorPolicy::Synchronous;
#endif  // DISABLE_ASYNC_ALLOCATOR
#endif  // ALPAKA_ACC_GPU_CUDA_ENABLED

#ifdef ALPAKA_ACC_GPU_HIP_ENABLED
  template <>
  constexpr inline auto allocator_policy<alpaka::DevHipRt> =
#if !defined DISABLE_ASYNC_ALLOCATOR
      AllocatorPolicy::Asynchronous;
#else
      AllocatorPolicy::Synchronous;
#endif  // DISABLE_ASYNC_ALLOCATOR
#endif  // ALPAKA_ACC_GPU_HIP_ENABLED

}  // namespace ffx
