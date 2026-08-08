#pragma once

#include "ffx/core/detail/concepts.h"
#include "ffx/core/mem/alloc/device_caching_alloc.h"
#include "ffx/core/mem/alloc/host_caching_alloc.h"

namespace ffx::mem {

  namespace traits {

    //! The caching memory allocator trait.
    template <typename TElem,
              typename TDim,
              typename TIdx,
              concepts::device TDev,
              concepts::queue TQueue,
              typename TSfinae = void>
    struct AllocCachedBuf {
      static_assert(alpaka::meta::DependentFalseType<TDev>::value, "This device does not support a caching allocator");
    };

    //! The caching memory allocator implementation for the CPU device
    template <typename TElem, typename TDim, typename TIdx, concepts::queue TQueue>
    struct AllocCachedBuf<TElem, TDim, TIdx, alpaka::DevCpu, TQueue, void> {
      template <typename TExtent>
      ALPAKA_FN_HOST static alpaka::BufCpu<TElem, TDim, TIdx> allocCachedBuf(alpaka::DevCpu const&,
                                                                             TQueue queue,
                                                                             TExtent const& extent) {
        // non-cached host-only memory
        return alpaka::allocAsyncBuf<TElem, TIdx>(queue, extent);
      }
    };

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED

    //! The caching memory allocator implementation for the pinned host memory
    template <typename TElem, typename TDim, typename TIdx>
    struct AllocCachedBuf<TElem, TDim, TIdx, alpaka::DevCpu, alpaka::QueueCudaRtNonBlocking, void> {
      template <typename TExtent>
      ALPAKA_FN_HOST static alpaka::BufCpu<TElem, TDim, TIdx> allocCachedBuf(alpaka::DevCpu const& dev,
                                                                             alpaka::QueueCudaRtNonBlocking queue,
                                                                             TExtent const& extent) {
        auto& allocator = get_host_caching_allocator<alpaka::QueueCudaRtNonBlocking>();

        // FIXME the BufCpu does not support a pitch ?
        size_t size = alpaka::getExtentProduct(extent);
        size_t size_bytes = size * sizeof(TElem);
        void* mem_ptr = allocator.allocate(size_bytes, queue);

        // use a custom deleter to return the buffer to the CachingAllocator
        auto deleter = [alloc = &allocator](TElem* ptr) { alloc->free(ptr); };
        return alpaka::BufCpu<TElem, TDim, TIdx>(dev, static_cast<TElem*>(mem_ptr), std::move(deleter), extent);
      }
    };

    //! The caching memory allocator implementation for the CUDA device
    template <typename TElem, typename TDim, typename TIdx, concepts::queue TQueue>
    struct AllocCachedBuf<TElem, TDim, TIdx, alpaka::DevCudaRt, TQueue, void> {
      template <typename TExtent>
      ALPAKA_FN_HOST static alpaka::BufCudaRt<TElem, TDim, TIdx> allocCachedBuf(alpaka::DevCudaRt const& dev,
                                                                                TQueue queue,
                                                                                TExtent const& extent) {
        auto& allocator = get_device_caching_allocator<alpaka::DevCudaRt, TQueue>(dev);

        size_t width = alpaka::getWidth(extent);
        size_t width_bytes = width * static_cast<TIdx>(sizeof(TElem));

        // TODO implement pitch for TDim > 1
        size_t pitch_bytes = width_bytes;
        size_t size = alpaka::getExtentProduct(extent);
        size_t size_bytes = size * sizeof(TElem);
        void* mem_ptr = allocator.allocate(size_bytes, queue);

        // use a custom deleter to return the buffer to the CachingAllocator
        auto deleter = [alloc = &allocator](TElem* ptr) { alloc->free(ptr); };
        return alpaka::BufCudaRt<TElem, TDim, TIdx>(
            dev, static_cast<TElem*>(mem_ptr), std::move(deleter), extent, pitch_bytes);
      }
    };

#endif  // ALPAKA_ACC_GPU_CUDA_ENABLED

#ifdef ALPAKA_ACC_GPU_HIP_ENABLED

    //! The caching memory allocator implementation for the pinned host memory
    template <typename TElem, typename TDim, typename TIdx>
    struct AllocCachedBuf<TElem, TDim, TIdx, alpaka::DevCpu, alpaka::QueueHipRtNonBlocking, void> {
      template <typename TExtent>
      ALPAKA_FN_HOST static alpaka::BufCpu<TElem, TDim, TIdx> allocCachedBuf(alpaka::DevCpu const& dev,
                                                                             alpaka::QueueHipRtNonBlocking queue,
                                                                             TExtent const& extent) {
        auto& allocator = get_host_caching_allocator<alpaka::QueueHipRtNonBlocking>();

        // FIXME the BufCpu does not support a pitch ?
        size_t size = alpaka::getExtentProduct(extent);
        size_t size_bytes = size * sizeof(TElem);
        void* mem_ptr = allocator.allocate(size_bytes, queue);

        // use a custom deleter to return the buffer to the CachingAllocator
        auto deleter = [alloc = &allocator](TElem* ptr) { alloc->free(ptr); };
        return alpaka::BufCpu<TElem, TDim, TIdx>(dev, reinterpret_cast<TElem*>(mem_ptr), std::move(deleter), extent);
      }
    };

    //! The caching memory allocator implementation for the ROCm/HIP device
    template <typename TElem, typename TDim, typename TIdx, concepts::queue TQueue>
    struct AllocCachedBuf<TElem, TDim, TIdx, alpaka::DevHipRt, TQueue, void> {
      template <typename TExtent>
      ALPAKA_FN_HOST static alpaka::BufHipRt<TElem, TDim, TIdx> allocCachedBuf(alpaka::DevHipRt const& dev,
                                                                               TQueue queue,
                                                                               TExtent const& extent) {
        auto& allocator = get_device_caching_allocator<alpaka::DevHipRt, TQueue>(dev);

        size_t width = alpaka::getWidth(extent);
        size_t width_bytes = width * static_cast<TIdx>(sizeof(TElem));
        // TODO implement pitch for TDim > 1
        size_t pitch_bytes = width_bytes;
        size_t size = alpaka::getExtentProduct(extent);
        size_t size_bytes = size * sizeof(TElem);
        void* mem_ptr = allocator.allocate(size_bytes, queue);

        // use a custom deleter to return the buffer to the CachingAllocator
        auto deleter = [alloc = &allocator](TElem* ptr) { alloc->free(ptr); };
        return alpaka::BufHipRt<TElem, TDim, TIdx>(
            dev, reinterpret_cast<TElem*>(mem_ptr), std::move(deleter), extent, pitch_bytes);
      }
    };

#endif  // ALPAKA_ACC_GPU_HIP_ENABLED

  }  // namespace traits

  template <typename TElem, typename TIdx, typename TExtent, concepts::queue TQueue, concepts::device TDev>
  ALPAKA_FN_HOST auto alloc_cached_buf(TDev const& dev, TQueue queue, TExtent const& extent = TExtent()) {
    return traits::AllocCachedBuf<TElem, alpaka::Dim<TExtent>, TIdx, TDev, TQueue>::allocCachedBuf(dev, queue, extent);
  }

}  // namespace ffx::mem
