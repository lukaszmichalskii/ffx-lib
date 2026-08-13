#pragma once

#include "ffx/core/alpaka/devices.h"
#include "ffx/core/mem/alloc/caching_alloc.h"

#include <memory>

namespace ffx::mem {

  namespace detail {

    template <typename TDevice, typename TQueue>
    auto allocate_device_allocators() {
      using Allocator = CachingAllocator<TDevice, TQueue>;
      auto const& devices = ffx::devices<alpaka::Platform<TDevice>>();
      auto const size = devices.size();

      // allocate the storage for the objects
      auto ptr = std::allocator<Allocator>().allocate(size);

      // construct the objects in the storage
      for (size_t index = 0; index < size; ++index) {
        new (ptr + index) Allocator(devices[index],
                                    kBinGrowth,
                                    kMinBin,
                                    kMaxBin,
                                    kMaxCachedBytes,
                                    kMaxCachedFraction,
                                    true,    // reuse_same_queue_allocations
                                    false);  // debug
      }

      // use a custom deleter to destroy all objects and deallocate the memory
      auto deleter = [size](Allocator* pointer) {
        for (size_t i = size; i > 0; --i) {
          (pointer + i - 1)->~Allocator();
        }
        std::allocator<Allocator>().deallocate(pointer, size);
      };

      return std::unique_ptr<Allocator[], decltype(deleter)>(ptr, deleter);
    }

  }  // namespace detail

  template <typename TDevice, typename TQueue>
  CachingAllocator<TDevice, TQueue>& get_device_caching_allocator(TDevice const& device) {
    // initialise all allocators, one per device
    static auto allocators = detail::allocate_device_allocators<TDevice, TQueue>();

    size_t const index = alpaka::getNativeHandle(device);
    std::vector<TDevice> devs = alpaka::getDevs(alpaka::Platform<TDevice>{});
    assert(index < ffx::devices<alpaka::Platform<TDevice>>().size());

    // the public interface is thread safe
    return allocators[index];
  }

}  // namespace ffx::mem
