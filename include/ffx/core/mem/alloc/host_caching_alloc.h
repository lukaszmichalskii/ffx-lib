#pragma once

#include "ffx/core/alpaka/config.h"
#include "ffx/core/alpaka/host.h"
#include "ffx/core/detail/concepts.h"
#include "ffx/core/mem/alloc/caching_alloc.h"

namespace ffx::mem {

  template <concepts::queue TQueue>
  CachingAllocator<DevHost, TQueue>& get_host_caching_allocator() {
    static CachingAllocator<DevHost, TQueue> allocator(host(),
                                                       detail::kBinGrowth,
                                                       detail::kMinBin,
                                                       detail::kMaxBin,
                                                       detail::kMaxCachedBytes,
                                                       detail::kMaxCachedFraction,
                                                       false,   // reuseSameQueueAllocations
                                                       false);  // debug

    // the public interface is thread safe
    return allocator;
  }

}  // namespace ffx::mem