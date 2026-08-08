#pragma once

#include "ffx/core/detail/concepts.h"
#include "ffx/core/mem/buf/host_buffer.h"

namespace ffx::soa::detail {

  template <typename THostData>
  struct CopyToDevice;

  template <typename T>
  struct CopyToDevice<alpaka::BufCpu<T, Dim0D, Idx>> {
    template <concepts::queue TQueue>
    static auto copyAsync(TQueue& queue, host_buffer<T> const& src) {
      auto dest = make_device_buffer<T>(queue);
      alpaka::memcpy(queue, dest, src);
      return dest;
    }
  };

  template <typename T>
  struct CopyToDevice<alpaka::BufCpu<T, Dim1D, Idx>> {
    template <concepts::queue TQueue>
    static auto copyAsync(TQueue& queue, host_buffer<T[]> const& src) {
      auto dest = make_device_buffer<T[]>(queue, alpaka::getExtentProduct(src));
      alpaka::memcpy(queue, dest, src);
      return dest;
    }
  };

}  // namespace ffx::soa::detail
