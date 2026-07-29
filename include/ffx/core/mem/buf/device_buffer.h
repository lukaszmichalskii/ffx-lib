#pragma once

#include "ffx/core/alpaka/config.h"
#include "ffx/core/detail/concepts.h"
#include "ffx/core/mem/buf/traits.h"
#include "ffx/core/mem/alloc/alloc_policy.h"

namespace ffx {

  using namespace ffx_alpaka;

  template <concepts::device TDev, typename T>
  using device_buffer = detail::buffer_type<TDev, T>::type;

  template <concepts::device TDev, typename T>
  using const_device_buffer = alpaka::ViewConst<device_buffer<TDev, T>>;

  // non-cached, scalar and 1-dimensional device buffers

  template <concepts::scalar T, concepts::device TDev>
  device_buffer<TDev, T> make_device_buffer(TDev const& device) {
    return alpaka::allocBuf<T, Idx>(device, Scalar{});
  }

  template <concepts::unbounded_array T, concepts::device TDev>
  device_buffer<TDev, T> make_device_buffer(TDev const& device, Extent extent) {
    return alpaka::allocBuf<std::remove_extent_t<T>, Idx>(device, Vec1D{extent});
  }

  template <concepts::bounded_array T, concepts::device TDev>
  device_buffer<TDev, T> make_device_buffer(TDev const& device) {
    return alpaka::allocBuf<std::remove_extent_t<T>, Idx>(device, Vec1D{std::extent_v<T>});
  }

  // scalar and 1-dimensional device buffers with queue-ordered semantic

  template <concepts::scalar T, concepts::queue TQueue>
  device_buffer<alpaka::Dev<TQueue>, T> make_device_buffer(TQueue const& queue) {
    if constexpr (allocator_policy<alpaka::Dev<TQueue>> == AllocatorPolicy::Asynchronous) {
      return alpaka::allocAsyncBuf<T, Idx>(queue, Scalar{});
    } else if constexpr (allocator_policy<alpaka::Dev<TQueue>> == AllocatorPolicy::Synchronous) {
      return alpaka::allocBuf<T, Idx>(alpaka::getDev(queue), Scalar{});
    } else {
      static_assert(alpaka::meta::DependentFalseType<TQueue>::value, "Unsupported allocator policy");
    }
  }

  template <concepts::unbounded_array T, concepts::queue TQueue>
  device_buffer<alpaka::Dev<TQueue>, T> make_device_buffer(TQueue const& queue, Extent extent) {
    if constexpr (allocator_policy<alpaka::Dev<TQueue>> == AllocatorPolicy::Asynchronous) {
      return alpaka::allocAsyncBuf<std::remove_extent_t<T>, Idx>(queue, Vec1D{extent});
    } else if constexpr (allocator_policy<alpaka::Dev<TQueue>> == AllocatorPolicy::Synchronous) {
      return alpaka::allocBuf<std::remove_extent_t<T>, Idx>(alpaka::getDev(queue), Vec1D{extent});
    } else {
      static_assert(alpaka::meta::DependentFalseType<TQueue>::value, "Unsupported allocator policy");
    }
  }

  template <concepts::bounded_array T, concepts::queue TQueue>
  device_buffer<alpaka::Dev<TQueue>, T> make_device_buffer(TQueue const& queue) {
    if constexpr (allocator_policy<alpaka::Dev<TQueue>> == AllocatorPolicy::Asynchronous) {
      return alpaka::allocAsyncBuf<std::remove_extent_t<T>, Idx>(queue, Vec1D{std::extent_v<T>});
    } else if constexpr (allocator_policy<alpaka::Dev<TQueue>> == AllocatorPolicy::Synchronous) {
      return alpaka::allocBuf<std::remove_extent_t<T>, Idx>(alpaka::getDev(queue), Vec1D{std::extent_v<T>});
    } else {
      static_assert(alpaka::meta::DependentFalseType<TQueue>::value, "Unsupported allocator policy");
    }
  }

}  // namespace ffx
