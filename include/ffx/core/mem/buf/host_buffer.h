#pragma once

#include "ffx/core/alpaka/config.h"
#include "ffx/core/alpaka/host.h"
#include "ffx/core/alpaka/platform.h"
#include "ffx/core/detail/concepts.h"
#include "ffx/core/mem/buf/traits.h"

namespace ffx {

  using namespace ffx_alpaka;

  template <typename T>
  using host_buffer = detail::buffer_type<DevHost, T>::type;

  template <typename T>
  using const_host_buffer = alpaka::ViewConst<host_buffer<T>>;

  // non-cached, non-pinned, scalar and 1-dimensional host buffers

  template <concepts::scalar T>
  host_buffer<T> make_host_buffer() {
    return alpaka::allocBuf<T, Idx>(host(), Scalar{});
  }

  template <concepts::unbounded_array T>
  host_buffer<T> make_host_buffer(Extent extent) {
    return alpaka::allocBuf<std::remove_extent_t<T>, Idx>(host(), Vec1D{extent});
  }

  template <concepts::bounded_array T>
  host_buffer<T> make_host_buffer() {
    return alpaka::allocBuf<std::remove_extent_t<T>, Idx>(host(), Vec1D{std::extent_v<T>});
  }

  // non-cached, pinned, scalar and 1-dimensional host buffers
  // the memory is pinned according to the device associated to the platform

  template <concepts::scalar T, concepts::platform TPlatform>
  host_buffer<T> make_host_buffer() {
    return alpaka::allocMappedBuf<T, Idx>(host(), platform<TPlatform>(), Scalar{});
  }

  template <concepts::unbounded_array T, concepts::platform TPlatform>
  host_buffer<T> make_host_buffer(Extent extent) {
    return alpaka::allocMappedBuf<std::remove_extent_t<T>, Idx>(host(), platform<TPlatform>(), Vec1D{extent});
  }

  template <concepts::bounded_array T, typename TPlatform>
  host_buffer<T> make_host_buffer() {
    return alpaka::allocMappedBuf<std::remove_extent_t<T>, Idx>(host(), platform<TPlatform>(), Vec1D{std::extent_v<T>});
  }

  // potentially cached, pinned, scalar and 1-dimensional host buffers, associated to a work queue
  // the memory is pinned according to the device associated to the queue

  template <concepts::scalar T, concepts::queue TQueue>
  host_buffer<T> make_host_buffer() {
    using TPlatform = alpaka::Platform<alpaka::Dev<TQueue>>;
    return alpaka::allocMappedBuf<T, Idx>(host(), platform<TPlatform>(), Scalar{});
  }

  template <concepts::unbounded_array T, concepts::queue TQueue>
  host_buffer<T> make_host_buffer(Extent extent) {
    using TPlatform = alpaka::Platform<alpaka::Dev<TQueue>>;
    return alpaka::allocMappedBuf<std::remove_extent_t<T>, Idx>(host(), platform<TPlatform>(), Vec1D{extent});
  }

  template <concepts::bounded_array T, concepts::queue TQueue>
  host_buffer<T> make_host_buffer() {
    using TPlatform = alpaka::Platform<alpaka::Dev<TQueue>>;
    return alpaka::allocMappedBuf<std::remove_extent_t<T>, Idx>(host(), platform<TPlatform>(), Vec1D{std::extent_v<T>});
  }

}  // namespace ffx
